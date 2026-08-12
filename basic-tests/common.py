from __future__ import annotations

import json
import math
import os
import platform
import random
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DATASET_REPO = "bdsaglam/musique"
DATASET_CONFIG = "answerable"
DATASET_SPLIT = "validation"

SYSTEM_HOP = (
    "You are solving a document-grounded multi-hop QA benchmark. "
    "Use only the supplied documents. Return only the shortest factual answer span. "
    "Do not explain."
)

SYSTEM_FINAL = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Return only the shortest factual answer span to the main question. Do not explain."
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, requested: str = "auto"):
    if requested != "auto":
        table = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if requested.lower() not in table:
            raise ValueError(f"Unknown dtype: {requested}")
        return table[requested.lower()]

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        # FP16 is the conservative default for broad MPS compatibility.
        return torch.float16
    return torch.float32


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def memory_status(device: torch.device) -> dict:
    result = {
        "ram_rss_gb": psutil.Process(os.getpid()).memory_info().rss / 1e9,
    }
    if device.type == "cuda":
        result.update({
            "accelerator_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "accelerator_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "accelerator_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        })
    elif device.type == "mps":
        try:
            result["accelerator_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
        except Exception:
            pass
        try:
            result["accelerator_driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 1e9
        except Exception:
            pass
        try:
            result["accelerator_recommended_max_gb"] = torch.mps.recommended_max_memory() / 1e9
        except Exception:
            pass
    return result


def clear_accelerator_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def environment_metadata(model_name: str, device: torch.device, dtype) -> dict:
    return {
        "model": model_name,
        "device": str(device),
        "dtype": dtype_name(dtype),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "memory": memory_status(device),
    }


def load_subject_model(
    model_name: str = DEFAULT_MODEL,
    device_name: str = "auto",
    dtype_name_arg: str = "auto",
):
    device = resolve_device(device_name)
    dtype = resolve_dtype(device, dtype_name_arg)

    print(f"Loading {model_name}")
    print(f"Device: {device} | dtype: {dtype_name(dtype)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device, dtype


def load_musique():
    ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split=DATASET_SPLIT)
    if "answerable" in ds.column_names:
        ds = ds.filter(lambda x: bool(x["answerable"]))
    return ds


def balanced_selection(ds, n_questions: int, seed: int, allowed_hops=(3, 4)) -> list[int]:
    rng = random.Random(seed)
    buckets = {h: [] for h in allowed_hops}
    for idx, ex in enumerate(ds):
        h = len(ex["question_decomposition"])
        if h in buckets:
            buckets[h].append(idx)
    for h in buckets:
        rng.shuffle(buckets[h])

    chosen = []
    while len(chosen) < n_questions and any(buckets.values()):
        for h in allowed_hops:
            if buckets[h] and len(chosen) < n_questions:
                chosen.append(buckets[h].pop())
    rng.shuffle(chosen)
    return chosen


def load_selection(path: str | Path) -> list[int]:
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict):
        return [int(x) for x in obj["dataset_indices"]]
    return [int(x) for x in obj]


def save_selection(path: str | Path, ds, indices: Sequence[int], seed: int) -> None:
    payload = {
        "seed": seed,
        "dataset_indices": [int(i) for i in indices],
        "question_ids": [str(ds[i]["id"]) for i in indices],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def context_block(ex) -> str:
    # Keep every MuSiQue paragraph in the base experiment.
    blocks = []
    for p in ex["paragraphs"]:
        title = p.get("title", "")
        idx = p.get("idx", "?")
        text = p.get("paragraph_text", "")
        blocks.append(f"[Document {idx}] {title}\n{text}")
    return "\n\n".join(blocks)


def resolve_refs(template_question: str, prior_answers: Sequence[str]) -> str:
    q = str(template_question)
    for j, ans in enumerate(prior_answers, start=1):
        q = q.replace(f"#{j}", str(ans))
    return q


def gold_resolved_hops(ex) -> list[dict]:
    golds = []
    rows = []
    for i, step in enumerate(ex["question_decomposition"]):
        q = resolve_refs(step["question"], golds)
        gold = str(step["answer"])
        rows.append({
            "hop_idx": i,
            "hop": i + 1,
            "template_question": str(step["question"]),
            "resolved_question": q,
            "gold": gold,
        })
        golds.append(gold)
    return rows


def _apply_chat_template(tokenizer, messages) -> str:
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def hop_prompt(tokenizer, ctx: str, subq: str) -> str:
    user = (
        f"DOCUMENTS:\n{ctx}\n\n"
        f"SUB-QUESTION:\n{subq}\n\n"
        "SHORT ANSWER:"
    )
    return _apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_HOP},
            {"role": "user", "content": user},
        ],
    )


def final_prompt(tokenizer, ctx: str, main_q: str, pairs: Sequence[tuple[str, str]]) -> str:
    chain = "\n".join(f"- {q} -> {a}" for q, a in pairs)
    user = (
        f"DOCUMENTS:\n{ctx}\n\n"
        f"STRUCTURED REASONING STATE:\n{chain}\n\n"
        f"MAIN QUESTION:\n{main_q}\n\n"
        "SHORT ANSWER:"
    )
    return _apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_FINAL},
            {"role": "user", "content": user},
        ],
    )


def clean_generated_answer(text: str) -> str:
    text = str(text).strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    text = lines[0] if lines else ""
    text = re.sub(r"^(answer|short answer)\s*:\s*", "", text, flags=re.I)
    return text.strip("`\"' ")


def normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("–", "-").replace("—", "-").replace("-", " ")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(p)
    recall = n / len(g)
    return 2 * precision * recall / (precision + recall)


def phrase_contains(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return False
    return re.search(rf"(^|\s){re.escape(g)}($|\s)", p) is not None


def grade_answer(
    pred: str,
    gold: str,
    aliases: Sequence[str] | None = None,
    f1_correct: float = 0.80,
    f1_review: float = 0.35,
):
    candidates = [str(gold)] + [str(x) for x in (aliases or []) if str(x).strip()]
    best_f1 = -1.0
    best_gold = str(gold)

    for g in candidates:
        f1 = token_f1(pred, g)
        if f1 > best_f1:
            best_f1, best_gold = f1, g

        if normalize(pred) == normalize(g) or phrase_contains(pred, g) or f1 >= f1_correct:
            return "correct", best_f1, best_gold

    if best_f1 < f1_review:
        return "incorrect", best_f1, best_gold

    return "needs_review", best_f1, best_gold


def _iter_batches(items: Sequence, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i:i + batch_size]


@torch.inference_mode()
def batched_generate_with_confidence(
    model,
    tokenizer,
    device: torch.device,
    prompts: Sequence[str],
    batch_size: int = 4,
    max_new_tokens: int = 16,
    progress_label: str = "generate",
    checkpoint_callback=None,
):
    """
    Batched greedy generation plus four white-box confidence signals:
    mean token logprob, minimum token logprob, mean entropy, top1-top2 margin.

    Qwen3 is run in non-thinking mode by the prompt-building functions.
    """
    tokenizer.padding_side = "left"
    results = [None] * len(prompts)

    for batch_num, (start, batch_prompts) in enumerate(_iter_batches(prompts, batch_size), start=1):
        t0 = time.perf_counter()
        enc = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_width = enc["input_ids"].shape[1]

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        for b in range(len(batch_prompts)):
            gen_ids = out.sequences[b, input_width:]
            kept_ids = []
            token_logps = []
            entropies = []
            margins = []

            for t, logits_batch in enumerate(out.scores):
                if t >= len(gen_ids):
                    break
                token_id = int(gen_ids[t].item())
                if token_id == tokenizer.eos_token_id:
                    break
                if tokenizer.pad_token_id is not None and token_id == tokenizer.pad_token_id:
                    break

                logits = logits_batch[b].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                token_logps.append(float(log_probs[token_id].item()))

                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * log_probs).sum()
                entropies.append(float(ent.item()))

                top2 = torch.topk(logits, 2).values
                margins.append(float((top2[0] - top2[1]).item()))
                kept_ids.append(token_id)

            raw = tokenizer.decode(kept_ids, skip_special_tokens=True)
            results[start + b] = {
                "answer": clean_generated_answer(raw),
                "raw_answer": raw,
                "mean_logprob": float(np.mean(token_logps)) if token_logps else np.nan,
                "min_logprob": float(np.min(token_logps)) if token_logps else np.nan,
                "entropy": float(np.mean(entropies)) if entropies else np.nan,
                "margin": float(np.mean(margins)) if margins else np.nan,
                "generated_tokens": int(len(kept_ids)),
                "prompt_tokens": int(enc["attention_mask"][b].sum().item()),
            }

        elapsed = time.perf_counter() - t0
        done = min(start + len(batch_prompts), len(prompts))
        print(
            f"{progress_label}: {done}/{len(prompts)} | "
            f"batch={len(batch_prompts)} | {elapsed:.2f}s | "
            f"memory={memory_status(device)}",
            flush=True,
        )

        if checkpoint_callback is not None:
            checkpoint_callback(done, results)

        # Release large generation logits promptly.
        del out, enc
        clear_accelerator_cache(device)

    return results


@torch.inference_mode()
def batched_target_logprob(
    model,
    tokenizer,
    device: torch.device,
    prompts: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 2,
    progress_label: str = "score",
):
    """
    Average token log P(target | prompt), teacher-forced.

    Efficiency trick:
    - full sequences are left-padded so every target ends at the same position;
    - Qwen3's `logits_to_keep` computes LM-head logits only for the last
      max_target_len + 1 positions rather than for the entire long context.
    """
    if len(prompts) != len(targets):
        raise ValueError("prompts and targets must have equal length")

    tokenizer.padding_side = "left"
    outputs_all = [None] * len(prompts)

    for _, (start, batch_prompts) in enumerate(_iter_batches(prompts, batch_size), start=1):
        batch_targets = list(targets[start:start + len(batch_prompts)])

        prompt_ids = [
            tokenizer.encode(p, add_special_tokens=False)
            for p in batch_prompts
        ]
        target_ids = [
            tokenizer.encode(t, add_special_tokens=False)
            for t in batch_targets
        ]

        # Empty target is not a meaningful score.
        if any(len(t) == 0 for t in target_ids):
            raise ValueError("Encountered an empty target tokenization")

        full_ids = [p + t for p, t in zip(prompt_ids, target_ids)]
        max_len = max(len(x) for x in full_ids)
        max_target = max(len(x) for x in target_ids)
        keep = max_target + 1
        pad_id = tokenizer.pad_token_id

        padded = []
        masks = []
        for ids in full_ids:
            pad_n = max_len - len(ids)
            padded.append([pad_id] * pad_n + ids)
            masks.append([0] * pad_n + [1] * len(ids))

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

        t0 = time.perf_counter()
        model_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=keep,
        )
        logits = model_out.logits

        for b, tgt in enumerate(target_ids):
            L = len(tgt)
            # Kept logits correspond to absolute positions [max_len-keep, ..., max_len-1].
            # First target token sits at absolute position max_len-L and is predicted
            # by the preceding position max_len-L-1.
            rel_start = keep - L - 1
            pred_logits = logits[b, rel_start:rel_start + L, :].float()
            labels = torch.tensor(tgt, dtype=torch.long, device=device)
            lp = torch.log_softmax(pred_logits, dim=-1)
            vals = lp.gather(1, labels.unsqueeze(1)).squeeze(1)
            outputs_all[start + b] = {
                "mean_logprob": float(vals.mean().item()),
                "sum_logprob": float(vals.sum().item()),
                "target_tokens": int(L),
                "prompt_tokens": int(len(prompt_ids[b])),
            }

        done = min(start + len(batch_prompts), len(prompts))
        print(
            f"{progress_label}: {done}/{len(prompts)} | "
            f"batch={len(batch_prompts)} | {time.perf_counter()-t0:.2f}s | "
            f"memory={memory_status(device)}",
            flush=True,
        )

        del model_out, logits, input_ids, attention_mask
        clear_accelerator_cache(device)

    return outputs_all


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_auroc_auprc(uncertainty: Sequence[float], error_labels: Sequence[int]) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score

    x = np.asarray(uncertainty, dtype=float)
    y = np.asarray(error_labels, dtype=int)
    mask = np.isfinite(x)
    x, y = x[mask], y[mask]

    if len(x) == 0 or len(np.unique(y)) < 2:
        return {"n": int(len(x)), "auroc": None, "auprc": None}

    return {
        "n": int(len(x)),
        "auroc": float(roc_auc_score(y, x)),
        "auprc": float(average_precision_score(y, x)),
    }


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
