#!/usr/bin/env python3
from __future__ import annotations

"""
06_sequential_confidence.py

Sequential / within-trace confidence experiment for MuSiQue.

Purpose
-------
Keep the existing isolated-hop Test 1 as a baseline, but add a more realistic
condition where each hop is still queried individually while references are
resolved with the MODEL'S previous answers.

For each hop we record:
  - benchmark correctness label
  - mean token log-probability
  - minimum token log-probability
  - entropy
  - top1-top2 logit margin
  - separately verbalized confidence (0-100)

Then we group H1/H2/H3(/H4) within each question and compute:
  - raw uncertainty
  - within-trace centered uncertainty
  - within-trace z-score
  - within-trace min-max uncertainty
  - within-trace rank (1 = most uncertain)

We report both:
  A) pooled error-detection AUROC/AUPRC across hops, before and after
     within-trace normalization; and
  B) within-trace localization: does the most uncertain hop identify the
     first clearly incorrect hop / any clearly incorrect hop?

Important interpretation
------------------------
A monotone within-question normalization (center/z/minmax) does NOT change
which hop ranks first within that question. It can, however, change pooled
cross-question metrics by removing question-level difficulty offsets.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    batched_generate_with_confidence,
    batched_target_logprob,
    context_block,
    environment_metadata,
    final_prompt,
    grade_answer,
    hop_prompt,
    load_musique,
    load_selection,
    load_subject_model,
    normalize,
    resolve_refs,
    safe_auroc_auprc,
    seed_everything,
    write_json,
    write_jsonl,
)


SIGNALS = {
    # name: (source column, sign converting raw metric -> uncertainty)
    "mean_logprob": ("mean_logprob", -1.0),
    "min_logprob": ("min_logprob", -1.0),
    "entropy": ("entropy", +1.0),
    "margin": ("margin", -1.0),
    "verbal": ("verbal_confidence", -1.0),  # lower verbal confidence = more uncertain
}


def apply_chat_template(tok, messages) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def verbal_confidence_prompt(tok, ctx: str, subq: str, answer: str) -> str:
    """Separate confidence query so eliciting confidence does not alter the answer."""
    system = (
        "You are evaluating your own short answer to a document-grounded question. "
        "Estimate the probability that the supplied answer is correct given the documents. "
        "Return ONLY one integer from 0 to 100. Do not explain."
    )
    user = (
        f"DOCUMENTS:\n{ctx}\n\n"
        f"SUB-QUESTION:\n{subq}\n\n"
        f"YOUR ANSWER:\n{answer}\n\n"
        "CONFIDENCE (0-100):"
    )
    return apply_chat_template(
        tok,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )


def parse_confidence(text: str) -> float | None:
    """Parse the first 0-100 number from verbalized-confidence output."""
    m = re.search(r"(?<!\d)(100|\d{1,2})(?:\.\d+)?(?!\d)", str(text))
    if not m:
        return None
    try:
        x = float(m.group(0))
    except ValueError:
        return None
    return x if 0.0 <= x <= 100.0 else None


def safe_one_target_logprob(model, tok, device, prompt: str, target: str, label: str) -> float:
    """Use batch size 1 for the MPS-safe teacher-forced score."""
    out = batched_target_logprob(
        model,
        tok,
        device,
        [prompt],
        [target],
        batch_size=1,
        progress_label=label,
    )[0]
    value = float(out["mean_logprob"])
    if not np.isfinite(value):
        raise RuntimeError(f"Non-finite target logprob in {label}")
    return value


def add_within_trace_normalization(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    for signal_name, (col, sign) in SIGNALS.items():
        ucol = f"u_{signal_name}"
        d[ucol] = sign * pd.to_numeric(d[col], errors="coerce")

        centered_col = f"{ucol}_centered"
        z_col = f"{ucol}_z"
        minmax_col = f"{ucol}_minmax"
        rank_col = f"{ucol}_rank"

        d[centered_col] = np.nan
        d[z_col] = np.nan
        d[minmax_col] = np.nan
        d[rank_col] = np.nan

        for _, idx in d.groupby("question_id").groups.items():
            idx = list(idx)
            vals = d.loc[idx, ucol].astype(float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                continue

            mean = float(finite.mean())
            sd = float(finite.std(ddof=0))
            lo = float(finite.min())
            hi = float(finite.max())

            d.loc[idx, centered_col] = vals - mean
            if sd > 1e-12:
                d.loc[idx, z_col] = (vals - mean) / sd
            else:
                d.loc[idx, z_col] = 0.0

            if hi - lo > 1e-12:
                d.loc[idx, minmax_col] = (vals - lo) / (hi - lo)
            else:
                d.loc[idx, minmax_col] = 0.0

            # 1 = most uncertain. Rank only finite values.
            finite_idx = vals[np.isfinite(vals)].index
            d.loc[finite_idx, rank_col] = (
                vals.loc[finite_idx].rank(method="average", ascending=False).astype(float)
            )

    return d


def detection_metrics(hops: pd.DataFrame, uncertainty_col: str) -> dict:
    scored = hops[hops["label"].isin(["correct", "incorrect"])].copy()
    y = (scored["label"] == "incorrect").astype(int).to_numpy()
    x = pd.to_numeric(scored[uncertainty_col], errors="coerce").to_numpy(dtype=float)
    return safe_auroc_auprc(x, y)


def within_trace_localization(hops: pd.DataFrame, uncertainty_col: str) -> dict:
    """
    Evaluate ranking INSIDE each question.

    first_error_hit: most uncertain hop equals earliest clearly incorrect hop.
    any_error_hit: most uncertain hop is any clearly incorrect hop.
    pairwise_wrong_vs_correct: within-question wrong/correct pairs are ordered correctly.
    """
    first_hits = []
    any_hits = []
    random_first = []
    random_any = []
    pair_hits = []
    n_with_error = 0
    n_with_both_classes = 0

    for _, g0 in hops.groupby("question_id"):
        g = g0[
            g0["label"].isin(["correct", "incorrect"])
            & np.isfinite(pd.to_numeric(g0[uncertainty_col], errors="coerce"))
        ].copy()
        if len(g) == 0:
            continue

        wrong = g[g["label"] == "incorrect"]
        if len(wrong) == 0:
            continue
        n_with_error += 1

        selected_idx = pd.to_numeric(g[uncertainty_col], errors="coerce").idxmax()
        first_error_hop = int(wrong["hop"].min())
        selected_hop = int(g.loc[selected_idx, "hop"])

        first_hits.append(float(selected_hop == first_error_hop))
        any_hits.append(float(g.loc[selected_idx, "label"] == "incorrect"))
        random_first.append(1.0 / len(g))
        random_any.append(len(wrong) / len(g))

        correct = g[g["label"] == "correct"]
        if len(correct) and len(wrong):
            n_with_both_classes += 1
            for wi, wr in wrong.iterrows():
                for ci, cr in correct.iterrows():
                    uw = float(wr[uncertainty_col])
                    uc = float(cr[uncertainty_col])
                    if abs(uw - uc) <= 1e-12:
                        pair_hits.append(0.5)
                    else:
                        pair_hits.append(float(uw > uc))

    return {
        "questions_with_clear_error": int(n_with_error),
        "questions_with_wrong_and_correct_hops": int(n_with_both_classes),
        "first_error_top1": float(np.mean(first_hits)) if first_hits else None,
        "random_expected_first_error_top1": float(np.mean(random_first)) if random_first else None,
        "any_error_top1": float(np.mean(any_hits)) if any_hits else None,
        "random_expected_any_error_top1": float(np.mean(random_any)) if random_any else None,
        "pairwise_wrong_vs_correct_accuracy": float(np.mean(pair_hits)) if pair_hits else None,
        "n_wrong_correct_pairs": int(len(pair_hits)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--selection-file",
        default="outputs/qwen17b_seed42/selection.json",
        help="Use the SAME 60 questions as the isolated-hop pilot for a paired comparison.",
    )
    ap.add_argument("--run-dir", default="outputs/qwen17b_sequential_seed42")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--confidence-max-new-tokens", type=int, default=4)
    ap.add_argument(
        "--skip-verbal",
        action="store_true",
        help="Debug option. Main experiment should leave verbalized confidence enabled.",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = load_musique()
    selected = load_selection(args.selection_file)
    model, tok, device, dtype = load_subject_model(args.model, args.device, args.dtype)

    hop_rows = []
    trace_rows = []
    question_rows = []
    review_rows = []

    for qnum, dataset_idx in enumerate(selected, start=1):
        ex = ds[int(dataset_idx)]
        qid = str(ex["id"])
        ctx = context_block(ex)
        decomp = list(ex["question_decomposition"])

        model_answers: list[str] = []
        gold_answers: list[str] = []
        this_hops = []

        print(f"\n=== Sequential question {qnum}/{len(selected)} | {qid} ===")

        for i, step in enumerate(decomp):
            template_q = str(step["question"])
            gold = str(step["answer"])
            model_resolved_q = resolve_refs(template_q, model_answers)
            gold_resolved_q = resolve_refs(template_q, gold_answers)

            p = hop_prompt(tok, ctx, model_resolved_q)
            r = batched_generate_with_confidence(
                model,
                tok,
                device,
                [p],
                batch_size=1,
                max_new_tokens=args.max_new_tokens,
                progress_label=f"seq-hop-{i+1}",
            )[0]

            pred = str(r["answer"])
            label, f1, matched_gold = grade_answer(pred, gold)

            verbal_conf = None
            verbal_raw = ""
            if not args.skip_verbal:
                vp = verbal_confidence_prompt(tok, ctx, model_resolved_q, pred)
                vr = batched_generate_with_confidence(
                    model,
                    tok,
                    device,
                    [vp],
                    batch_size=1,
                    max_new_tokens=args.confidence_max_new_tokens,
                    progress_label=f"verbal-conf-{i+1}",
                )[0]
                verbal_raw = str(vr.get("raw_answer", vr.get("answer", "")))
                verbal_conf = parse_confidence(verbal_raw)

            # Approximate whether this hop is being asked on the benchmark/gold path.
            # Exact normalized question equality is conservative; we also record whether
            # all referenced prior benchmark states are still correct below.
            question_matches_gold_path = normalize(model_resolved_q) == normalize(gold_resolved_q)

            referenced = [int(x) - 1 for x in re.findall(r"#(\d+)", template_q)]
            referenced = [j for j in referenced if 0 <= j < i]
            parents_benchmark_correct = all(
                this_hops[j]["label"] == "correct" for j in referenced
            ) if referenced else True

            row = {
                "dataset_idx": int(dataset_idx),
                "question_id": qid,
                "n_hops": len(decomp),
                "hop_idx": i,
                "hop": i + 1,
                "template_question": template_q,
                "model_resolved_question": model_resolved_q,
                "gold_resolved_question": gold_resolved_q,
                "question_matches_gold_path": bool(question_matches_gold_path),
                "referenced_parent_indices": ",".join(str(j + 1) for j in referenced),
                "referenced_parents_benchmark_correct": bool(parents_benchmark_correct),
                "gold": gold,
                "pred": pred,
                "raw_pred": str(r.get("raw_answer", pred)),
                "label": label,
                "token_f1": float(f1),
                "matched_gold": matched_gold,
                "mean_logprob": float(r["mean_logprob"]),
                "min_logprob": float(r["min_logprob"]),
                "entropy": float(r["entropy"]),
                "margin": float(r["margin"]),
                "verbal_confidence": float(verbal_conf) if verbal_conf is not None else np.nan,
                "verbal_raw": verbal_raw,
                "generated_tokens": int(r["generated_tokens"]),
                "prompt_tokens": int(r["prompt_tokens"]),
            }
            hop_rows.append(row)
            this_hops.append(row)

            if label == "needs_review" or (not args.skip_verbal and verbal_conf is None):
                review_rows.append({
                    "kind": "hop" if label == "needs_review" else "verbal_parse",
                    "question_id": qid,
                    "hop": i + 1,
                    "question": model_resolved_q,
                    "gold": gold,
                    "pred": pred,
                    "verbal_raw": verbal_raw,
                    "manual_label_or_confidence": "",
                })

            model_answers.append(pred)
            gold_answers.append(gold)

        # Baseline final answer from the actual sequential trajectory.
        pairs = [
            (h["model_resolved_question"], h["pred"])
            for h in this_hops
        ]
        fp = final_prompt(tok, ctx, str(ex["question"]), pairs)
        fr = batched_generate_with_confidence(
            model,
            tok,
            device,
            [fp],
            batch_size=1,
            max_new_tokens=args.max_new_tokens,
            progress_label="seq-final",
        )[0]
        final_pred = str(fr["answer"])
        final_label, final_f1, final_matched = grade_answer(
            final_pred,
            str(ex["answer"]),
            list(ex.get("answer_aliases", []) or []),
        )
        final_gold_lp = safe_one_target_logprob(
            model,
            tok,
            device,
            fp,
            str(ex["answer"]),
            "seq-final-gold-logp",
        )

        clear_wrong_hops = [int(h["hop"]) for h in this_hops if h["label"] == "incorrect"]
        first_clear_error = min(clear_wrong_hops) if clear_wrong_hops else None

        qrow = {
            "dataset_idx": int(dataset_idx),
            "question_id": qid,
            "n_hops": len(decomp),
            "main_question": str(ex["question"]),
            "gold_final": str(ex["answer"]),
            "baseline_final_pred": final_pred,
            "baseline_final_label": final_label,
            "baseline_final_f1": float(final_f1),
            "baseline_final_matched_gold": final_matched,
            "baseline_final_gold_logprob": final_gold_lp,
            "first_clear_error_hop": first_clear_error,
            "n_clear_wrong_hops": len(clear_wrong_hops),
            "final_prompt_tokens": int(fr["prompt_tokens"]),
        }
        question_rows.append(qrow)

        trace_rows.append({
            **qrow,
            "answer_aliases": list(ex.get("answer_aliases", []) or []),
            "hops": this_hops,
        })

        if final_label == "needs_review":
            review_rows.append({
                "kind": "final",
                "question_id": qid,
                "hop": "",
                "question": str(ex["question"]),
                "gold": str(ex["answer"]),
                "pred": final_pred,
                "verbal_raw": "",
                "manual_label_or_confidence": "",
            })

    hops = pd.DataFrame(hop_rows).sort_values(["dataset_idx", "hop_idx"]).reset_index(drop=True)
    hops = add_within_trace_normalization(hops)
    questions = pd.DataFrame(question_rows)

    hops.to_csv(run_dir / "06_sequential_hops.csv", index=False)
    questions.to_csv(run_dir / "06_sequential_questions.csv", index=False)
    pd.DataFrame(review_rows).to_csv(run_dir / "06_review_queue.csv", index=False)
    write_jsonl(run_dir / "06_sequential_trace.jsonl", trace_rows)

    # Add normalized values back into trace JSONL for convenient downstream repair use.
    enriched_trace_rows = []
    for tr in trace_rows:
        qh = hops[hops["question_id"] == tr["question_id"]].sort_values("hop_idx")
        tr2 = dict(tr)
        tr2["hops"] = qh.replace({np.nan: None}).to_dict(orient="records")
        enriched_trace_rows.append(tr2)
    write_jsonl(run_dir / "06_sequential_trace.jsonl", enriched_trace_rows)

    scored_hops = hops[hops["label"].isin(["correct", "incorrect"])].copy()
    final_scored = questions[questions["baseline_final_label"].isin(["correct", "incorrect"])]

    detection = {}
    localization = {}
    for name in SIGNALS:
        raw_col = f"u_{name}"
        z_col = f"u_{name}_z"
        centered_col = f"u_{name}_centered"
        minmax_col = f"u_{name}_minmax"

        detection[name] = {
            "raw": detection_metrics(hops, raw_col),
            "within_trace_centered": detection_metrics(hops, centered_col),
            "within_trace_z": detection_metrics(hops, z_col),
            "within_trace_minmax": detection_metrics(hops, minmax_col),
        }
        localization[name] = {
            "raw": within_trace_localization(hops, raw_col),
            "within_trace_z": within_trace_localization(hops, z_col),
        }

    summary = {
        "environment": environment_metadata(args.model, device, dtype),
        "design": {
            "hop_calls": "individual sequential calls; # references resolved using prior MODEL answers",
            "verbalized_confidence": "separate post-answer self-evaluation call returning 0-100",
            "normalization_note": (
                "Within-trace affine normalization cannot change within-trace rank/top1; "
                "its main test is whether pooled cross-question error detection improves after "
                "removing question-level confidence offsets."
            ),
            "correctness_note": (
                "Benchmark correctness is recorded for every sequential hop, plus whether the "
                "resolved question remains on the benchmark/gold path. Downstream mismatches "
                "after an upstream error should not automatically be interpreted as independent errors."
            ),
        },
        "seed": args.seed,
        "selection_file": args.selection_file,
        "n_questions": int(len(questions)),
        "n_hops_total": int(len(hops)),
        "n_hops_scored": int(len(scored_hops)),
        "n_hops_needing_review": int((hops["label"] == "needs_review").sum()),
        "n_verbal_parse_failures": int(hops["verbal_confidence"].isna().sum()),
        "hop_accuracy_benchmark": float((scored_hops["label"] == "correct").mean()) if len(scored_hops) else None,
        "baseline_final_accuracy": float((final_scored["baseline_final_label"] == "correct").mean()) if len(final_scored) else None,
        "pooled_error_detection": detection,
        "within_trace_error_localization": localization,
    }

    write_json(run_dir / "06_summary.json", summary)

    print("\n================ 06 SEQUENTIAL CONFIDENCE ================")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    for name in [
        "06_sequential_hops.csv",
        "06_sequential_questions.csv",
        "06_sequential_trace.jsonl",
        "06_review_queue.csv",
        "06_summary.json",
    ]:
        print(f"  {run_dir / name}")


if __name__ == "__main__":
    main()
