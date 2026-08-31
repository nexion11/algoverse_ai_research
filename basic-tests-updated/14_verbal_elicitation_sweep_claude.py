#!/usr/bin/env python3
from __future__ import annotations

"""
14_verbal_elicitation_sweep_claude.py

TODO 1: raw verbalized confidence, and "tune the prompt a little bit".

PROBLEM
-------
Test 06's verbalized confidence is nearly degenerate: 9 distinct values across
210 hops, 112 of them exactly 40, and 18 of 60 traces completely flat. That
caps every downstream use (pooled AUROC 0.626, and ~random for first-error
localization). No aggregation or normalization can recover a signal that has
almost no variance.

DESIGN
------
Every variant is a POST-HOC elicitation on the FIXED Test 06 answers, exactly
like the original: the answer is never regenerated, so the variants are
directly comparable to each other and to Test 06, and eliciting confidence
cannot change the answer being judged. A joint answer+confidence condition
would change the answers and so is deliberately excluded here.

VARIANTS
--------
  baseline       byte-identical to 06_sequential_confidence.py. Control:
                 should reproduce the 40-attractor.
  anchored       full-range instruction with explicit calibration anchors and
                 an instruction against round numbers.
  bins           seven labeled bins mapped to numbers, removing free choice of
                 an integer entirely.
  evidence_first a short evidence statement before the number, so the value is
                 conditioned on retrieved evidence rather than emitted cold.
  twosided       asks for P(answer is INCORRECT), then inverts. Tests whether
                 the attractor is a property of the question framing.

METRICS
-------
Distributional health (the actual bottleneck):
  n_distinct, mode_share, normalized entropy, flat-trace count, parse failures
Discriminative value:
  AUROC / AUPRC for benchmark error detection, Spearman vs token_f1
A variant is only an improvement if it moves BOTH: spreading the distribution
while staying uninformative is not progress.

Reads Test 06 artifacts; writes only into --run-dir.
"""

import argparse, json, re, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    batched_generate_with_confidence, context_block, environment_metadata,
    load_musique, load_subject_model, read_jsonl, safe_auroc_auprc,
    seed_everything, write_json,
)

BIN_MAP = {
    "CERTAINLY_WRONG": 0.0, "PROBABLY_WRONG": 17.0, "LEANING_WRONG": 33.0,
    "UNSURE": 50.0, "LEANING_RIGHT": 67.0, "PROBABLY_RIGHT": 83.0,
    "CERTAINLY_RIGHT": 100.0,
}


def chat(tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_int_0_100(text):
    m = re.search(r"(?<!\d)(100|\d{1,2})(?:\.\d+)?(?!\d)", str(text))
    if not m:
        return None
    x = float(m.group(0))
    return x if 0.0 <= x <= 100.0 else None


# ---- variants: (system, user_builder, max_new_tokens, parser) --------------

def _v_baseline(ctx, subq, ans):
    return (
        "You are evaluating your own short answer to a document-grounded question. "
        "Estimate the probability that the supplied answer is correct given the documents. "
        "Return ONLY one integer from 0 to 100. Do not explain.",
        f"DOCUMENTS:\n{ctx}\n\nSUB-QUESTION:\n{subq}\n\nYOUR ANSWER:\n{ans}\n\n"
        "CONFIDENCE (0-100):",
    )


def _v_anchored(ctx, subq, ans):
    return (
        "You are evaluating your own short answer to a document-grounded question. "
        "Estimate the probability that the supplied answer is correct given the documents. "
        "Use the FULL 0-100 range. Calibration anchors: 0 = certainly incorrect, "
        "25 = probably incorrect, 50 = genuinely uncertain, 75 = probably correct, "
        "100 = certainly correct. Do NOT default to round numbers such as 40 or 50; "
        "give your true estimate, for example 37 or 82. "
        "Return ONLY one integer from 0 to 100. Do not explain.",
        f"DOCUMENTS:\n{ctx}\n\nSUB-QUESTION:\n{subq}\n\nYOUR ANSWER:\n{ans}\n\n"
        "CONFIDENCE (0-100):",
    )


def _v_bins(ctx, subq, ans):
    return (
        "You are evaluating your own short answer to a document-grounded question. "
        "Judge whether the supplied answer is correct given the documents. "
        "Return ONLY one of these exact labels: CERTAINLY_WRONG, PROBABLY_WRONG, "
        "LEANING_WRONG, UNSURE, LEANING_RIGHT, PROBABLY_RIGHT, CERTAINLY_RIGHT. "
        "Do not explain.",
        f"DOCUMENTS:\n{ctx}\n\nSUB-QUESTION:\n{subq}\n\nYOUR ANSWER:\n{ans}\n\nVERDICT:",
    )


def _v_evidence_first(ctx, subq, ans):
    return (
        "You are evaluating your own short answer to a document-grounded question. "
        "First state, in at most 8 words, which document supports or contradicts the "
        "answer. Then on a new line output exactly: CONFIDENCE: <integer 0-100> "
        "giving the probability the answer is correct. Use the full 0-100 range.",
        f"DOCUMENTS:\n{ctx}\n\nSUB-QUESTION:\n{subq}\n\nYOUR ANSWER:\n{ans}\n\nEVIDENCE:",
    )


def _v_twosided(ctx, subq, ans):
    return (
        "You are evaluating your own short answer to a document-grounded question. "
        "Estimate the probability that the supplied answer is INCORRECT given the "
        "documents. Use the full 0-100 range. "
        "Return ONLY one integer from 0 to 100. Do not explain.",
        f"DOCUMENTS:\n{ctx}\n\nSUB-QUESTION:\n{subq}\n\nYOUR ANSWER:\n{ans}\n\n"
        "PROBABILITY THE ANSWER IS INCORRECT (0-100):",
    )


def parse_bins(text):
    up = str(text).upper()
    for k, v in BIN_MAP.items():          # longest-first to avoid prefix clashes
        pass
    for k in sorted(BIN_MAP, key=len, reverse=True):
        if k in up:
            return BIN_MAP[k]
    return None


def parse_evidence(text):
    m = re.search(r"CONFIDENCE\s*:\s*(100|\d{1,2})", str(text), re.I)
    if m:
        x = float(m.group(1))
        return x if 0 <= x <= 100 else None
    return parse_int_0_100(str(text).split("\n")[-1])


VARIANTS = {
    "baseline":       (_v_baseline, 4, parse_int_0_100, False),
    "anchored":       (_v_anchored, 6, parse_int_0_100, False),
    "bins":           (_v_bins, 8, parse_bins, False),
    "evidence_first": (_v_evidence_first, 48, parse_evidence, False),
    "twosided":       (_v_twosided, 6, parse_int_0_100, True),   # invert
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-file",
                    default="outputs/qwen17b_sequential_seed42/06_sequential_trace.jsonl")
    ap.add_argument("--source-summary",
                    default="outputs/qwen17b_sequential_seed42/06_summary.json")
    ap.add_argument("--run-dir", default="outputs/qwen17b_elicitation_claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = set(variants) - set(VARIANTS)
    if bad:
        raise SystemExit(f"unknown variants: {sorted(bad)}")

    seed_everything(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    traces = read_jsonl(args.trace_file)
    if args.limit_questions:
        traces = traces[:args.limit_questions]
    model_name = args.model or json.loads(
        Path(args.source_summary).read_text())["environment"]["model"]

    ds = load_musique()
    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    csv_path = run_dir / "14_elicitation.csv"
    jsonl_path = run_dir / "14_elicitation.jsonl"
    done = set()
    if args.resume and jsonl_path.exists():
        for r in read_jsonl(jsonl_path):
            done.add((r["question_id"], int(r["hop"]), r["variant"]))
        print(f"resume: {len(done)} rows done")
    sink = open(jsonl_path, "a" if args.resume else "w", encoding="utf-8")
    rows = pd.read_csv(csv_path).to_dict("records") if (args.resume and csv_path.exists()) else []

    t0 = time.perf_counter()
    try:
        for qnum, tr in enumerate(traces, start=1):
            ex = ds[int(tr["dataset_idx"])]
            ctx = context_block(ex)
            hops = sorted(tr["hops"], key=lambda x: int(x["hop_idx"]))
            print(f"\n=== q{qnum}/{len(traces)} {tr['question_id']} ===", flush=True)
            for h in hops:
                subq, ans = str(h["model_resolved_question"]), str(h["pred"])
                for vname in variants:
                    key = (str(tr["question_id"]), int(h["hop"]), vname)
                    if key in done:
                        continue
                    builder, mnt, parser, invert = VARIANTS[vname]
                    system, user = builder(ctx, subq, ans)
                    p = chat(tok, system, user)
                    r = batched_generate_with_confidence(
                        model, tok, device, [p], batch_size=1, max_new_tokens=mnt,
                        progress_label=f"q{qnum}-h{h['hop']}-{vname}")[0]
                    raw = str(r.get("raw_answer", r.get("answer", "")))
                    val = parser(raw)
                    if val is not None and invert:
                        val = 100.0 - val
                    row = {
                        "question_id": str(tr["question_id"]),
                        "dataset_idx": int(tr["dataset_idx"]),
                        "n_hops": int(tr["n_hops"]),
                        "hop": int(h["hop"]),
                        "variant": vname,
                        "confidence": val,
                        "parsed": val is not None,
                        "raw_output": raw,
                        "pred": ans,
                        "gold": str(h["gold"]),
                        "label": str(h["label"]),
                        "token_f1": float(h["token_f1"]),
                        "test06_verbal_confidence": h.get("verbal_confidence"),
                        "mean_logprob": h.get("mean_logprob"),
                    }
                    rows.append(row)
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n"); sink.flush()
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    finally:
        sink.close()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # ---------------- analysis ----------------
    summary = {"environment": environment_metadata(model_name, device, dtype),
               "n_hops": int(df.groupby(["question_id", "hop"]).ngroups),
               "wall_clock_min": (time.perf_counter() - t0) / 60.0,
               "variants": {}}
    for v, g in df.groupby("variant"):
        ok = g[g.parsed]
        vals = ok.confidence.astype(float)
        cnt = Counter(vals)
        n = len(vals)
        ent = (-sum((c / n) * np.log(c / n) for c in cnt.values()) / np.log(n)) if n > 1 else 0.0
        flat = 0
        for _, gg in ok.groupby("question_id"):
            if len(gg) > 1 and gg.confidence.nunique() == 1:
                flat += 1
        sc = ok[ok.label.isin(["correct", "incorrect"])]
        det = safe_auroc_auprc((-sc.confidence.astype(float)).to_numpy(),
                               (sc.label == "incorrect").astype(int).to_numpy())
        from scipy.stats import spearmanr
        sp = spearmanr(ok.confidence.astype(float), ok.token_f1.astype(float))
        summary["variants"][v] = {
            "n_parsed": int(len(ok)), "parse_failure_rate": float(1 - g.parsed.mean()),
            "n_distinct_values": int(len(cnt)),
            "mode_value": float(max(cnt, key=cnt.get)) if cnt else None,
            "mode_share": float(max(cnt.values()) / n) if n else None,
            "normalized_entropy": float(ent),
            "n_flat_traces": int(flat),
            "mean": float(vals.mean()) if n else None,
            "std": float(vals.std(ddof=0)) if n else None,
            "error_detection_auroc": det["auroc"], "error_detection_auprc": det["auprc"],
            "spearman_vs_token_f1": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "value_counts": {str(k): int(c) for k, c in sorted(cnt.items())},
        }
    summary["reference_test06"] = {
        "n_distinct_values": 9, "mode_value": 40.0, "mode_share": 112 / 210,
        "n_flat_traces": 18, "error_detection_auroc": 0.626,
    }
    summary["reading"] = (
        "A variant improves on Test 06 only if it BOTH spreads the distribution "
        "(more distinct values, lower mode share, higher entropy, fewer flat traces) "
        "AND discriminates better (higher AUROC/AUPRC). Spread alone is not progress."
    )
    write_json(run_dir / "14_summary.json", summary)

    tbl = pd.DataFrame([{
        "variant": v, "n_distinct": s["n_distinct_values"], "mode": s["mode_value"],
        "mode_share": s["mode_share"], "norm_entropy": s["normalized_entropy"],
        "flat_traces": s["n_flat_traces"], "auroc": s["error_detection_auroc"],
        "auprc": s["error_detection_auprc"], "spearman_f1": s["spearman_vs_token_f1"],
        "parse_fail": s["parse_failure_rate"],
    } for v, s in summary["variants"].items()]).sort_values("auroc", ascending=False)
    tbl.to_csv(run_dir / "14_variant_comparison.csv", index=False)

    print("\n============ VERBAL ELICITATION SWEEP ============")
    print(tbl.to_string(index=False))
    print(f"\nTest 06 reference: distinct=9 mode=40 mode_share=0.533 "
          f"flat_traces=18 auroc=0.626")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
