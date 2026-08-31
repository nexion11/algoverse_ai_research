#!/usr/bin/env python3
from __future__ import annotations

"""
16_bake_confidence_claude.py

TODO 3: bake the per-step confidence and the aggregation method into the model,
and TODO 2's causal half ("does it HELP the final result", not just predict it).

TIERS OF "BAKING IN"
--------------------
  Tier 1  supply per-step confidence and the trace aggregate as INPUT STATE to
          the final-answer step, and measure whether the final answer improves.
  Tier 2  train on it (LoRA / a confidence head).

This script is Tier 1. Tier 1 must come first: fine-tuning a model to consume a
signal that demonstrably does not help when supplied directly would be
expensive and uninformative. Tier 1 is also the only version that isolates
whether the information is USABLE, separately from whether the model has been
taught to use it.

CONDITIONS (per question; hop answers are IDENTICAL in all of them)
------------------------------------------------------------------
  control        the original confidence-free final prompt from common.py.
                 Reproduces the Test 06 baseline exactly.
  per_step       + a CONFIDENCE value on every step
  agg_only       + a single TRACE RELIABILITY number, no per-step values
  per_step_agg   + both
  shuffled       + per-step confidences RANDOMLY PERMUTED across steps, and the
                 aggregate recomputed from the permuted values so it is
                 unchanged. THE CRITICAL CONTROL.

WHY THE SHUFFLED CONTROL DECIDES THE RESULT
-------------------------------------------
A permutation keeps the multiset of numbers, the prompt length, the format, and
the aggregate identical; only the ASSIGNMENT of confidence to step changes. So:
  per_step > shuffled   the model is using WHICH step is uncertain.
  per_step == shuffled  any gain comes from the presence of numbers or from
                        being told the state is uncertain, not from step-level
                        confidence. That is not baking in a signal, it is a
                        prompt-format effect.
Without it, an improvement over `control` cannot be attributed to confidence.

MEASURES
--------
  mean logP(gold final)   the repo's metric axis (repair_gain is a difference
                          of exactly this quantity)
  final answer accuracy
  paired per-question deltas against control, with bootstrap CIs

Confidence is supplied RAW (0-100 verbalized, and raw white-box values), since
raw beat every within-trace normalization in this pilot. The aggregate is a
plain mean, since the aggregation sweep found no aggregator beats the mean.
It is a within-trace summary for prompting, NOT a calibrated probability.

Reads Test 06 artifacts; writes only into --run-dir.
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    batched_generate_with_confidence, batched_target_logprob, context_block,
    environment_metadata, final_prompt, grade_answer, load_musique,
    load_subject_model, read_jsonl, seed_everything, write_json,
)

SYS_CONF = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Each step lists its SUB-QUESTION, its ANSWER, and a CONFIDENCE value from 0 to 100 "
    "giving how reliable that step's answer is. Rely on high-confidence steps; treat "
    "low-confidence steps with caution and re-check them against the documents. "
    "Return only the shortest factual answer span to the main question. Do not explain."
)
SYS_AGG = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "You are given TRACE RELIABILITY, a 0-100 summary of how reliable the reasoning "
    "state is overall. If it is low, re-check the state against the documents. "
    "Return only the shortest factual answer span to the main question. Do not explain."
)
SYS_BOTH = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Each step lists its SUB-QUESTION, its ANSWER, and a CONFIDENCE value from 0 to 100 "
    "giving how reliable that step's answer is. TRACE RELIABILITY is a 0-100 summary "
    "over the whole state. Rely on high-confidence steps; treat low-confidence steps "
    "with caution and re-check them against the documents. "
    "Return only the shortest factual answer span to the main question. Do not explain."
)


def chat(tok, system, user):
    m = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def build(tok, ctx, main_q, qs, ans, conf, agg, mode):
    if mode == "control":
        return final_prompt(tok, ctx, main_q, list(zip(qs, ans)))
    if mode == "agg_only":
        chain = "\n".join(f"- {q} -> {a}" for q, a in zip(qs, ans))
        user = (f"DOCUMENTS:\n{ctx}\n\nSTRUCTURED REASONING STATE:\n{chain}\n\n"
                f"TRACE RELIABILITY (0-100): {agg:.1f}\n\n"
                f"MAIN QUESTION:\n{main_q}\n\nSHORT ANSWER:")
        return chat(tok, SYS_AGG, user)
    chain = "\n".join(f"- {q} -> {a}  [CONFIDENCE: {c:.1f}]"
                      for q, a, c in zip(qs, ans, conf))
    if mode == "per_step" or mode == "shuffled":
        user = (f"DOCUMENTS:\n{ctx}\n\nSTRUCTURED REASONING STATE:\n{chain}\n\n"
                f"MAIN QUESTION:\n{main_q}\n\nSHORT ANSWER:")
        return chat(tok, SYS_CONF, user)
    user = (f"DOCUMENTS:\n{ctx}\n\nSTRUCTURED REASONING STATE:\n{chain}\n\n"
            f"TRACE RELIABILITY (0-100): {agg:.1f}\n\n"
            f"MAIN QUESTION:\n{main_q}\n\nSHORT ANSWER:")
    return chat(tok, SYS_BOTH, user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-file",
                    default="outputs/qwen17b_sequential_seed42/06_sequential_trace.jsonl")
    ap.add_argument("--source-summary",
                    default="outputs/qwen17b_sequential_seed42/06_summary.json")
    ap.add_argument("--run-dir", default="outputs/qwen17b_bake_claude")
    ap.add_argument("--signal", default="verbal_confidence",
                    choices=["verbal_confidence", "mean_logprob", "entropy", "margin"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--conditions", default="control,per_step,agg_only,per_step_agg,shuffled")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    traces = read_jsonl(args.trace_file)
    if args.limit_questions:
        traces = traces[:args.limit_questions]
    model_name = args.model or json.loads(
        Path(args.source_summary).read_text())["environment"]["model"]

    ds = load_musique()
    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    csv_path = run_dir / "16_bake.csv"
    jl = run_dir / "16_bake.jsonl"
    done = set()
    if args.resume and jl.exists():
        for r in read_jsonl(jl):
            done.add((r["question_id"], r["condition"]))
    sink = open(jl, "a" if args.resume else "w", encoding="utf-8")
    rows = pd.read_csv(csv_path).to_dict("records") if (args.resume and csv_path.exists()) else []
    t0 = time.perf_counter()

    try:
        for qnum, tr in enumerate(traces, start=1):
            ex = ds[int(tr["dataset_idx"])]
            ctx = context_block(ex)
            hops = sorted(tr["hops"], key=lambda x: int(x["hop_idx"]))
            qs = [str(h["model_resolved_question"]) for h in hops]
            ans = [str(h["pred"]) for h in hops]
            gold_final = str(tr["gold_final"])
            aliases = list(tr.get("answer_aliases", []) or [])

            raw = pd.to_numeric(pd.Series([h.get(args.signal) for h in hops]),
                                errors="coerce").astype(float)
            if args.signal == "verbal_confidence":
                conf = raw.fillna(50.0).to_numpy()
            else:  # map white-box signal to a 0-100 confidence-oriented display value
                v = raw.to_numpy()
                sign = -1.0 if args.signal == "entropy" else 1.0
                v = sign * v
                lo, hi = np.nanmin(v), np.nanmax(v)
                conf = (np.full_like(v, 50.0) if not np.isfinite(hi - lo) or hi - lo < 1e-12
                        else 100.0 * (v - lo) / (hi - lo))
            agg = float(np.nanmean(conf))
            perm = rng.permutation(len(conf))
            shuffled_conf = conf[perm]

            print(f"\n=== q{qnum}/{len(traces)} {tr['question_id']} | agg={agg:.1f} ===",
                  flush=True)
            for cond in conds:
                if (str(tr["question_id"]), cond) in done:
                    continue
                c = shuffled_conf if cond == "shuffled" else conf
                p = build(tok, ctx, str(tr["main_question"]), qs, ans, c, agg, cond)
                lp = float(batched_target_logprob(
                    model, tok, device, [p], [gold_final], batch_size=1,
                    progress_label=f"q{qnum}-{cond}-goldlp")[0]["mean_logprob"])
                r = batched_generate_with_confidence(
                    model, tok, device, [p], batch_size=1,
                    max_new_tokens=args.max_new_tokens,
                    progress_label=f"q{qnum}-{cond}-final")[0]
                pred = str(r["answer"])
                label, f1, _ = grade_answer(pred, gold_final, aliases)
                row = {
                    "question_id": str(tr["question_id"]),
                    "dataset_idx": int(tr["dataset_idx"]),
                    "n_hops": int(tr["n_hops"]), "condition": cond,
                    "signal": args.signal, "trace_aggregate": agg,
                    "confidences": ",".join(f"{x:.1f}" for x in c),
                    "gold_final_logprob": lp, "final_pred": pred,
                    "final_label": label, "final_correct": bool(label == "correct"),
                    "final_f1": float(f1),
                    "test06_baseline_gold_logprob": float(tr["baseline_final_gold_logprob"]),
                    "test06_baseline_label": str(tr["baseline_final_label"]),
                }
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n"); sink.flush()
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    finally:
        sink.close()

    df = pd.DataFrame(rows); df.to_csv(csv_path, index=False)

    # ---------------- paired analysis vs control ----------------
    piv_lp = df.pivot_table(index="question_id", columns="condition",
                            values="gold_final_logprob")
    piv_ok = df.pivot_table(index="question_id", columns="condition",
                            values="final_correct")
    scored = df[df.final_label.isin(["correct", "incorrect"])]
    rngb = np.random.default_rng(0)
    res = {}
    for cond in conds:
        if cond == "control" or cond not in piv_lp.columns:
            continue
        d = (piv_lp[cond] - piv_lp["control"]).dropna().to_numpy()
        a = (piv_ok[cond] - piv_ok["control"]).dropna().astype(float).to_numpy()
        bl = [float(d[s].mean()) for s in (rngb.integers(0, len(d), len(d)) for _ in range(5000))]
        ba = [float(a[s].mean()) for s in (rngb.integers(0, len(a), len(a)) for _ in range(5000))]
        res[cond] = {
            "n_questions": int(len(d)),
            "mean_d_gold_logprob_vs_control": float(d.mean()),
            "d_logprob_ci95": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
            "accuracy_control": float(piv_ok["control"].mean()),
            "accuracy_condition": float(piv_ok[cond].mean()),
            "mean_d_accuracy_vs_control": float(a.mean()),
            "d_accuracy_ci95": [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))],
            "n_final_answers_changed": int((
                df[df.condition == cond].set_index("question_id").final_pred
                != df[df.condition == "control"].set_index("question_id").final_pred).sum()),
        }
    # the decisive contrast
    if "per_step" in piv_lp.columns and "shuffled" in piv_lp.columns:
        d = (piv_lp["per_step"] - piv_lp["shuffled"]).dropna().to_numpy()
        a = (piv_ok["per_step"] - piv_ok["shuffled"]).dropna().astype(float).to_numpy()
        bl = [float(d[s].mean()) for s in (rngb.integers(0, len(d), len(d)) for _ in range(5000))]
        res["per_step_vs_shuffled"] = {
            "n_questions": int(len(d)),
            "mean_d_gold_logprob": float(d.mean()),
            "ci95": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
            "mean_d_accuracy": float(a.mean()),
            "interpretation": (
                "Real confidence assignment vs the same numbers permuted across steps. "
                "A CI containing 0 means the model is not using WHICH step is uncertain, "
                "so any gain over control is a prompt-format effect, not baked-in confidence."
            ),
        }
    summary = {
        "environment": environment_metadata(model_name, device, dtype),
        "tier": "1 (supply as input state); tier 2 (fine-tuning) not attempted",
        "signal": args.signal, "conditions": conds,
        "n_questions": int(df.question_id.nunique()),
        "n_needs_review": int((df.final_label == "needs_review").sum()),
        "wall_clock_min": (time.perf_counter() - t0) / 60.0,
        "results_vs_control": res,
        "caveats": [
            "Confidence is supplied raw; the aggregate is a plain mean and is a prompting "
            "summary, not a calibrated probability.",
            "Hop answers are identical across all conditions; only the metadata differs.",
            "Read per_step_vs_shuffled before attributing any gain to confidence.",
        ],
    }
    write_json(run_dir / "16_summary.json", summary)
    print("\n============ BAKE-IN (TIER 1) ============")
    print(json.dumps(res, indent=2))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
