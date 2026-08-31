#!/usr/bin/env python3
from __future__ import annotations

"""
13_step_influence_claude.py

STEP INFLUENCE ("load-bearing vs inert") on the Test 06 sequential traces.

MOTIVATION
----------
The repair experiments can only rank hops on questions with >=2 wrong
non-terminal candidates. In this pilot that is 8 questions -- and 4 of those 8
share hops 2-4 (the `88460_30152_20999` chain), so the effective sample is
smaller still. Nothing can be ranked reliably at that size.

This script measures a different quantity that is defined for EVERY hop,
correct or wrong, on every question:

    How much does the final answer actually depend on this step's answer?

That is the load-bearing question, and it is separable from whether the step
was right. It yields ~150 scored hops instead of 8 ranked candidates.

DESIGN (frozen substitution)
----------------------------
For each non-terminal hop Hi:
  1. Take the original Test 06 trace.
  2. Substitute ONLY Hi's answer.
  3. DO NOT regenerate any descendant. Every later answer stays exactly as it
     was. This holds regeneration depth at zero for every hop, so hop position
     becomes a covariate that can be tested rather than a mechanical advantage.
  4. Rebuild the final prompt and rescore.

Substitutions:
    redact      answer -> "[unavailable]"   (leave-one-out; always defined,
                                             needs no gold, applies equally to
                                             correct and wrong hops)
    gold        answer -> MuSiQue hop gold  (repair direction; skipped when the
                                             model answer already matches gold)
    crossq      answer -> the gold answer of the same hop index taken from a
                          DIFFERENT question (a plausible, type-similar but
                          wrong entity; the corruption direction)
    sham        answer -> itself            (CONTROL: prompt is byte-identical,
                                             so any movement is decoding noise)

MEASURES
--------
    d_gold_logprob      change in mean logP(gold final answer)
    influence           |d_gold_logprob| under a substitution
    final_changed       did the generated final answer change
    final_correct       is the new final answer correct

The primary influence measure is redaction. `common.final_prompt` is used, i.e.
the ORIGINAL confidence-free final prompt, because this measures dependence on
step ANSWERS. Confidence enters only as a predictor to be correlated against
influence afterwards.

WHAT THIS CAN AND CANNOT SHOW
-----------------------------
It measures how much the FINAL-ANSWER step depends on each element of the
reasoning state it is shown. If the model largely re-derives the answer from
the documents and ignores the state, influence will be near zero everywhere.
That is a real possible outcome and the sham control plus the null distribution
of |d_gold_logprob| are what distinguish it from a measurement failure.

Substituting a step's answer does not re-run the reasoning that followed it, so
this is dependence of the final read-out on the state, not the full causal
effect a step would have in a live rollout. Test 07 measures the propagated
version; this is deliberately the frozen counterpart.

Only reads existing artifacts; writes exclusively into --run-dir.
"""

import argparse
import json
import time
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
    load_musique,
    load_subject_model,
    normalize,
    read_jsonl,
    seed_everything,
    write_json,
)

REDACT_TOKEN = "[unavailable]"


def gen_one(model, tok, device, prompt, max_new_tokens, label):
    return batched_generate_with_confidence(
        model, tok, device, [prompt], batch_size=1,
        max_new_tokens=max_new_tokens, progress_label=label,
    )[0]


def score_one(model, tok, device, prompt, target, label) -> float:
    out = batched_target_logprob(
        model, tok, device, [prompt], [target],
        batch_size=1, progress_label=label,
    )[0]
    x = float(out["mean_logprob"])
    if not np.isfinite(x):
        raise RuntimeError(f"Non-finite target logprob in {label}")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-file",
                    default="outputs/qwen17b_sequential_seed42/06_sequential_trace.jsonl")
    ap.add_argument("--source-summary",
                    default="outputs/qwen17b_sequential_seed42/06_summary.json")
    ap.add_argument("--run-dir", default="outputs/qwen17b_influence_claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--substitutions", default="redact,gold,crossq",
                    help="Subset of redact,gold,crossq")
    ap.add_argument("--sham-every", type=int, default=5,
                    help="Run the byte-identical sham control on every Nth hop. 0 disables.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    subs = [s.strip() for s in args.substitutions.split(",") if s.strip()]
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    traces = read_jsonl(args.trace_file)
    if args.limit_questions:
        traces = traces[:args.limit_questions]

    model_name = args.model or json.loads(
        Path(args.source_summary).read_text())["environment"]["model"]

    # Pool of cross-question substitution values, keyed by hop index so the
    # replacement is drawn from the same structural position.
    # Tagged with the source question so a hop can never draw its OWN gold,
    # which would silently collapse the corruption arm into the repair arm.
    pool: dict[int, list[tuple[str, str]]] = {}
    for tr in traces:
        for h in sorted(tr["hops"], key=lambda x: int(x["hop_idx"])):
            pool.setdefault(int(h["hop_idx"]), []).append(
                (str(tr["question_id"]), str(h["gold"]))
            )

    ds = load_musique()
    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    rows_path = run_dir / "13_influence.jsonl"
    csv_path = run_dir / "13_influence.csv"
    done = set()
    if args.resume and rows_path.exists():
        for r in read_jsonl(rows_path):
            done.add((r["question_id"], int(r["target_hop"]), r["substitution"]))
        print(f"resume: {len(done)} rows already complete")

    sink = open(rows_path, "a" if args.resume else "w", encoding="utf-8")
    rows = pd.read_csv(csv_path).to_dict("records") if (args.resume and csv_path.exists()) else []
    t0 = time.perf_counter()
    hop_counter = 0

    try:
        for qnum, tr in enumerate(traces, start=1):
            hops = sorted(tr["hops"], key=lambda x: int(x["hop_idx"]))
            n = len(hops)
            if n < 2:
                continue

            ex = ds[int(tr["dataset_idx"])]
            ctx = context_block(ex)
            main_q = str(tr["main_question"])
            gold_final = str(tr["gold_final"])
            aliases = list(tr.get("answer_aliases", []) or [])
            questions = [str(h["model_resolved_question"]) for h in hops]
            answers = [str(h["pred"]) for h in hops]

            base_prompt = final_prompt(tok, ctx, main_q,
                                       list(zip(questions, answers)))
            base_lp = score_one(model, tok, device, base_prompt, gold_final,
                                f"q{qnum}-base-goldlp")
            base_fin = gen_one(model, tok, device, base_prompt,
                               args.max_new_tokens, f"q{qnum}-base-final")
            base_pred = str(base_fin["answer"])
            base_label, base_f1, _ = grade_answer(base_pred, gold_final, aliases)

            print(f"\n=== q{qnum}/{len(traces)} {tr['question_id']} "
                  f"| {n} hops | base logP(gold)={base_lp:.3f} ===", flush=True)

            for i in range(n - 1):        # non-terminal hops only
                hop_counter += 1
                orig = answers[i]

                plan = []
                for s in subs:
                    if s == "redact":
                        plan.append(("redact", REDACT_TOKEN))
                    elif s == "gold":
                        g = str(hops[i]["gold"])
                        if normalize(g) != normalize(orig):
                            plan.append(("gold", g))
                    elif s == "crossq":
                        own_gold = normalize(str(hops[i]["gold"]))
                        cand = [
                            x for qid_src, x in pool.get(i, [])
                            if qid_src != str(tr["question_id"])
                            and normalize(x) != normalize(orig)
                            and normalize(x) != own_gold
                        ]
                        if cand:
                            plan.append(("crossq", str(cand[int(rng.integers(len(cand)))])))
                if args.sham_every and hop_counter % args.sham_every == 0:
                    plan.append(("sham", orig))

                for sub_name, new_val in plan:
                    key = (str(tr["question_id"]), i + 1, sub_name)
                    if key in done:
                        continue

                    alt = list(answers)
                    alt[i] = str(new_val)
                    p = final_prompt(tok, ctx, main_q, list(zip(questions, alt)))
                    lp = score_one(model, tok, device, p, gold_final,
                                   f"q{qnum}-h{i+1}-{sub_name}-goldlp")
                    fr = gen_one(model, tok, device, p, args.max_new_tokens,
                                 f"q{qnum}-h{i+1}-{sub_name}-final")
                    pred = str(fr["answer"])
                    label, f1, _ = grade_answer(pred, gold_final, aliases)

                    row = {
                        "question_id": str(tr["question_id"]),
                        "dataset_idx": int(tr["dataset_idx"]),
                        "n_hops": n,
                        "target_hop": i + 1,
                        "target_hop_idx": i,
                        "hop_position_frac": (i + 1) / n,
                        "hops_from_end": n - (i + 1),
                        "substitution": sub_name,
                        "original_answer": orig,
                        "substituted_answer": str(new_val),
                        "hop_gold": str(hops[i]["gold"]),
                        "hop_label": str(hops[i]["label"]),
                        "hop_benchmark_correct": bool(hops[i]["label"] == "correct"),
                        "hop_matches_gold_path": bool(
                            hops[i].get("question_matches_gold_path", False)),
                        # confidence predictors, carried for the analysis stage
                        "mean_logprob": hops[i].get("mean_logprob"),
                        "min_logprob": hops[i].get("min_logprob"),
                        "entropy": hops[i].get("entropy"),
                        "margin": hops[i].get("margin"),
                        "verbal_confidence": hops[i].get("verbal_confidence"),
                        # outcomes
                        "base_gold_logprob": base_lp,
                        "alt_gold_logprob": lp,
                        "d_gold_logprob": lp - base_lp,
                        "influence": abs(lp - base_lp),
                        "base_final_pred": base_pred,
                        "alt_final_pred": pred,
                        "final_changed": bool(normalize(pred) != normalize(base_pred)),
                        "base_final_label": base_label,
                        "alt_final_label": label,
                        "alt_final_correct": bool(label == "correct"),
                        "alt_final_f1": float(f1),
                        "gold_final": gold_final,
                    }
                    rows.append(row)
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()

            pd.DataFrame(rows).to_csv(csv_path, index=False)
    finally:
        sink.close()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    write_json(run_dir / "13_meta.json", {
        "script": "13_step_influence_claude.py",
        "environment": environment_metadata(model_name, device, dtype),
        "seed": args.seed,
        "trace_file": args.trace_file,
        "substitutions": subs,
        "sham_every": args.sham_every,
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()) if len(df) else 0,
        "n_hops_scored": int(df.groupby(["question_id", "target_hop"]).ngroups) if len(df) else 0,
        "wall_clock_seconds": float(time.perf_counter() - t0),
        "design": "frozen substitution; descendants are NOT regenerated",
    })
    print(f"\nWrote {len(df)} rows to {csv_path}")
    print(f"Analyse with: 13b_analyze_influence_claude.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
