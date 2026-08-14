#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    batched_target_logprob,
    context_block,
    final_prompt,
    load_musique,
    load_subject_model,
    normalize,
    read_jsonl,
    write_json,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/qwen17b_seed42")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--score-batch-size", type=int, default=2)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traces = read_jsonl(run_dir / "01_trace.jsonl")
    exp1 = json.loads((run_dir / "01_summary.json").read_text())
    model_name = args.model or exp1["environment"]["model"]

    ds = load_musique()
    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    jobs = []
    for tr in traces:
        # Match main repair analysis: focus on baseline failures.
        if tr["baseline_final_label"] != "incorrect":
            continue

        ex = ds[int(tr["dataset_idx"])]
        ctx = context_block(ex)
        hops = sorted(tr["hops"], key=lambda x: int(x["hop_idx"]))
        original_pairs = [(h["resolved_question"], h["pred"]) for h in hops]

        for h in hops:
            i = int(h["hop_idx"])
            if i >= len(hops) - 1:
                continue
            if h["label"] != "correct":
                continue

            changed = normalize(h["pred"]) != normalize(h["gold"])
            control_pairs = list(original_pairs)
            control_pairs[i] = (h["resolved_question"], h["gold"])
            p = final_prompt(tok, ctx, tr["main_question"], control_pairs)

            jobs.append({
                "question_id": tr["question_id"],
                "dataset_idx": int(tr["dataset_idx"]),
                "hop": i + 1,
                "original_correct_pred": h["pred"],
                "gold_inserted": h["gold"],
                "lexically_changed": bool(changed),
                "baseline_final_gold_logprob": float(tr["baseline_final_gold_logprob"]),
                "gold_final": tr["gold_final"],
                "prompt": p,
            })

    if not jobs:
        raise RuntimeError("No eligible correct-hop controls found.")

    scores = batched_target_logprob(
        model, tok, device,
        [x["prompt"] for x in jobs],
        [x["gold_final"] for x in jobs],
        batch_size=args.score_batch_size,
        progress_label="noop-gold-logp",
    )

    rows = []
    for job, s in zip(jobs, scores):
        row = {k: v for k, v in job.items() if k != "prompt"}
        row["noop_gold_logprob"] = s["mean_logprob"]
        row["noop_gain"] = s["mean_logprob"] - job["baseline_final_gold_logprob"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "04_noop_controls.csv", index=False)

    repairs = pd.read_csv(run_dir / "02_repairs.csv")
    lexical = df[df["lexically_changed"] == True]

    summary = {
        "model": model_name,
        "n_controls_total": int(len(df)),
        "n_lexically_changed_controls": int(len(lexical)),
        "mean_abs_noop_gain_all": float(df["noop_gain"].abs().mean()),
        "median_abs_noop_gain_all": float(df["noop_gain"].abs().median()),
        "mean_abs_noop_gain_lexical": (
            float(lexical["noop_gain"].abs().mean()) if len(lexical) else None
        ),
        "mean_abs_true_repair_gain": float(repairs["repair_gain"].abs().mean()),
        "median_abs_true_repair_gain": float(repairs["repair_gain"].abs().median()),
    }
    if summary["mean_abs_noop_gain_lexical"] is not None and summary["mean_abs_true_repair_gain"]:
        summary["lexical_noop_to_true_repair_abs_ratio"] = (
            summary["mean_abs_noop_gain_lexical"] /
            summary["mean_abs_true_repair_gain"]
        )

    write_json(run_dir / "04_noop_summary.json", summary)
    print("\n================ 04 NO-OP CONTROL ================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
