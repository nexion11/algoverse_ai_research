#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    balanced_selection,
    batched_generate_with_confidence,
    batched_target_logprob,
    context_block,
    environment_metadata,
    final_prompt,
    gold_resolved_hops,
    grade_answer,
    hop_prompt,
    load_musique,
    load_selection,
    load_subject_model,
    safe_auroc_auprc,
    save_selection,
    seed_everything,
    write_json,
    write_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--n-questions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--score-batch-size", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--selection-file", default=None)
    ap.add_argument("--run-dir", default="outputs/qwen17b_seed42")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    ds = load_musique()
    if args.selection_file:
        selected = load_selection(args.selection_file)
    else:
        selected = balanced_selection(ds, args.n_questions, args.seed)

    save_selection(run_dir / "selection.json", ds, selected, args.seed)

    model, tok, device, dtype = load_subject_model(args.model, args.device, args.dtype)

    # -------------------------------------------------------
    # Build ALL benchmark-defined hop prompts before inference.
    # Each reference (#1/#2/...) is resolved with gold benchmark
    # states so each hop has a fixed semantic target.
    # -------------------------------------------------------
    jobs = []
    for dataset_idx in selected:
        ex = ds[dataset_idx]
        ctx = context_block(ex)
        for h in gold_resolved_hops(ex):
            jobs.append({
                "dataset_idx": dataset_idx,
                "question_id": str(ex["id"]),
                "n_hops": len(ex["question_decomposition"]),
                "main_question": str(ex["question"]),
                "gold_final": str(ex["answer"]),
                "answer_aliases": list(ex.get("answer_aliases", []) or []),
                **h,
                "prompt": hop_prompt(tok, ctx, h["resolved_question"]),
            })

    prompts = [x["prompt"] for x in jobs]

    print(f"\nGenerating {len(prompts)} intermediate-hop answers in batches...")
    gen = batched_generate_with_confidence(
        model, tok, device, prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        progress_label="hops",
    )

    hop_rows = []
    review_rows = []
    grouped = defaultdict(list)

    for job, r in zip(jobs, gen):
        label, f1, matched_gold = grade_answer(r["answer"], job["gold"])
        row = {
            "dataset_idx": job["dataset_idx"],
            "question_id": job["question_id"],
            "n_hops": job["n_hops"],
            "hop_idx": job["hop_idx"],
            "hop": job["hop"],
            "template_question": job["template_question"],
            "resolved_question": job["resolved_question"],
            "gold": job["gold"],
            "pred": r["answer"],
            "raw_pred": r["raw_answer"],
            "label": label,
            "token_f1": f1,
            "matched_gold": matched_gold,
            "mean_logprob": r["mean_logprob"],
            "min_logprob": r["min_logprob"],
            "entropy": r["entropy"],
            "margin": r["margin"],
            "generated_tokens": r["generated_tokens"],
            "prompt_tokens": r["prompt_tokens"],
        }
        hop_rows.append(row)
        grouped[job["question_id"]].append(row)

        if label == "needs_review":
            review_rows.append({
                "kind": "hop",
                "question_id": job["question_id"],
                "hop": job["hop"],
                "question": job["resolved_question"],
                "gold": job["gold"],
                "pred": r["answer"],
                "manual_label": "",
            })

    hops_df = pd.DataFrame(hop_rows).sort_values(["dataset_idx", "hop_idx"])
    hops_df.to_csv(run_dir / "01_hops.csv", index=False)
    pd.DataFrame(review_rows).to_csv(run_dir / "01_review_queue.csv", index=False)

    # -------------------------------------------------------
    # Construct one fixed structured reasoning state per question
    # and obtain the baseline final answer.
    # -------------------------------------------------------
    final_jobs = []
    traces = []

    for dataset_idx in selected:
        ex = ds[dataset_idx]
        qid = str(ex["id"])
        hs = sorted(grouped[qid], key=lambda x: x["hop_idx"])
        ctx = context_block(ex)
        pairs = [(h["resolved_question"], h["pred"]) for h in hs]
        p = final_prompt(tok, ctx, ex["question"], pairs)

        final_jobs.append({
            "dataset_idx": dataset_idx,
            "question_id": qid,
            "n_hops": len(hs),
            "main_question": str(ex["question"]),
            "gold_final": str(ex["answer"]),
            "answer_aliases": list(ex.get("answer_aliases", []) or []),
            "prompt": p,
            "pairs": pairs,
            "hops": hs,
        })

    final_prompts = [x["prompt"] for x in final_jobs]
    print(f"\nGenerating {len(final_prompts)} baseline final answers...")
    final_gen = batched_generate_with_confidence(
        model, tok, device, final_prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        progress_label="baseline-final",
    )

    print("\nScoring gold final answers...")
    final_lp = batched_target_logprob(
        model, tok, device,
        final_prompts,
        [x["gold_final"] for x in final_jobs],
        batch_size=args.score_batch_size,
        progress_label="baseline-gold-logp",
    )

    question_rows = []
    for job, g, lp in zip(final_jobs, final_gen, final_lp):
        label, f1, matched = grade_answer(
            g["answer"], job["gold_final"], job["answer_aliases"]
        )
        qrow = {
            "dataset_idx": job["dataset_idx"],
            "question_id": job["question_id"],
            "n_hops": job["n_hops"],
            "main_question": job["main_question"],
            "gold_final": job["gold_final"],
            "baseline_final_pred": g["answer"],
            "baseline_final_label": label,
            "baseline_final_f1": f1,
            "baseline_final_gold_logprob": lp["mean_logprob"],
            "final_prompt_tokens": g["prompt_tokens"],
        }
        question_rows.append(qrow)

        traces.append({
            **qrow,
            "answer_aliases": job["answer_aliases"],
            "hops": job["hops"],
        })

        if label == "needs_review":
            review_rows.append({
                "kind": "final",
                "question_id": job["question_id"],
                "hop": "",
                "question": job["main_question"],
                "gold": job["gold_final"],
                "pred": g["answer"],
                "manual_label": "",
            })

    questions_df = pd.DataFrame(question_rows)
    questions_df.to_csv(run_dir / "01_questions.csv", index=False)
    pd.DataFrame(review_rows).to_csv(run_dir / "01_review_queue.csv", index=False)
    write_jsonl(run_dir / "01_trace.jsonl", traces)

    # -------------------------------------------------------
    # Confidence -> correctness metrics.
    # -------------------------------------------------------
    scored = hops_df[hops_df["label"].isin(["correct", "incorrect"])].copy()
    error = (scored["label"] == "incorrect").astype(int).to_numpy()

    detection = {
        "mean_logprob": safe_auroc_auprc(-scored["mean_logprob"].to_numpy(), error),
        "min_logprob": safe_auroc_auprc(-scored["min_logprob"].to_numpy(), error),
        "entropy": safe_auroc_auprc(scored["entropy"].to_numpy(), error),
        "margin": safe_auroc_auprc(-scored["margin"].to_numpy(), error),
    }

    final_scored = questions_df[
        questions_df["baseline_final_label"].isin(["correct", "incorrect"])
    ]
    summary = {
        "environment": environment_metadata(args.model, device, dtype),
        "seed": args.seed,
        "n_questions": int(len(questions_df)),
        "n_hops_total": int(len(hops_df)),
        "n_hops_scored": int(len(scored)),
        "n_hops_manual_review": int((hops_df["label"] == "needs_review").sum()),
        "hop_accuracy": float((scored["label"] == "correct").mean()) if len(scored) else None,
        "baseline_final_accuracy": (
            float((final_scored["baseline_final_label"] == "correct").mean())
            if len(final_scored) else None
        ),
        "confidence_to_error": detection,
    }

    write_json(run_dir / "01_summary.json", summary)
    print("\n================ 01 SUMMARY ================")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
