#!/usr/bin/env python3
from __future__ import annotations

"""
07_sequential_repair.py

Repair-value experiment on the sequential traces produced by
06_sequential_confidence.py.

For each clearly wrong non-terminal hop on a failed sequential trajectory:
  1) start from the same saved original sequential trace;
  2) replace that hop with MuSiQue gold;
  3) regenerate only dependency descendants, resolving # references with the
     current repaired/regenerated state;
  4) regenerate the final answer;
  5) measure change in average token log P(gold final answer).

The output carries forward raw + normalized token confidence and verbalized
confidence, so the next analysis can ask whether any confidence signal selects
the highest-value repair.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    batched_generate_with_confidence,
    batched_target_logprob,
    context_block,
    final_prompt,
    grade_answer,
    hop_prompt,
    load_musique,
    load_subject_model,
    read_jsonl,
    resolve_refs,
    write_json,
    write_jsonl,
)

REF_RE = re.compile(r"#(\d+)")


def dependency_parents(hops: list[dict]) -> list[set[int]]:
    parents: list[set[int]] = []
    for j, h in enumerate(hops):
        refs = set()
        template = str(h["template_question"])
        for m in REF_RE.finditer(template):
            p = int(m.group(1)) - 1
            if 0 <= p < j:
                refs.add(p)
        parents.append(refs)
    return parents


def descendants_of(node: int, parents: list[set[int]]) -> list[int]:
    descendants = set()
    for j in range(node + 1, len(parents)):
        if node in parents[j] or any(p in descendants for p in parents[j]):
            descendants.add(j)
    return sorted(descendants)


def target_logprob_one(model, tok, device, prompt: str, target: str, label: str) -> float:
    out = batched_target_logprob(
        model, tok, device,
        [prompt], [target],
        batch_size=1,
        progress_label=label,
    )[0]
    x = float(out["mean_logprob"])
    if not np.isfinite(x):
        raise RuntimeError(f"Non-finite target score in {label}")
    return x


def build_pairs(hops: list[dict], questions: list[str], answers: list[str]):
    return [(questions[i], answers[i]) for i in range(len(hops))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/qwen17b_sequential_seed42")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument(
        "--include-final-correct",
        action="store_true",
        help="Default focuses on failed final trajectories, matching the prior repair pilot.",
    )
    ap.add_argument(
        "--include-downstream-corrupted",
        action="store_true",
        help=(
            "Also allow repair candidates whose referenced parent hops were benchmark-incorrect. "
            "OFF by default because the MuSiQue gold answer may not be a valid repair target "
            "after an upstream error changes the downstream question."
        ),
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traces = read_jsonl(run_dir / "06_sequential_trace.jsonl")
    exp6 = json.loads((run_dir / "06_summary.json").read_text())
    model_name = args.model or exp6["environment"]["model"]

    ds = load_musique()
    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    rows = []
    state_rows = []
    review_rows = []
    n_wrong_nonterminal_before_parent_filter = 0
    n_skipped_downstream_corrupted = 0

    for qnum, tr in enumerate(traces, start=1):
        if not args.include_final_correct and tr["baseline_final_label"] != "incorrect":
            continue

        ex = ds[int(tr["dataset_idx"])]
        ctx = context_block(ex)
        hops = sorted(tr["hops"], key=lambda x: int(x["hop_idx"]))
        if len(hops) < 2:
            continue

        parents = dependency_parents(hops)
        original_questions = [str(h["model_resolved_question"]) for h in hops]
        original_answers = [str(h["pred"]) for h in hops]

        # Reconstruct + rescore common original sequential baseline safely at bs=1.
        base_prompt = final_prompt(
            tok,
            ctx,
            str(tr["main_question"]),
            build_pairs(hops, original_questions, original_answers),
        )
        base_lp = target_logprob_one(
            model, tok, device, base_prompt, str(tr["gold_final"]),
            f"seq-repair-base-{qnum}",
        )

        for h in hops:
            i = int(h["hop_idx"])
            if i >= len(hops) - 1:
                continue
            if h["label"] != "incorrect":
                continue

            n_wrong_nonterminal_before_parent_filter += 1

            # If a referenced upstream parent was benchmark-incorrect, this hop may
            # now be answering a different question from the benchmark hop. In that
            # case, injecting the benchmark gold answer is not a clean local repair.
            parents_valid = bool(
                h.get("referenced_parents_benchmark_correct", True)
            )
            if (not parents_valid) and (not args.include_downstream_corrupted):
                n_skipped_downstream_corrupted += 1
                continue

            descendants = descendants_of(i, parents)
            answers = list(original_answers)
            questions = list(original_questions)
            answers[i] = str(h["gold"])

            # After repairing Hi, regenerate only nodes that depend on it.
            for j in descendants:
                template = str(hops[j]["template_question"])
                resolved_q = resolve_refs(template, answers[:j])
                questions[j] = resolved_q
                p = hop_prompt(tok, ctx, resolved_q)
                r = batched_generate_with_confidence(
                    model, tok, device,
                    [p], batch_size=1,
                    max_new_tokens=args.max_new_tokens,
                    progress_label=f"repair-h{i+1}-regen-h{j+1}",
                )[0]
                answers[j] = str(r["answer"])

            repaired_prompt = final_prompt(
                tok,
                ctx,
                str(tr["main_question"]),
                build_pairs(hops, questions, answers),
            )
            repaired_lp = target_logprob_one(
                model, tok, device,
                repaired_prompt, str(tr["gold_final"]),
                f"repair-h{i+1}-gold-logp",
            )
            gain = repaired_lp - base_lp

            repaired_final = batched_generate_with_confidence(
                model, tok, device,
                [repaired_prompt], batch_size=1,
                max_new_tokens=args.max_new_tokens,
                progress_label=f"repair-h{i+1}-final",
            )[0]
            final_pred = str(repaired_final["answer"])
            final_label, final_f1, _ = grade_answer(
                final_pred,
                str(tr["gold_final"]),
                list(tr.get("answer_aliases", []) or []),
            )

            row = {
                "dataset_idx": int(tr["dataset_idx"]),
                "question_id": str(tr["question_id"]),
                "n_hops": int(tr["n_hops"]),
                "repair_hop_idx": i,
                "repair_hop": i + 1,
                "candidate_referenced_parents_benchmark_correct": parents_valid,
                "candidate_question_matches_gold_path": bool(
                    h.get("question_matches_gold_path", False)
                ),
                "repair_question_original": str(h["model_resolved_question"]),
                "hop_gold": str(h["gold"]),
                "hop_pred": str(h["pred"]),
                "question_matches_gold_path": bool(h.get("question_matches_gold_path", False)),
                "referenced_parents_benchmark_correct": bool(h.get("referenced_parents_benchmark_correct", False)),
                "n_descendants_regenerated": len(descendants),
                "descendants_regenerated": ",".join(str(j + 1) for j in descendants),
                "mean_logprob": h.get("mean_logprob"),
                "min_logprob": h.get("min_logprob"),
                "entropy": h.get("entropy"),
                "margin": h.get("margin"),
                "verbal_confidence": h.get("verbal_confidence"),
                "u_mean_logprob": h.get("u_mean_logprob"),
                "u_mean_logprob_z": h.get("u_mean_logprob_z"),
                "u_min_logprob": h.get("u_min_logprob"),
                "u_min_logprob_z": h.get("u_min_logprob_z"),
                "u_entropy": h.get("u_entropy"),
                "u_entropy_z": h.get("u_entropy_z"),
                "u_margin": h.get("u_margin"),
                "u_margin_z": h.get("u_margin_z"),
                "u_verbal": h.get("u_verbal"),
                "u_verbal_z": h.get("u_verbal_z"),
                "baseline_final_pred": str(tr["baseline_final_pred"]),
                "baseline_final_label": str(tr["baseline_final_label"]),
                "baseline_final_gold_logprob": base_lp,
                "repaired_final_gold_logprob": repaired_lp,
                "repair_gain": gain,
                "repaired_final_pred": final_pred,
                "repaired_final_label": final_label,
                "repaired_final_f1": float(final_f1),
            }
            rows.append(row)
            state_rows.append({
                **row,
                "repaired_questions": questions,
                "repaired_answers": answers,
            })

            if final_label == "needs_review":
                review_rows.append({
                    "question_id": str(tr["question_id"]),
                    "repair_hop": i + 1,
                    "gold_final": str(tr["gold_final"]),
                    "repaired_final_pred": final_pred,
                    "manual_label": "",
                })

    if not rows:
        raise RuntimeError("No eligible sequential repair candidates found.")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "07_repairs.csv", index=False)
    write_jsonl(run_dir / "07_propagated_repairs.jsonl", state_rows)
    pd.DataFrame(review_rows).to_csv(run_dir / "07_review_queue.csv", index=False)

    counts = df.groupby("question_id").size()
    multi = counts[counts >= 2]
    spreads = []
    for qid in multi.index:
        g = df[df["question_id"] == qid]
        spreads.append(float(g["repair_gain"].max() - g["repair_gain"].min()))

    summary = {
        "model": model_name,
        "intervention": (
            "sequential trace: gold-repair one clearly wrong non-terminal hop, "
            "regenerate dependency descendants, regenerate final answer"
        ),
        "question_filter": "baseline-final incorrect only" if not args.include_final_correct else "all",
        "candidate_filter": (
            "referenced parents benchmark-correct only"
            if not args.include_downstream_corrupted
            else "all wrong non-terminal hops, including downstream-corrupted"
        ),
        "n_wrong_nonterminal_before_parent_filter": int(
            n_wrong_nonterminal_before_parent_filter
        ),
        "n_skipped_downstream_corrupted": int(
            n_skipped_downstream_corrupted
        ),
        "n_repair_candidates": int(len(df)),
        "n_questions_with_repair_candidate": int(df["question_id"].nunique()),
        "n_questions_with_2plus_candidates": int(len(multi)),
        "candidate_count_distribution": {
            str(int(k)): int(v) for k, v in counts.value_counts().sort_index().items()
        },
        "mean_repair_gain": float(df["repair_gain"].mean()),
        "median_repair_gain": float(df["repair_gain"].median()),
        "mean_abs_repair_gain": float(df["repair_gain"].abs().mean()),
        "mean_within_question_gain_spread_2plus": float(np.mean(spreads)) if spreads else None,
        "repair_rescue_rate_all_candidates": float((df["repaired_final_label"] == "correct").mean()),
        "n_repaired_finals_needing_manual_review": int((df["repaired_final_label"] == "needs_review").sum()),
    }
    write_json(run_dir / "07_summary.json", summary)

    print("\n================ 07 SEQUENTIAL REPAIR ================")
    print(json.dumps(summary, indent=2))
    print("\nNext: python 08_sequential_policy_analysis.py --run-dir", run_dir)


if __name__ == "__main__":
    main()
