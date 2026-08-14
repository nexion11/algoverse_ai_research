#!/usr/bin/env python3
from __future__ import annotations

"""
10_tree_confidence_perturbation.py

Counterfactual branch-independence experiment.

Requires:
    outputs/qwen17b_tree_seed42/09_tree_joint_eval.jsonl

For each clean two-branch MuSiQue tree:

        Branch A ----\
                      > Merge
        Branch B ----/

we keep ALL questions and answers fixed.

We then feed the model prior self-reported branch confidences and perturb only
ONE branch's stated confidence (default factors 0.9x and 1.1x).

Primary independence test
-------------------------
If only A's confidence is changed while Branch B's evidence/question/answer are
identical, how much does the model's re-evaluated confidence for B change?

    CrossLeakage A->B = |B_perturbed - B_baseline|

Symmetrically:

    CrossLeakage B->A = |A_perturbed - A_baseline|

The merge is allowed to react because it depends on both branches. Therefore we
also record merge-confidence sensitivity separately.

Tree-of-Thoughts search connection
----------------------------------
The kyegomez/tree-of-thoughts code ranks/prunes branches by evaluation. We do
NOT switch to that package or its default model stack. Instead we emulate the
relevant decision rule on our Qwen outputs: which sibling branch has the higher
evaluation/confidence? We measure whether a small confidence perturbation flips
that selected branch.

This is a controlled robustness/independence test, not a claim that the repo
itself is being benchmarked.
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
    context_block,
    environment_metadata,
    load_musique,
    load_subject_model,
    read_jsonl,
    seed_everything,
    write_json,
    write_jsonl,
)


def apply_chat_template(tok, messages) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def parse_abc(text: str) -> tuple[float | None, float | None, float | None]:
    s = str(text)
    vals = {}
    for key in ["A", "B", "M"]:
        m = re.search(
            rf"\b{key}\s*[:=]\s*(100|[0-9]{{1,2}})(?:\.\d+)?",
            s,
            flags=re.I,
        )
        if not m:
            vals[key] = None
            continue
        try:
            # Parse full numeric token including decimal if one was printed.
            m2 = re.search(
                rf"\b{key}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                s,
                flags=re.I,
            )
            x = float(m2.group(1))
        except Exception:
            vals[key] = None
            continue
        vals[key] = x if 0.0 <= x <= 100.0 else None
    return vals["A"], vals["B"], vals["M"]


def state_eval_prompt(
    tok,
    ctx: str,
    case: dict,
    prior_a: float,
    prior_b: float,
) -> str:
    """
    Important: prior confidence is included as INPUT STATE here. This is what
    makes changing 40 -> 44 a real model intervention rather than editing a
    number after generation.
    """
    system = (
        "You are evaluating a reasoning tree. Branch A and Branch B are "
        "independent sibling branches before the merge. Their questions, answers, "
        "and evidence are fixed. You are also shown PRIOR SELF-REPORTED CONFIDENCE "
        "values. Re-evaluate correctness from the evidence. Score A and B "
        "independently: changing confidence metadata for A should not by itself "
        "change your assessment of B, and vice versa. The merge may depend on both. "
        "Return ONLY: A=<integer 0-100>; B=<integer 0-100>; M=<integer 0-100>."
    )
    user = (
        f"DOCUMENTS:\n{ctx}\n\n"
        f"BRANCH A QUESTION:\n{case['branch_a_question']}\n"
        f"BRANCH A ANSWER:\n{case['branch_a_answer']}\n"
        f"BRANCH A PRIOR CONFIDENCE:\n{prior_a:.1f}\n\n"
        f"BRANCH B QUESTION:\n{case['branch_b_question']}\n"
        f"BRANCH B ANSWER:\n{case['branch_b_answer']}\n"
        f"BRANCH B PRIOR CONFIDENCE:\n{prior_b:.1f}\n\n"
        f"MERGE QUESTION:\n{case['merge_question']}\n"
        f"MERGE ANSWER:\n{case['merge_answer']}\n\n"
        "RE-EVALUATED CONFIDENCE:"
    )
    return apply_chat_template(
        tok,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )


def run_eval(model, tok, device, prompt: str, max_new_tokens: int, label: str):
    r = batched_generate_with_confidence(
        model,
        tok,
        device,
        [prompt],
        batch_size=1,
        max_new_tokens=max_new_tokens,
        progress_label=label,
    )[0]
    raw = str(r.get("raw_answer", r.get("answer", "")))
    a, b, m = parse_abc(raw)
    return a, b, m, raw


def branch_choice(a: float | None, b: float | None, eps: float = 1e-12) -> str | None:
    if a is None or b is None:
        return None
    if not np.isfinite(float(a)) or not np.isfinite(float(b)):
        return None
    if abs(float(a) - float(b)) <= eps:
        return "tie"
    return "A" if float(a) > float(b) else "B"


def parse_factors(s: str) -> list[float]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("No perturbation factors supplied.")
    return vals


def summarize_group(g: pd.DataFrame, tolerance: float) -> dict:
    if len(g) == 0:
        return {"n": 0}
    cross = pd.to_numeric(g["cross_branch_delta"], errors="coerce")
    own = pd.to_numeric(g["own_branch_delta"], errors="coerce")
    merge = pd.to_numeric(g["merge_delta"], errors="coerce")
    valid = np.isfinite(cross) & np.isfinite(own) & np.isfinite(merge)
    gg = g.loc[valid].copy()
    cross = cross[valid]
    own = own[valid]
    merge = merge[valid]
    if len(gg) == 0:
        return {"n": 0}

    return {
        "n": int(len(gg)),
        "mean_abs_cross_branch_leakage_points": float(cross.abs().mean()),
        "median_abs_cross_branch_leakage_points": float(cross.abs().median()),
        "cross_exactly_unchanged_rate": float((cross.abs() <= 1e-12).mean()),
        f"cross_within_{tolerance:g}_points_rate": float((cross.abs() <= tolerance).mean()),
        "mean_abs_own_branch_change_points": float(own.abs().mean()),
        "mean_abs_merge_change_points": float(merge.abs().mean()),
        "reevaluated_branch_selection_flip_rate": float(
            pd.to_numeric(gg["reevaluated_selection_flip"], errors="coerce").mean()
        ),
        "input_confidence_selection_flip_rate": float(
            pd.to_numeric(gg["input_selection_flip"], errors="coerce").mean()
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tree-run-dir", default="outputs/qwen17b_tree_seed42")
    ap.add_argument("--run-dir", default="outputs/qwen17b_tree_seed42")
    ap.add_argument(
        "--factors",
        default="0.9,1.1",
        help="Comma-separated confidence multipliers. Default matches +/-10% style perturbation.",
    )
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional debug limit on cases. 0 means all.",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    factors = parse_factors(args.factors)
    cases = read_jsonl(Path(args.tree_run_dir) / "09_tree_joint_eval.jsonl")
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    if not cases:
        raise RuntimeError("No Test 09 tree cases found.")

    ds = load_musique()
    model, tok, device, dtype = load_subject_model(args.model, args.device, args.dtype)

    rows = []
    review_rows = []
    raw_rows = []

    for case_num, case in enumerate(cases, start=1):
        qid = str(case["question_id"])
        ex = ds[int(case["dataset_idx"])]
        ctx = context_block(ex)

        ca0 = case.get("branch_a_independent_confidence")
        cb0 = case.get("branch_b_independent_confidence")
        if ca0 is None or cb0 is None:
            review_rows.append({
                "question_id": qid,
                "kind": "missing_input_confidence",
                "raw": "",
                "manual_notes": "",
            })
            continue

        ca0 = float(ca0)
        cb0 = float(cb0)

        print(f"\n=== Perturbation case {case_num}/{len(cases)} | {qid} ===")

        # Baseline in the exact same prompt family used for perturbations.
        bp = state_eval_prompt(tok, ctx, case, ca0, cb0)
        ba, bb, bm, braw = run_eval(
            model, tok, device, bp, args.max_new_tokens, "tree-perturb-baseline"
        )

        if any(x is None for x in [ba, bb, bm]):
            review_rows.append({
                "question_id": qid,
                "kind": "baseline_parse",
                "raw": braw,
                "manual_notes": "",
            })
            continue

        baseline_input_choice = branch_choice(ca0, cb0)
        baseline_reeval_choice = branch_choice(ba, bb)

        raw_case = {
            "question_id": qid,
            "dataset_idx": int(case["dataset_idx"]),
            "baseline_prior_A": ca0,
            "baseline_prior_B": cb0,
            "baseline_reeval_A": ba,
            "baseline_reeval_B": bb,
            "baseline_reeval_M": bm,
            "baseline_raw": braw,
            "conditions": [],
        }

        for target in ["A", "B"]:
            for factor in factors:
                pa, pb = ca0, cb0
                if target == "A":
                    pa = max(0.0, min(100.0, ca0 * factor))
                    if abs(pa - ca0) <= 1e-12:
                        continue
                else:
                    pb = max(0.0, min(100.0, cb0 * factor))
                    if abs(pb - cb0) <= 1e-12:
                        continue

                pp = state_eval_prompt(tok, ctx, case, pa, pb)
                ra, rb, rm, rraw = run_eval(
                    model,
                    tok,
                    device,
                    pp,
                    args.max_new_tokens,
                    f"tree-perturb-{target}-{factor:g}x",
                )

                if any(x is None for x in [ra, rb, rm]):
                    review_rows.append({
                        "question_id": qid,
                        "kind": f"perturb_parse_{target}_{factor:g}",
                        "raw": rraw,
                        "manual_notes": "",
                    })
                    continue

                if target == "A":
                    own_delta = float(ra - ba)
                    cross_delta = float(rb - bb)
                else:
                    own_delta = float(rb - bb)
                    cross_delta = float(ra - ba)

                merge_delta = float(rm - bm)
                input_choice_after = branch_choice(pa, pb)
                reeval_choice_after = branch_choice(ra, rb)

                row = {
                    "dataset_idx": int(case["dataset_idx"]),
                    "question_id": qid,
                    "branch_a_hop": int(case["branch_a_hop"]),
                    "branch_b_hop": int(case["branch_b_hop"]),
                    "merge_hop": int(case["merge_hop"]),
                    "perturbed_branch": target,
                    "factor": float(factor),
                    "prior_A_before": ca0,
                    "prior_B_before": cb0,
                    "prior_A_after": pa,
                    "prior_B_after": pb,
                    "actual_input_delta": (pa - ca0) if target == "A" else (pb - cb0),
                    "baseline_reeval_A": ba,
                    "baseline_reeval_B": bb,
                    "baseline_reeval_M": bm,
                    "perturbed_reeval_A": ra,
                    "perturbed_reeval_B": rb,
                    "perturbed_reeval_M": rm,
                    "own_branch_delta": own_delta,
                    "cross_branch_delta": cross_delta,
                    "cross_branch_abs_leakage": abs(cross_delta),
                    "merge_delta": merge_delta,
                    "merge_abs_change": abs(merge_delta),
                    "baseline_input_choice": baseline_input_choice,
                    "perturbed_input_choice": input_choice_after,
                    "input_selection_flip": float(
                        baseline_input_choice != input_choice_after
                    ),
                    "baseline_reevaluated_choice": baseline_reeval_choice,
                    "perturbed_reevaluated_choice": reeval_choice_after,
                    "reevaluated_selection_flip": float(
                        baseline_reeval_choice != reeval_choice_after
                    ),
                    "raw_output": rraw,
                }
                rows.append(row)
                raw_case["conditions"].append(row)

        raw_rows.append(raw_case)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "10_tree_perturbations.csv", index=False)
    pd.DataFrame(review_rows).to_csv(run_dir / "10_review_queue.csv", index=False)
    write_jsonl(run_dir / "10_tree_perturbations.jsonl", raw_rows)

    by_direction = {}
    if len(df):
        for target in ["A", "B"]:
            by_direction[target] = summarize_group(
                df[df["perturbed_branch"] == target], args.tolerance
            )

    by_factor = {}
    if len(df):
        for factor in sorted(df["factor"].unique()):
            by_factor[str(float(factor))] = summarize_group(
                df[np.isclose(df["factor"], factor)], args.tolerance
            )

    summary = {
        "environment": environment_metadata(args.model, device, dtype),
        "design": {
            "answers_and_evidence": "held fixed in every perturbation",
            "intervention": (
                "change only one branch's prior self-reported confidence as input metadata; "
                "then re-evaluate A, B, and merge"
            ),
            "primary_metric": (
                "absolute change in the UNPERTURBED sibling branch's re-evaluated confidence"
            ),
            "merge_note": (
                "merge confidence is allowed to change because the merge depends on both branches"
            ),
            "tot_connection": (
                "branch-selection flip emulates the confidence/evaluation ranking decision used "
                "by Tree-of-Thoughts search; this script does not benchmark the external package"
            ),
        },
        "factors": factors,
        "tolerance_points": float(args.tolerance),
        "n_tree_cases_input": int(len(cases)),
        "n_perturbation_rows": int(len(df)),
        "n_parse_review_rows": int(len(review_rows)),
        "overall": summarize_group(df, args.tolerance) if len(df) else {"n": 0},
        "by_perturbed_branch": by_direction,
        "by_factor": by_factor,
    }

    write_json(run_dir / "10_summary.json", summary)

    print("\n================ 10 TREE CONFIDENCE PERTURBATION ================")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    for name in [
        "10_tree_perturbations.csv",
        "10_tree_perturbations.jsonl",
        "10_review_queue.csv",
        "10_summary.json",
    ]:
        print(f"  {run_dir / name}")


if __name__ == "__main__":
    main()
