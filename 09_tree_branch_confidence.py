#!/usr/bin/env python3
from __future__ import annotations

"""
09_tree_branch_confidence.py

Controlled Tree-of-Thought-style branch confidence experiment.

Motivation
----------
The kyegomez/tree-of-thoughts repository attaches an evaluation score to each
thought and uses those evaluations to rank/prune branches. Instead of switching
our subject model or installing that package, this experiment keeps the SAME
Qwen3-1.7B + MuSiQue pipeline and imports the structural idea:

    independent branch A ----\
                              > merge node
    independent branch B ----/

We use real MuSiQue decomposition graphs. A clean tree case is a merge hop with
exactly two direct parents whose ancestor closures are disjoint. This means the
two parent branches do not depend on each other before the merge.

The script reuses the already-generated 06 sequential traces:
- branch answers are NOT regenerated;
- independent verbalized confidence comes from Test 06;
- white-box confidence comes from Test 06.

New model calls:
1) JOINT confidence evaluation: show A and B together and ask for independent
   confidence for A, B, and the merge answer.
2) ToT-style quality evaluation: mirror the repository's self-evaluation idea
   with scores from 0.1 to 1.0 for A, B, and merge.

Primary question:
Does merely putting an unrelated sibling branch in context change the model's
confidence in the other branch?

This is a pilot / methodology experiment, not a final result.
"""

import argparse
import itertools
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
    safe_auroc_auprc,
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


def refs(template_question: str) -> list[int]:
    return [int(x) - 1 for x in re.findall(r"#(\d+)", str(template_question))]


def ancestor_closure(i: int, parents: list[list[int]], memo: dict[int, set[int]]) -> set[int]:
    if i in memo:
        return set(memo[i])
    out = set()
    for p in parents[i]:
        if 0 <= p < i:
            out.add(p)
            out |= ancestor_closure(p, parents, memo)
    memo[i] = set(out)
    return out


def find_clean_two_branch_merge(trace: dict) -> dict | None:
    """
    Return the first clean two-parent merge in a trace.

    Clean means:
    - merge directly references exactly two prior hops;
    - the two parent branches' ancestor closures are disjoint;
    - both parent endpoint questions are still locally benchmark-valid
      (their referenced parents were benchmark-correct).

    In the current 60-question seed-42 selection, this identifies the intended
    branch/fan structures without inventing synthetic data.
    """
    hops = list(trace["hops"])
    parents = [refs(h["template_question"]) for h in hops]
    memo: dict[int, set[int]] = {}

    for merge_idx, ps in enumerate(parents):
        ps = [p for p in ps if 0 <= p < merge_idx]
        if len(ps) != 2:
            continue

        a, b = ps
        closure_a = {a} | ancestor_closure(a, parents, memo)
        closure_b = {b} | ancestor_closure(b, parents, memo)

        if not closure_a.isdisjoint(closure_b):
            continue

        ha, hb = hops[a], hops[b]
        if not bool(ha.get("referenced_parents_benchmark_correct", True)):
            continue
        if not bool(hb.get("referenced_parents_benchmark_correct", True)):
            continue

        return {
            "branch_a_idx": a,
            "branch_b_idx": b,
            "merge_idx": merge_idx,
            "branch_a_closure": sorted(x + 1 for x in closure_a),
            "branch_b_closure": sorted(x + 1 for x in closure_b),
        }

    return None


def joint_confidence_prompt(tok, ctx: str, a: dict, b: dict, m: dict) -> str:
    system = (
        "You are evaluating a small reasoning tree. Branch A and Branch B are "
        "independent sibling branches before the merge. Evaluate the correctness "
        "of each supplied answer from the documents. Score A and B independently: "
        "do not lower or raise one branch's confidence merely because of the other "
        "branch. The merge may depend on both branches. Return ONLY this exact form: "
        "A=<integer 0-100>; B=<integer 0-100>; M=<integer 0-100>."
    )
    user = (
        f"DOCUMENTS:\n{ctx}\n\n"
        f"BRANCH A QUESTION:\n{a['model_resolved_question']}\n"
        f"BRANCH A ANSWER:\n{a['pred']}\n\n"
        f"BRANCH B QUESTION:\n{b['model_resolved_question']}\n"
        f"BRANCH B ANSWER:\n{b['pred']}\n\n"
        f"MERGE QUESTION:\n{m['model_resolved_question']}\n"
        f"MERGE ANSWER:\n{m['pred']}\n\n"
        "CONFIDENCE:"
    )
    return apply_chat_template(
        tok,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )


def tot_style_prompt(tok, ctx: str, a: dict, b: dict, m: dict) -> str:
    """
    Mirrors the repo's core self-evaluation concept without changing model family.
    The repo describes evaluation in the 0.1-1.0 range and uses it to rank/prune.
    """
    system = (
        "Use a Tree-of-Thoughts-style self-evaluation. Evaluate the accuracy and "
        "quality of each supplied thought/answer on a scale from 0.1 to 1.0, where "
        "0.1 is very poor and 1.0 is fully correct and well supported. Branch A and "
        "Branch B are independent before the merge, so score them independently. "
        "The merge may depend on both. Return ONLY this exact form: "
        "A=<0.1-1.0>; B=<0.1-1.0>; M=<0.1-1.0>."
    )
    user = (
        f"DOCUMENTS:\n{ctx}\n\n"
        f"BRANCH A:\nQuestion: {a['model_resolved_question']}\nAnswer: {a['pred']}\n\n"
        f"BRANCH B:\nQuestion: {b['model_resolved_question']}\nAnswer: {b['pred']}\n\n"
        f"MERGE:\nQuestion: {m['model_resolved_question']}\nAnswer: {m['pred']}\n\n"
        "EVALUATION:"
    )
    return apply_chat_template(
        tok,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )


def parse_abc(text: str, low: float, high: float) -> tuple[float | None, float | None, float | None]:
    s = str(text)
    vals = {}
    for key in ["A", "B", "M"]:
        m = re.search(
            rf"\b{key}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            s,
            flags=re.I,
        )
        if not m:
            vals[key] = None
            continue
        try:
            x = float(m.group(1))
        except ValueError:
            vals[key] = None
            continue
        vals[key] = x if low <= x <= high else None
    return vals["A"], vals["B"], vals["M"]


def corr_spearman(x, y):
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    mask = np.isfinite(sx) & np.isfinite(sy)
    if int(mask.sum()) < 3:
        return None
    return float(sx[mask].corr(sy[mask], method="spearman"))


def change_summary(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0}
    delta = pd.to_numeric(df["joint_minus_independent"], errors="coerce")
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return {"n": 0}
    return {
        "n": int(len(delta)),
        "mean_signed_change_points": float(delta.mean()),
        "mean_abs_change_points": float(delta.abs().mean()),
        "median_abs_change_points": float(delta.abs().median()),
        "exact_same_rate": float((delta.abs() <= 1e-12).mean()),
        "within_5_points_rate": float((delta.abs() <= 5.0).mean()),
    }


def error_detection_from_score(df: pd.DataFrame, score_col: str) -> dict:
    d = df[
        df["label"].isin(["correct", "incorrect"])
        & np.isfinite(pd.to_numeric(df[score_col], errors="coerce"))
    ].copy()
    if len(d) == 0:
        return {"n": 0, "auroc": None, "auprc": None}
    # Larger score = MORE confident, so uncertainty is negative score.
    uncertainty = -pd.to_numeric(d[score_col], errors="coerce").to_numpy(dtype=float)
    y = (d["label"] == "incorrect").astype(int).to_numpy()
    return safe_auroc_auprc(uncertainty, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seq-run-dir",
        default="outputs/qwen17b_sequential_seed42",
        help="Must contain 06_sequential_trace.jsonl from the completed sequential experiment.",
    )
    ap.add_argument("--run-dir", default="outputs/qwen17b_tree_seed42")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional debug limit on tree cases. 0 means all eligible cases.",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    traces = read_jsonl(Path(args.seq_run_dir) / "06_sequential_trace.jsonl")
    ds = load_musique()

    structures = []
    for tr in traces:
        found = find_clean_two_branch_merge(tr)
        if found is None:
            continue
        structures.append({**found, "trace": tr})

    if args.limit and args.limit > 0:
        structures = structures[: args.limit]

    print(f"Eligible clean two-branch merge structures: {len(structures)}")
    if not structures:
        raise RuntimeError("No clean two-branch merge structures found.")

    model, tok, device, dtype = load_subject_model(args.model, args.device, args.dtype)

    rows = []
    raw_rows = []
    review_rows = []

    for case_num, item in enumerate(structures, start=1):
        tr = item["trace"]
        ex = ds[int(tr["dataset_idx"])]
        ctx = context_block(ex)
        hops = tr["hops"]

        ai = int(item["branch_a_idx"])
        bi = int(item["branch_b_idx"])
        mi = int(item["merge_idx"])
        a, b, m = hops[ai], hops[bi], hops[mi]

        print(
            f"\n=== Tree case {case_num}/{len(structures)} | "
            f"{tr['question_id']} | A=H{ai+1} B=H{bi+1} M=H{mi+1} ==="
        )

        jp = joint_confidence_prompt(tok, ctx, a, b, m)
        jr = batched_generate_with_confidence(
            model,
            tok,
            device,
            [jp],
            batch_size=1,
            max_new_tokens=args.max_new_tokens,
            progress_label="tree-joint-conf",
        )[0]
        joint_raw = str(jr.get("raw_answer", jr.get("answer", "")))
        ja, jb, jm = parse_abc(joint_raw, 0.0, 100.0)

        tp = tot_style_prompt(tok, ctx, a, b, m)
        trr = batched_generate_with_confidence(
            model,
            tok,
            device,
            [tp],
            batch_size=1,
            max_new_tokens=args.max_new_tokens,
            progress_label="tree-tot-eval",
        )[0]
        tot_raw = str(trr.get("raw_answer", trr.get("answer", "")))
        ta, tb, tm = parse_abc(tot_raw, 0.1, 1.0)

        case_base = {
            "dataset_idx": int(tr["dataset_idx"]),
            "question_id": str(tr["question_id"]),
            "n_hops": int(tr["n_hops"]),
            "branch_a_hop": ai + 1,
            "branch_b_hop": bi + 1,
            "merge_hop": mi + 1,
            "branch_a_closure": ",".join(map(str, item["branch_a_closure"])),
            "branch_b_closure": ",".join(map(str, item["branch_b_closure"])),
        }

        for role, h, independent, joint, tot in [
            ("A", a, a.get("verbal_confidence"), ja, ta),
            ("B", b, b.get("verbal_confidence"), jb, tb),
        ]:
            independent = float(independent) if independent is not None else np.nan
            joint = float(joint) if joint is not None else np.nan
            tot100 = float(tot) * 100.0 if tot is not None else np.nan

            rows.append({
                **case_base,
                "branch_role": role,
                "hop": int(h["hop"]),
                "question": str(h["model_resolved_question"]),
                "answer": str(h["pred"]),
                "gold": str(h["gold"]),
                "label": str(h["label"]),
                "mean_logprob": float(h["mean_logprob"]),
                "min_logprob": float(h["min_logprob"]),
                "entropy": float(h["entropy"]),
                "margin": float(h["margin"]),
                "independent_verbal_confidence": independent,
                "joint_verbal_confidence": joint,
                "joint_minus_independent": (
                    joint - independent if np.isfinite(joint) and np.isfinite(independent) else np.nan
                ),
                "tot_style_evaluation": float(tot) if tot is not None else np.nan,
                "tot_style_evaluation_100": tot100,
            })

        raw_rows.append({
            **case_base,
            "branch_a_question": str(a["model_resolved_question"]),
            "branch_a_answer": str(a["pred"]),
            "branch_a_label": str(a["label"]),
            "branch_a_independent_confidence": a.get("verbal_confidence"),
            "branch_b_question": str(b["model_resolved_question"]),
            "branch_b_answer": str(b["pred"]),
            "branch_b_label": str(b["label"]),
            "branch_b_independent_confidence": b.get("verbal_confidence"),
            "merge_question": str(m["model_resolved_question"]),
            "merge_answer": str(m["pred"]),
            "merge_label": str(m["label"]),
            "merge_independent_confidence": m.get("verbal_confidence"),
            "joint_A": ja,
            "joint_B": jb,
            "joint_M": jm,
            "joint_raw": joint_raw,
            "tot_A": ta,
            "tot_B": tb,
            "tot_M": tm,
            "tot_raw": tot_raw,
        })

        if any(x is None for x in [ja, jb, jm, ta, tb, tm]):
            review_rows.append({
                "question_id": str(tr["question_id"]),
                "branch_a_hop": ai + 1,
                "branch_b_hop": bi + 1,
                "merge_hop": mi + 1,
                "joint_raw": joint_raw,
                "tot_raw": tot_raw,
                "manual_notes": "",
            })

    branch_df = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw_rows)

    branch_df.to_csv(run_dir / "09_tree_branches.csv", index=False)
    raw_df.to_csv(run_dir / "09_tree_cases.csv", index=False)
    pd.DataFrame(review_rows).to_csv(run_dir / "09_review_queue.csv", index=False)
    write_jsonl(run_dir / "09_tree_joint_eval.jsonl", raw_rows)

    indep = pd.to_numeric(branch_df["independent_verbal_confidence"], errors="coerce")
    joint = pd.to_numeric(branch_df["joint_verbal_confidence"], errors="coerce")
    tot100 = pd.to_numeric(branch_df["tot_style_evaluation_100"], errors="coerce")

    summary = {
        "environment": environment_metadata(args.model, device, dtype),
        "design": {
            "source": "real MuSiQue dependency graphs from the same 60-question sequential pilot",
            "tree_definition": (
                "a merge hop with exactly two direct parent branches whose ancestor closures "
                "are disjoint; sibling branches do not depend on each other before merge"
            ),
            "answers": "reused from Test 06; branch answers are not regenerated in Test 09",
            "independent_confidence": "reused separate post-answer 0-100 verbal confidence from Test 06",
            "joint_confidence": (
                "A/B/M evaluated together; prompt explicitly asks A and B to be scored independently"
            ),
            "tot_style": (
                "0.1-1.0 self-evaluation inspired by kyegomez/tree-of-thoughts; same Qwen subject "
                "model retained rather than switching to the repository's default API stack"
            ),
        },
        "n_tree_cases": int(len(raw_df)),
        "n_branch_endpoints": int(len(branch_df)),
        "n_parse_review_cases": int(len(review_rows)),
        "independent_vs_joint": {
            "all_branches": change_summary(branch_df),
            "branch_A": change_summary(branch_df[branch_df["branch_role"] == "A"]),
            "branch_B": change_summary(branch_df[branch_df["branch_role"] == "B"]),
            "spearman_independent_vs_joint": corr_spearman(indep, joint),
        },
        "error_detection_on_branch_endpoints": {
            "whitebox_mean_logprob": error_detection_from_score(
                branch_df.assign(score_for_metric=branch_df["mean_logprob"]),
                "score_for_metric",
            ),
            "independent_verbal": error_detection_from_score(
                branch_df, "independent_verbal_confidence"
            ),
            "joint_verbal": error_detection_from_score(
                branch_df, "joint_verbal_confidence"
            ),
            "tot_style_evaluation": error_detection_from_score(
                branch_df, "tot_style_evaluation_100"
            ),
        },
        "score_discreteness": {
            "independent_verbal_unique_values": int(indep.nunique(dropna=True)),
            "joint_verbal_unique_values": int(joint.nunique(dropna=True)),
            "tot_style_unique_values": int(tot100.nunique(dropna=True)),
        },
    }

    write_json(run_dir / "09_summary.json", summary)

    print("\n================ 09 TREE BRANCH CONFIDENCE ================")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    for name in [
        "09_tree_branches.csv",
        "09_tree_cases.csv",
        "09_tree_joint_eval.jsonl",
        "09_review_queue.csv",
        "09_summary.json",
    ]:
        print(f"  {run_dir / name}")


if __name__ == "__main__":
    main()
