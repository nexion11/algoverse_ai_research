#!/usr/bin/env python3
from __future__ import annotations

"""
02_repair_value.py

Main Basic Test 2: confidence vs. intervention-defined repair value.

What this script does
---------------------
It reuses the fixed hop traces produced by 01_confidence_correctness.py.

For every CLEARLY INCORRECT, NON-TERMINAL hop on an eligible question:

1. Start from the same saved Test-1 trace.
2. Replace only that wrong hop's answer with the MuSiQue gold answer.
3. MAIN intervention (propagated repair):
      Regenerate every downstream decomposition node that depends on the
      repaired hop, using the newly repaired/regenerated answers to resolve
      MuSiQue #1/#2/... references.
4. Regenerate the final answer.
5. Measure:
      repair_gain =
          avg log P(gold final | propagated repaired trace)
        - avg log P(gold final | original trace)

It ALSO computes a frozen-repair control:
    replace the wrong hop with gold, keep every other hop fixed,
    then rescore the final answer.

Important design choices
------------------------
- Test 1 used gold-resolved benchmark subquestions to make hop correctness
  independently gradeable. We keep those saved answers as the common original
  trace for ranking candidate errors.
- Propagation is dependency-aware: only descendants of the repaired node are
  regenerated, not unrelated branches.
- The last decomposition hop is excluded as a repair candidate because in
  MuSiQue it is typically the answer-bearing terminal state; directly replacing
  it with gold would make repair selection artificially easy.
- By default, we analyze baseline-final INCORRECT questions, because the main
  practical question is which error to repair on a failed trajectory.
- On Apple MPS, score-batch-size=1 is the safe default because the earlier
  batch-2 teacher-forced scoring produced NaNs.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
    """
    Parse MuSiQue references.

    Example:
        hop 3 template: "where was #2 born?"
        -> parents[2] == {1}

    Returned parent indices are 0-based.
    """
    parents: list[set[int]] = []

    for j, h in enumerate(hops):
        refs = set()

        for match in REF_RE.finditer(str(h["template_question"])):
            parent = int(match.group(1)) - 1

            if 0 <= parent < j:
                refs.add(parent)

        parents.append(refs)

    return parents


def descendants_of(node: int, parents: list[set[int]]) -> list[int]:
    """
    Return all transitive descendants of `node`, in decomposition order.
    """
    descendants = set()

    for j in range(node + 1, len(parents)):
        if node in parents[j] or any(p in descendants for p in parents[j]):
            descendants.add(j)

    return sorted(descendants)


def json_safe_float(x):
    try:
        x = float(x)
    except Exception:
        return None

    return x if np.isfinite(x) else None


def safe_target_logprob(
    model,
    tok,
    device,
    prompts: list[str],
    targets: list[str],
    batch_size: int,
    label: str,
):
    """
    Run common.batched_target_logprob, but protect the experiment from the
    NaN issue previously observed with batched MPS teacher-forced scoring.

    If any result is non-finite and batch_size > 1, retry ONLY those examples
    one at a time. If batch-size-1 still gives a non-finite value, fail loudly
    rather than silently writing a broken repair gain.
    """
    if not prompts:
        return []

    scores = batched_target_logprob(
        model,
        tok,
        device,
        prompts,
        targets,
        batch_size=batch_size,
        progress_label=label,
    )

    bad = [
        i
        for i, s in enumerate(scores)
        if s is None or not np.isfinite(float(s["mean_logprob"]))
    ]

    if bad and batch_size > 1:
        print(
            f"\nWARNING: {len(bad)} non-finite score(s) in {label}. "
            "Retrying those items with batch size 1..."
        )

        retry_prompts = [prompts[i] for i in bad]
        retry_targets = [targets[i] for i in bad]

        retry_scores = batched_target_logprob(
            model,
            tok,
            device,
            retry_prompts,
            retry_targets,
            batch_size=1,
            progress_label=f"{label}-retry-bs1",
        )

        for original_i, retry_s in zip(bad, retry_scores):
            scores[original_i] = retry_s

    still_bad = [
        i
        for i, s in enumerate(scores)
        if s is None or not np.isfinite(float(s["mean_logprob"]))
    ]

    if still_bad:
        raise RuntimeError(
            f"{label}: non-finite gold logprob remained at indices "
            f"{still_bad[:10]} even after safe scoring. "
            "Stop here rather than computing invalid repair gains."
        )

    return scores


def build_baseline_prompt(tok, ex, tr: dict) -> str:
    """
    Reconstruct the same style of fixed structured trace used by Test 1.
    """
    ctx = context_block(ex)

    hops = sorted(
        tr["hops"],
        key=lambda x: int(x["hop_idx"])
    )

    pairs = [
        (h["resolved_question"], h["pred"])
        for h in hops
    ]

    return final_prompt(
        tok,
        ctx,
        tr["main_question"],
        pairs
    )


def make_repair_states(
    traces: list[dict],
    ds,
    only_failed: bool
) -> list[dict]:

    """
    Create one independent intervention state per wrong non-terminal hop.

    Every repair candidate within the same question uses the same original
    Test-1 trace as its baseline.
    """

    states = []

    for tr in traces:

        if only_failed and tr["baseline_final_label"] != "incorrect":
            continue

        hops = sorted(
            tr["hops"],
            key=lambda x: int(x["hop_idx"])
        )

        if len(hops) < 2:
            continue

        parents = dependency_parents(hops)

        for h in hops:

            i = int(h["hop_idx"])

            # Candidate must be clearly wrong.
            if h["label"] != "incorrect":
                continue

            # Exclude terminal decomposition step.
            if i >= len(hops) - 1:
                continue

            descendants = descendants_of(
                i,
                parents
            )

            state = {

                "dataset_idx":
                    int(tr["dataset_idx"]),

                "question_id":
                    str(tr["question_id"]),

                "n_hops":
                    int(tr["n_hops"]),

                "main_question":
                    str(tr["main_question"]),

                "gold_final":
                    str(tr["gold_final"]),

                "answer_aliases":
                    list(
                        tr.get("answer_aliases", [])
                        or []
                    ),

                "baseline_final_pred":
                    str(tr["baseline_final_pred"]),

                "baseline_final_label":
                    str(tr["baseline_final_label"]),

                "repair_hop_idx":
                    i,

                "repair_hop":
                    i + 1,

                "repair_question":
                    str(h["resolved_question"]),

                "hop_gold":
                    str(h["gold"]),

                "hop_pred":
                    str(h["pred"]),

                # Original confidence signals.
                "mean_logprob":
                    json_safe_float(
                        h["mean_logprob"]
                    ),

                "min_logprob":
                    json_safe_float(
                        h["min_logprob"]
                    ),

                "entropy":
                    json_safe_float(
                        h["entropy"]
                    ),

                "margin":
                    json_safe_float(
                        h["margin"]
                    ),

                # Dependency information.
                "parents":
                    [
                        sorted(list(x))
                        for x in parents
                    ],

                "descendants":
                    descendants,

                # Mutable propagated trace.
                "questions":
                    [
                        str(x["resolved_question"])
                        for x in hops
                    ],

                "answers":
                    [
                        str(x["pred"])
                        for x in hops
                    ],

                # Original fixed trace.
                "original_questions":
                    [
                        str(x["resolved_question"])
                        for x in hops
                    ],

                "original_answers":
                    [
                        str(x["pred"])
                        for x in hops
                    ],

                "templates":
                    [
                        str(x["template_question"])
                        for x in hops
                    ],

                "golds":
                    [
                        str(x["gold"])
                        for x in hops
                    ],

                "propagation_records":
                    [],
            }

            # ----------------------------------
            # THE ACTUAL REPAIR INTERVENTION
            # ----------------------------------
            state["answers"][i] = str(
                h["gold"]
            )

            states.append(state)

    return states


def propagate_repairs(
    states: list[dict],
    ds,
    model,
    tok,
    device,
    generate_batch_size: int,
    max_new_tokens: int,
    checkpoint_path: Path | None = None,
):
    """
    Regenerate downstream descendants after each repair.

    Processing proceeds one decomposition level at a time so causal order is
    preserved. Independent interventions at the same level can still be
    batched if desired.
    """

    if not states:
        return

    max_hops = max(
        s["n_hops"]
        for s in states
    )

    for j in range(1, max_hops):

        active = [
            s
            for s in states
            if j in s["descendants"]
        ]

        if not active:
            continue

        prompts = []
        resolved_questions = []

        for s in active:

            ex = ds[
                s["dataset_idx"]
            ]

            ctx = context_block(ex)

            # Resolve #1/#2/etc. using the CURRENT
            # counterfactual state.
            q = resolve_refs(
                s["templates"][j],
                s["answers"][:j],
            )

            resolved_questions.append(q)

            prompts.append(
                hop_prompt(
                    tok,
                    ctx,
                    q
                )
            )

        print(
            f"\nPropagating decomposition hop {j + 1}: "
            f"{len(active)} counterfactual state(s)"
        )

        outputs = batched_generate_with_confidence(
            model,
            tok,
            device,
            prompts,
            batch_size=generate_batch_size,
            max_new_tokens=max_new_tokens,
            progress_label=f"propagate-hop-{j + 1}",
        )

        for s, q, out in zip(
            active,
            resolved_questions,
            outputs
        ):

            old_answer = s["answers"][j]

            s["questions"][j] = q
            s["answers"][j] = out["answer"]

            s["propagation_records"].append(
                {
                    "hop_idx": j,
                    "hop": j + 1,
                    "resolved_question": q,
                    "old_original_answer": old_answer,
                    "regenerated_answer":
                        out["answer"],
                    "mean_logprob":
                        json_safe_float(
                            out["mean_logprob"]
                        ),
                    "min_logprob":
                        json_safe_float(
                            out["min_logprob"]
                        ),
                    "entropy":
                        json_safe_float(
                            out["entropy"]
                        ),
                    "margin":
                        json_safe_float(
                            out["margin"]
                        ),
                }
            )

        if checkpoint_path is not None:

            checkpoint_rows = []

            for s in states:

                checkpoint_rows.append(
                    {
                        "question_id":
                            s["question_id"],
                        "repair_hop":
                            s["repair_hop"],
                        "descendants":
                            s["descendants"],
                        "questions":
                            s["questions"],
                        "answers":
                            s["answers"],
                        "propagation_records":
                            s["propagation_records"],
                    }
                )

            write_jsonl(
                checkpoint_path,
                checkpoint_rows
            )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--run-dir",
        default="outputs/qwen17b_seed42"
    )

    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Defaults to the model recorded "
            "by experiment 01."
        )
    )

    ap.add_argument(
        "--device",
        default="auto"
    )

    ap.add_argument(
        "--dtype",
        default="auto"
    )

    # Your M4 Pro benchmark favored generation batch size 1.
    ap.add_argument(
        "--generate-batch-size",
        type=int,
        default=1
    )

    # Safe default after MPS NaN issue.
    ap.add_argument(
        "--score-batch-size",
        type=int,
        default=1
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=16
    )

    ap.add_argument(
        "--include-final-correct",
        action="store_true",
        help=(
            "Also analyze traces whose original "
            "final answer was already correct. "
            "Default is failed-final traces only."
        ),
    )

    args = ap.parse_args()

    run_dir = Path(
        args.run_dir
    )

    trace_path = (
        run_dir
        / "01_trace.jsonl"
    )

    summary1_path = (
        run_dir
        / "01_summary.json"
    )

    if not trace_path.exists():

        raise FileNotFoundError(
            f"Missing {trace_path}. "
            "Run 01_confidence_correctness.py first."
        )

    if not summary1_path.exists():

        raise FileNotFoundError(
            f"Missing {summary1_path}. "
            "Run 01_confidence_correctness.py first."
        )

    traces = read_jsonl(
        trace_path
    )

    exp1 = json.loads(
        summary1_path.read_text()
    )

    model_name = (
        args.model
        or exp1["environment"]["model"]
    )

    print("\n==============================================")
    print("BASIC TEST 2 — PROPAGATED REPAIR VALUE")
    print("==============================================")

    print(
        f"Run dir: {run_dir}"
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Saved Test-1 traces: {len(traces)}"
    )

    print(
        f"Generation batch size: "
        f"{args.generate_batch_size}"
    )

    print(
        f"Gold-logprob score batch size: "
        f"{args.score_batch_size}"
    )

    print(
        "Question filter: "
        + (
            "all final outcomes"
            if args.include_final_correct
            else "baseline-final incorrect only"
        )
    )

    ds = load_musique()

    model, tok, device, dtype = (
        load_subject_model(
            model_name,
            args.device,
            args.dtype,
        )
    )

    # ==========================================================
    # A. RECOMPUTE BASELINE FINAL GOLD LOGPROBS SAFELY
    # ==========================================================

    eligible_traces = [
        tr
        for tr in traces
        if (
            args.include_final_correct
            or tr["baseline_final_label"]
            == "incorrect"
        )
    ]

    if not eligible_traces:

        raise RuntimeError(
            "No eligible traces. "
            "Check 01_trace.jsonl."
        )

    baseline_prompts = []
    baseline_targets = []

    for tr in eligible_traces:

        ex = ds[
            int(tr["dataset_idx"])
        ]

        baseline_prompts.append(
            build_baseline_prompt(
                tok,
                ex,
                tr
            )
        )

        baseline_targets.append(
            str(
                tr["gold_final"]
            )
        )

    print(
        f"\nRe-scoring "
        f"{len(eligible_traces)} "
        f"original final states "
        f"with safe teacher-forced "
        f"gold logprob..."
    )

    baseline_scores = (
        safe_target_logprob(
            model,
            tok,
            device,
            baseline_prompts,
            baseline_targets,
            batch_size=
                args.score_batch_size,
            label=
                "baseline-gold-logp",
        )
    )

    baseline_lp_by_qid = {

        str(tr["question_id"]):
            float(
                score["mean_logprob"]
            )

        for tr, score in zip(
            eligible_traces,
            baseline_scores
        )
    }

    pd.DataFrame(
        [
            {
                "question_id":
                    str(tr["question_id"]),

                "dataset_idx":
                    int(tr["dataset_idx"]),

                "baseline_final_label":
                    tr[
                        "baseline_final_label"
                    ],

                "baseline_final_pred":
                    tr[
                        "baseline_final_pred"
                    ],

                "gold_final":
                    tr["gold_final"],

                "baseline_final_gold_logprob_recomputed":
                    baseline_lp_by_qid[
                        str(
                            tr[
                                "question_id"
                            ]
                        )
                    ],
            }

            for tr
            in eligible_traces
        ]
    ).to_csv(
        run_dir
        / "02_baseline_rescored.csv",
        index=False
    )

    # ==========================================================
    # B. CREATE REPAIR INTERVENTIONS
    # ==========================================================

    states = make_repair_states(
        traces,
        ds,
        only_failed=
            not args.include_final_correct,
    )

    if not states:

        raise RuntimeError(
            "No clearly incorrect "
            "non-terminal repair candidates "
            "were found."
        )

    print(
        f"\nRepair candidates: "
        f"{len(states)}"
    )

    print(
        "Questions with >=1 candidate:",
        len(
            set(
                s["question_id"]
                for s in states
            )
        ),
    )

    candidate_counts = Counter(
        s["question_id"]
        for s in states
    )

    print(
        "Questions with >=2 candidates:",
        sum(
            1
            for n
            in candidate_counts.values()
            if n >= 2
        ),
    )

    # ==========================================================
    # C. MAIN INTERVENTION:
    # REPAIR + PROPAGATE
    # ==========================================================

    propagate_repairs(
        states,
        ds,
        model,
        tok,
        device,
        generate_batch_size=
            args.generate_batch_size,
        max_new_tokens=
            args.max_new_tokens,
        checkpoint_path=
            run_dir
            / "02_propagation_checkpoint.jsonl",
    )

    # ==========================================================
    # D. BUILD PROPAGATED AND FROZEN FINAL PROMPTS
    # ==========================================================

    propagated_prompts = []
    frozen_prompts = []
    targets = []

    for s in states:

        ex = ds[
            s["dataset_idx"]
        ]

        ctx = context_block(ex)

        # ----------------------------
        # PROPAGATED STATE
        # ----------------------------

        propagated_pairs = list(
            zip(
                s["questions"],
                s["answers"]
            )
        )

        propagated_prompts.append(
            final_prompt(
                tok,
                ctx,
                s["main_question"],
                propagated_pairs
            )
        )

        # ----------------------------
        # FROZEN CONTROL
        # ----------------------------

        frozen_answers = list(
            s["original_answers"]
        )

        frozen_answers[
            s["repair_hop_idx"]
        ] = s["hop_gold"]

        frozen_pairs = list(
            zip(
                s["original_questions"],
                frozen_answers
            )
        )

        frozen_prompts.append(
            final_prompt(
                tok,
                ctx,
                s["main_question"],
                frozen_pairs
            )
        )

        targets.append(
            s["gold_final"]
        )

    # ==========================================================
    # E. SCORE REPAIR GAINS
    # ==========================================================

    print(
        f"\nScoring {len(states)} "
        f"PROPAGATED repaired states "
        f"against the gold final answer..."
    )

    propagated_scores = (
        safe_target_logprob(
            model,
            tok,
            device,
            propagated_prompts,
            targets,
            batch_size=
                args.score_batch_size,
            label=
                "propagated-repair-gold-logp",
        )
    )

    print(
        f"\nScoring {len(states)} "
        f"FROZEN repaired states "
        f"against the gold final answer..."
    )

    frozen_scores = (
        safe_target_logprob(
            model,
            tok,
            device,
            frozen_prompts,
            targets,
            batch_size=
                args.score_batch_size,
            label=
                "frozen-repair-gold-logp",
        )
    )

    # ==========================================================
    # F. GENERATE PROPAGATED FINALS
    # ==========================================================

    print(
        f"\nGenerating {len(states)} "
        f"propagated repaired final answers "
        f"for wrong->correct rescue analysis..."
    )

    propagated_final_outputs = (
        batched_generate_with_confidence(
            model,
            tok,
            device,
            propagated_prompts,
            batch_size=
                args.generate_batch_size,
            max_new_tokens=
                args.max_new_tokens,
            progress_label=
                "propagated-final",
        )
    )

    # ==========================================================
    # G. SAVE PER-REPAIR RESULTS
    # ==========================================================

    rows = []
    detailed_rows = []
    review_rows = []

    for (
        s,
        prop_score,
        frozen_score,
        final_out
    ) in zip(
        states,
        propagated_scores,
        frozen_scores,
        propagated_final_outputs,
    ):

        baseline_lp = (
            baseline_lp_by_qid[
                s["question_id"]
            ]
        )

        propagated_lp = float(
            prop_score[
                "mean_logprob"
            ]
        )

        frozen_lp = float(
            frozen_score[
                "mean_logprob"
            ]
        )

        propagated_gain = (
            propagated_lp
            - baseline_lp
        )

        frozen_gain = (
            frozen_lp
            - baseline_lp
        )

        (
            final_label,
            final_f1,
            matched_gold
        ) = grade_answer(
            final_out["answer"],
            s["gold_final"],
            s["answer_aliases"],
        )

        if final_label == "needs_review":

            review_rows.append(
                {
                    "kind":
                        "propagated_repair_final",

                    "question_id":
                        s["question_id"],

                    "repair_hop":
                        s["repair_hop"],

                    "question":
                        s["main_question"],

                    "gold":
                        s["gold_final"],

                    "pred":
                        final_out["answer"],

                    "manual_label":
                        "",
                }
            )

        row = {

            "dataset_idx":
                s["dataset_idx"],

            "question_id":
                s["question_id"],

            "n_hops":
                s["n_hops"],

            "repair_hop_idx":
                s["repair_hop_idx"],

            "repair_hop":
                s["repair_hop"],

            "repair_question":
                s["repair_question"],

            "hop_gold":
                s["hop_gold"],

            "hop_pred":
                s["hop_pred"],

            # Original confidence signal.
            "mean_logprob":
                s["mean_logprob"],

            "min_logprob":
                s["min_logprob"],

            "entropy":
                s["entropy"],

            "margin":
                s["margin"],

            "n_descendants_regenerated":
                len(
                    s["descendants"]
                ),

            "descendants_regenerated":
                json.dumps(
                    [
                        x + 1
                        for x
                        in s["descendants"]
                    ]
                ),

            "baseline_final_pred":
                s["baseline_final_pred"],

            "baseline_final_label":
                s["baseline_final_label"],

            "baseline_final_gold_logprob":
                baseline_lp,

            # ==========================
            # MAIN RESULT
            # ==========================
            "repaired_final_gold_logprob":
                propagated_lp,

            # IMPORTANT:
            # 03_policy_analysis.py
            # reads THIS column.
            "repair_gain":
                propagated_gain,

            "propagated_gain":
                propagated_gain,

            "repaired_final_pred":
                final_out["answer"],

            "repaired_final_label":
                final_label,

            "repaired_final_f1":
                final_f1,

            # ==========================
            # FROZEN CONTROL
            # ==========================
            "frozen_final_gold_logprob":
                frozen_lp,

            "frozen_gain":
                frozen_gain,
        }

        rows.append(row)

        detailed_rows.append(
            {
                **row,

                "original_questions":
                    s[
                        "original_questions"
                    ],

                "original_answers":
                    s[
                        "original_answers"
                    ],

                "propagated_questions":
                    s["questions"],

                "propagated_answers":
                    s["answers"],

                "propagation_records":
                    s[
                        "propagation_records"
                    ],

                "gold_final":
                    s["gold_final"],

                "matched_gold":
                    matched_gold,
            }
        )

    df = pd.DataFrame(
        rows
    ).sort_values(
        [
            "dataset_idx",
            "repair_hop_idx"
        ]
    )

    df.to_csv(
        run_dir
        / "02_repairs.csv",
        index=False
    )

    write_jsonl(
        run_dir
        / "02_propagated_repairs.jsonl",
        detailed_rows
    )

    review_df = pd.DataFrame(
        review_rows
    )

    review_df.to_csv(
        run_dir
        / "02_review_queue.csv",
        index=False
    )

    # ==========================================================
    # H. SUMMARY
    # ==========================================================

    grouped_counts = (
        df
        .groupby("question_id")
        .size()
    )

    eligible_qids = (
        grouped_counts[
            grouped_counts >= 2
        ].index
    )

    multi = df[
        df["question_id"].isin(
            eligible_qids
        )
    ].copy()

    spreads = []

    for qid, g in multi.groupby(
        "question_id"
    ):

        spreads.append(
            float(
                g["repair_gain"].max()
                - g["repair_gain"].min()
            )
        )

    if len(df) >= 3:

        rho_all, rho_all_p = (
            spearmanr(
                df["frozen_gain"],
                df["propagated_gain"],
                nan_policy="omit",
            )
        )

    else:

        rho_all = np.nan
        rho_all_p = np.nan

    candidate_distribution = {

        str(int(k)):
            int(v)

        for k, v
        in grouped_counts
        .value_counts()
        .sort_index()
        .items()
    }

    summary = {

        "model":
            model_name,

        "intervention": {

            "primary":
                (
                    "gold-repair one wrong "
                    "non-terminal hop, then "
                    "regenerate its dependency "
                    "descendants and regenerate "
                    "the final answer"
                ),

            "secondary":
                (
                    "frozen one-hop gold repair "
                    "with all other intermediate "
                    "answers held fixed"
                ),

            "repair_gain_definition":
                (
                    "average gold-final token "
                    "logprob after repair minus "
                    "average gold-final token "
                    "logprob in the common "
                    "original trace"
                ),
        },

        "question_filter":
            (
                "all final outcomes"
                if args.include_final_correct
                else
                "baseline-final incorrect only"
            ),

        "n_baseline_questions_rescored":
            int(
                len(
                    eligible_traces
                )
            ),

        "n_repair_candidates":
            int(
                len(df)
            ),

        "n_questions_with_repair_candidate":
            int(
                df[
                    "question_id"
                ].nunique()
            ),

        "n_questions_with_2plus_candidates":
            int(
                len(
                    eligible_qids
                )
            ),

        "candidate_count_distribution":
            candidate_distribution,

        "mean_propagated_repair_gain":
            float(
                df[
                    "propagated_gain"
                ].mean()
            ),

        "median_propagated_repair_gain":
            float(
                df[
                    "propagated_gain"
                ].median()
            ),

        "mean_abs_propagated_repair_gain":
            float(
                df[
                    "propagated_gain"
                ]
                .abs()
                .mean()
            ),

        "mean_frozen_repair_gain":
            float(
                df[
                    "frozen_gain"
                ].mean()
            ),

        "median_frozen_repair_gain":
            float(
                df[
                    "frozen_gain"
                ].median()
            ),

        "mean_abs_frozen_repair_gain":
            float(
                df[
                    "frozen_gain"
                ]
                .abs()
                .mean()
            ),

        "mean_within_question_propagated_gain_spread_2plus":
            (
                float(
                    np.mean(
                        spreads
                    )
                )
                if spreads
                else None
            ),

        "frozen_vs_propagated_spearman":
            (
                float(
                    rho_all
                )
                if np.isfinite(
                    rho_all
                )
                else None
            ),

        "frozen_vs_propagated_spearman_p":
            (
                float(
                    rho_all_p
                )
                if np.isfinite(
                    rho_all_p
                )
                else None
            ),

        "propagated_repair_rescue_rate_all_candidates":
            float(
                (
                    df[
                        "repaired_final_label"
                    ]
                    == "correct"
                ).mean()
            ),

        "n_repaired_finals_needing_manual_review":
            int(
                (
                    df[
                        "repaired_final_label"
                    ]
                    == "needs_review"
                ).sum()
            ),
    }

    write_json(
        run_dir
        / "02_summary.json",
        summary
    )

    print("\n==============================================")
    print("BASIC TEST 2 SUMMARY")
    print("==============================================")

    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    print("\nSaved:")

    print(
        f"  "
        f"{run_dir / '02_baseline_rescored.csv'}"
    )

    print(
        f"  "
        f"{run_dir / '02_repairs.csv'}"
    )

    print(
        f"  "
        f"{run_dir / '02_propagated_repairs.jsonl'}"
    )

    print(
        f"  "
        f"{run_dir / '02_summary.json'}"
    )

    print(
        f"  "
        f"{run_dir / '02_review_queue.csv'}"
    )

    print(
        f"  "
        f"{run_dir / '02_propagation_checkpoint.jsonl'}"
    )

    if len(eligible_qids) < 10:

        print(
            "\nWARNING: fewer than 10 questions "
            "have >=2 wrong non-terminal repair "
            "candidates. Treat ranking results "
            "as methodology/debugging evidence "
            "only, not a scientific conclusion."
        )

    print(
        "\nNext, run 03_policy_analysis.py. "
        "Its `repair_gain` column now refers "
        "to PROPAGATED repair gain, so Test 3 "
        "directly asks whether confidence selects "
        "the highest-value propagated repair."
    )


if __name__ == "__main__":
    main()