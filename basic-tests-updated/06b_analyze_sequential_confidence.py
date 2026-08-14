#!/usr/bin/env python3
from __future__ import annotations

"""
06b_analyze_sequential_confidence.py

Analysis-only follow-up for 06_sequential_confidence.py.

Why this exists:
1) Recompute error-detection metrics on hops whose referenced parents are still
   benchmark-correct. This avoids treating downstream off-gold-path mismatches
   as independent local errors.
2) Use tie-aware within-trace top-1 metrics. This matters especially for
   verbalized confidence, which can be highly discretized/tied.
3) Report the verbal-confidence distribution and tie frequency.

This script DOES NOT load Qwen and DOES NOT regenerate anything.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import safe_auroc_auprc, write_json

SIGNALS = {
    "mean_logprob": "u_mean_logprob",
    "min_logprob": "u_min_logprob",
    "entropy": "u_entropy",
    "margin": "u_margin",
    "verbal": "u_verbal",
}

NORMALIZED = {
    "centered": "_centered",
    "z": "_z",
    "minmax": "_minmax",
}


def detection(df: pd.DataFrame, col: str) -> dict:
    scored = df[
        df["label"].isin(["correct", "incorrect"])
        & np.isfinite(pd.to_numeric(df[col], errors="coerce"))
    ].copy()
    y = (scored["label"] == "incorrect").astype(int).to_numpy()
    x = pd.to_numeric(scored[col], errors="coerce").to_numpy(dtype=float)
    out = safe_auroc_auprc(x, y)
    out["error_prevalence"] = float(y.mean()) if len(y) else None
    return out


def tie_aware_localization(df: pd.DataFrame, col: str) -> dict:
    """
    For tied maximum-uncertainty hops, score the expected hit under uniform
    tie-breaking rather than silently selecting the first row.
    """
    rows = []
    pair_hits = []
    tie_questions = 0

    for qid, g0 in df.groupby("question_id"):
        g = g0[
            g0["label"].isin(["correct", "incorrect"])
            & np.isfinite(pd.to_numeric(g0[col], errors="coerce"))
        ].copy()
        if len(g) == 0:
            continue

        wrong = g[g["label"] == "incorrect"]
        if len(wrong) == 0:
            continue

        u = pd.to_numeric(g[col], errors="coerce")
        max_u = float(u.max())
        selected = g[np.isclose(u, max_u, atol=1e-12)].copy()
        if len(selected) > 1:
            tie_questions += 1

        first_error_hop = int(wrong["hop"].min())

        rows.append({
            "question_id": qid,
            "n_hops_scored": int(len(g)),
            "n_wrong": int(len(wrong)),
            "n_tied_most_uncertain": int(len(selected)),
            "first_error_hit_expected": float((selected["hop"] == first_error_hop).mean()),
            "any_error_hit_expected": float((selected["label"] == "incorrect").mean()),
            "random_expected_first_error": float(1.0 / len(g)),
            "random_expected_any_error": float(len(wrong) / len(g)),
        })

        correct = g[g["label"] == "correct"]
        if len(correct):
            for _, wr in wrong.iterrows():
                for _, cr in correct.iterrows():
                    du = float(wr[col]) - float(cr[col])
                    if abs(du) <= 1e-12:
                        pair_hits.append(0.5)
                    else:
                        pair_hits.append(float(du > 0))

    qdf = pd.DataFrame(rows)
    if len(qdf) == 0:
        return {"questions_with_clear_error": 0}

    return {
        "questions_with_clear_error": int(len(qdf)),
        "questions_with_tied_top_uncertainty": int(tie_questions),
        "first_error_top1_expected": float(qdf["first_error_hit_expected"].mean()),
        "random_expected_first_error_top1": float(qdf["random_expected_first_error"].mean()),
        "first_error_minus_random": float(
            (qdf["first_error_hit_expected"] - qdf["random_expected_first_error"]).mean()
        ),
        "any_error_top1_expected": float(qdf["any_error_hit_expected"].mean()),
        "random_expected_any_error_top1": float(qdf["random_expected_any_error"].mean()),
        "any_error_minus_random": float(
            (qdf["any_error_hit_expected"] - qdf["random_expected_any_error"]).mean()
        ),
        "pairwise_wrong_vs_correct_accuracy": float(np.mean(pair_hits)) if pair_hits else None,
        "n_wrong_correct_pairs": int(len(pair_hits)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/qwen17b_sequential_seed42")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    hops = pd.read_csv(run_dir / "06_sequential_hops.csv")

    scored_all = hops[hops["label"].isin(["correct", "incorrect"])].copy()
    valid_parent = hops[hops["referenced_parents_benchmark_correct"] == True].copy()
    scored_valid = valid_parent[valid_parent["label"].isin(["correct", "incorrect"])].copy()

    pooled = {}
    localization = {}

    for name, raw_col in SIGNALS.items():
        pooled[name] = {
            "all_sequential_hops_raw": detection(hops, raw_col),
            "valid_parent_hops_raw": detection(valid_parent, raw_col),
        }
        for norm_name, suffix in NORMALIZED.items():
            col = raw_col + suffix
            if col in hops.columns:
                pooled[name][f"all_sequential_hops_{norm_name}"] = detection(hops, col)
                pooled[name][f"valid_parent_hops_{norm_name}"] = detection(valid_parent, col)

        localization[name] = tie_aware_localization(hops, raw_col)

    vc = pd.to_numeric(hops["verbal_confidence"], errors="coerce")
    vc_counts = vc.dropna().value_counts().sort_index()

    payload = {
        "interpretation": {
            "valid_parent_definition": (
                "A hop is valid for local benchmark error analysis when all decomposition "
                "parents referenced by that hop were benchmark-correct. Downstream hops whose "
                "parents were wrong are retained in the trace but should not automatically be "
                "treated as independent repairable errors."
            ),
            "tie_rule": (
                "When multiple hops tie for maximum uncertainty, top-1 accuracy is the expected "
                "accuracy under uniform tie-breaking rather than choosing the first hop."
            ),
        },
        "counts": {
            "n_hops_total": int(len(hops)),
            "n_hops_scored_all": int(len(scored_all)),
            "n_hops_valid_parent": int(len(valid_parent)),
            "n_hops_scored_valid_parent": int(len(scored_valid)),
            "valid_parent_hop_accuracy": (
                float((scored_valid["label"] == "correct").mean()) if len(scored_valid) else None
            ),
        },
        "pooled_error_detection": pooled,
        "tie_aware_within_trace_localization": localization,
        "verbalized_confidence": {
            "n_unique_values": int(vc.nunique(dropna=True)),
            "value_counts": {str(float(k)): int(v) for k, v in vc_counts.items()},
        },
    }

    write_json(run_dir / "06b_analysis_summary.json", payload)
    print("\n================ 06B ANALYSIS-ONLY ================")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved {run_dir / '06b_analysis_summary.json'}")


if __name__ == "__main__":
    main()
