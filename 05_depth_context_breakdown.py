#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import safe_auroc_auprc, write_json


def detection_by_group(hops, group_col):
    out = {}
    for key, g in hops.groupby(group_col):
        scored = g[g["label"].isin(["correct", "incorrect"])].copy()
        y = (scored["label"] == "incorrect").astype(int).to_numpy()
        out[str(key)] = {
            "n": int(len(scored)),
            "mean_logprob": safe_auroc_auprc(-scored["mean_logprob"].to_numpy(), y),
            "entropy": safe_auroc_auprc(scored["entropy"].to_numpy(), y),
        }
    return out


def repair_by_group(repairs, group_col):
    out = {}
    for key, g in repairs.groupby(group_col):
        if len(g) >= 3:
            rho, p = spearmanr(-g["mean_logprob"], g["repair_gain"])
        else:
            rho, p = np.nan, np.nan
        multi = g.groupby("question_id").filter(lambda x: len(x) >= 2)
        spreads = (
            multi.groupby("question_id")["repair_gain"].agg(lambda x: x.max() - x.min())
            if len(multi) else pd.Series(dtype=float)
        )
        out[str(key)] = {
            "n_repair_candidates": int(len(g)),
            "n_questions": int(g["question_id"].nunique()),
            "n_multi_error_questions": int(
                g.groupby("question_id").size().ge(2).sum()
            ),
            "mean_repair_gain": float(g["repair_gain"].mean()) if len(g) else None,
            "mean_gain_spread": float(spreads.mean()) if len(spreads) else None,
            "spearman_low_confidence_vs_gain": float(rho) if np.isfinite(rho) else None,
            "spearman_p": float(p) if np.isfinite(p) else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/qwen17b_seed42")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    hops = pd.read_csv(run_dir / "01_hops.csv")
    questions = pd.read_csv(run_dir / "01_questions.csv")
    repairs = pd.read_csv(run_dir / "02_repairs.csv")

    # Add question-level final prompt length to hops / repairs.
    qmeta = questions[["question_id", "final_prompt_tokens"]].copy()
    hops2 = hops.merge(qmeta, on="question_id", how="left")
    repairs2 = repairs.merge(qmeta, on="question_id", how="left")

    # Context-length quartiles are exploratory, not a controlled context manipulation.
    # qcut can collapse bins when values tie.
    try:
        questions["context_length_quartile"] = pd.qcut(
            questions["final_prompt_tokens"],
            q=4,
            labels=["Q1-short", "Q2", "Q3", "Q4-long"],
            duplicates="drop",
        ).astype(str)
        quartile_map = questions.set_index("question_id")["context_length_quartile"]
        hops2["context_length_quartile"] = hops2["question_id"].map(quartile_map)
        repairs2["context_length_quartile"] = repairs2["question_id"].map(quartile_map)
        context_detection = detection_by_group(hops2, "context_length_quartile")
        context_repair = repair_by_group(repairs2, "context_length_quartile")
    except Exception as e:
        context_detection = {"error": str(e)}
        context_repair = {"error": str(e)}

    summary = {
        "by_reasoning_depth": {
            "confidence_to_error": detection_by_group(hops2, "n_hops"),
            "repair": repair_by_group(repairs2, "n_hops"),
        },
        "by_observed_prompt_length_quartile": {
            "note": (
                "Exploratory only. This is NOT the later controlled distractor/context-length "
                "experiment because naturally longer questions can differ in many other ways."
            ),
            "confidence_to_error": context_detection,
            "repair": context_repair,
        },
    }

    write_json(run_dir / "05_depth_context_summary.json", summary)
    print("\n================ 05 DEPTH / CONTEXT ================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
