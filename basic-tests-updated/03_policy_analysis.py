#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import write_json


SIGNALS = {
    "mean_logprob": ("mean_logprob", -1.0),  # lower confidence => more uncertain
    "min_logprob": ("min_logprob", -1.0),
    "entropy": ("entropy", +1.0),             # higher entropy => more uncertain
    "margin": ("margin", -1.0),
}


def analyze_signal(df: pd.DataFrame, col: str, sign: float):
    d = df[np.isfinite(df[col]) & np.isfinite(df["repair_gain"])].copy()
    d["uncertainty"] = sign * d[col]

    qrows = []
    pair_hits = []

    for qid, g in d.groupby("question_id"):
        if len(g) < 2:
            continue

        best_gain = g["repair_gain"].max()
        best_idx = set(g.index[np.isclose(g["repair_gain"], best_gain, atol=1e-9)])
        conf_idx = g["uncertainty"].idxmax()
        earliest_idx = g["repair_hop"].idxmin()
        latest_idx = g["repair_hop"].idxmax()

        spread = float(g["repair_gain"].max() - g["repair_gain"].min())
        regret = float(best_gain - g.loc[conf_idx, "repair_gain"])

        qrows.append({
            "question_id": qid,
            "n_candidates": int(len(g)),
            "confidence_hit": float(conf_idx in best_idx),
            "random_expected_hit": float(len(best_idx) / len(g)),
            "earliest_hit": float(earliest_idx in best_idx),
            "latest_hit": float(latest_idx in best_idx),
            "confidence_regret": regret,
            "normalized_regret": regret / spread if spread > 1e-12 else 0.0,
            "gain_spread": spread,
            "confidence_rescue": float(g.loc[conf_idx, "repaired_final_label"] == "correct"),
            "oracle_rescue": float((g["repaired_final_label"] == "correct").any()),
        })

        ids = list(g.index)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ia, ib = ids[a], ids[b]
                du = float(g.loc[ia, "uncertainty"] - g.loc[ib, "uncertainty"])
                dg = float(g.loc[ia, "repair_gain"] - g.loc[ib, "repair_gain"])
                if abs(du) > 1e-12 and abs(dg) > 1e-12:
                    pair_hits.append(float((du > 0) == (dg > 0)))

    qdf = pd.DataFrame(qrows)
    if len(qdf) == 0:
        return {"eligible_questions": 0}, qdf

    rho, p = spearmanr(d["uncertainty"], d["repair_gain"]) if len(d) >= 3 else (np.nan, np.nan)

    summary = {
        "eligible_questions": int(len(qdf)),
        "candidate_count_distribution": {
            str(int(k)): int(v)
            for k, v in qdf["n_candidates"].value_counts().sort_index().items()
        },
        "confidence_top1_hit": float(qdf["confidence_hit"].mean()),
        "random_expected_top1_hit": float(qdf["random_expected_hit"].mean()),
        "confidence_minus_random": float(
            (qdf["confidence_hit"] - qdf["random_expected_hit"]).mean()
        ),
        "earliest_top1_hit": float(qdf["earliest_hit"].mean()),
        "latest_top1_hit": float(qdf["latest_hit"].mean()),
        "pairwise_ranking_accuracy": (
            float(np.mean(pair_hits)) if pair_hits else None
        ),
        "pooled_spearman_uncertainty_vs_gain": (
            float(rho) if np.isfinite(rho) else None
        ),
        "pooled_spearman_p": float(p) if np.isfinite(p) else None,
        "mean_confidence_regret": float(qdf["confidence_regret"].mean()),
        "mean_normalized_regret": float(qdf["normalized_regret"].mean()),
        "mean_gain_spread": float(qdf["gain_spread"].mean()),
        "confidence_rescue_rate": float(qdf["confidence_rescue"].mean()),
        "oracle_one_repair_rescue_rate": float(qdf["oracle_rescue"].mean()),
    }
    return summary, qdf


def bootstrap(qdf: pd.DataFrame, n_boot: int, seed: int):
    if len(qdf) == 0:
        return {}

    rng = np.random.default_rng(seed)
    stats = {
        "confidence_top1_hit": [],
        "random_expected_top1_hit": [],
        "confidence_minus_random": [],
        "confidence_regret": [],
        "normalized_regret": [],
        "confidence_rescue_rate": [],
        "oracle_rescue_rate": [],
    }
    n = len(qdf)

    for _ in range(n_boot):
        b = qdf.iloc[rng.integers(0, n, size=n)]
        stats["confidence_top1_hit"].append(float(b["confidence_hit"].mean()))
        stats["random_expected_top1_hit"].append(float(b["random_expected_hit"].mean()))
        stats["confidence_minus_random"].append(
            float((b["confidence_hit"] - b["random_expected_hit"]).mean())
        )
        stats["confidence_regret"].append(float(b["confidence_regret"].mean()))
        stats["normalized_regret"].append(float(b["normalized_regret"].mean()))
        stats["confidence_rescue_rate"].append(float(b["confidence_rescue"].mean()))
        stats["oracle_rescue_rate"].append(float(b["oracle_rescue"].mean()))

    return {
        key: {
            "mean": float(np.mean(vals)),
            "ci95": [
                float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5)),
            ],
        }
        for key, vals in stats.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/qwen17b_seed42")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "02_repairs.csv")

    all_summaries = {}
    primary_qdf = None

    for name, (col, sign) in SIGNALS.items():
        summary, qdf = analyze_signal(df, col, sign)
        summary["bootstrap"] = bootstrap(qdf, args.bootstrap, args.seed)
        all_summaries[name] = summary

        qdf.to_csv(run_dir / f"03_question_policy_{name}.csv", index=False)
        if name == "mean_logprob":
            primary_qdf = qdf

    write_json(run_dir / "03_policy_summary.json", all_summaries)
    if primary_qdf is not None:
        primary_qdf.to_csv(run_dir / "03_question_policy.csv", index=False)

    print("\n================ 03 POLICY ANALYSIS ================")
    print(json.dumps(all_summaries, indent=2))

    primary = all_summaries["mean_logprob"]
    if primary.get("eligible_questions", 0) < 10:
        print(
            "\nWARNING: fewer than 10 eligible multi-error questions. "
            "Treat ranking numbers as debugging evidence, not a result."
        )


if __name__ == "__main__":
    main()
