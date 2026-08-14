#!/usr/bin/env python3
from __future__ import annotations

"""
08_sequential_policy_analysis.py

Analyze whether raw token confidence, within-trace-normalized confidence, or
verbalized confidence selects the highest-value propagated repair in the
sequential experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import write_json


SIGNALS = {
    "mean_logprob_raw": "u_mean_logprob",
    "mean_logprob_within_z": "u_mean_logprob_z",
    "min_logprob_raw": "u_min_logprob",
    "min_logprob_within_z": "u_min_logprob_z",
    "entropy_raw": "u_entropy",
    "entropy_within_z": "u_entropy_z",
    "margin_raw": "u_margin",
    "margin_within_z": "u_margin_z",
    "verbal_raw": "u_verbal",
    "verbal_within_z": "u_verbal_z",
}


def analyze(df: pd.DataFrame, col: str) -> tuple[dict, pd.DataFrame]:
    d = df[np.isfinite(pd.to_numeric(df[col], errors="coerce")) & np.isfinite(df["repair_gain"])].copy()
    qrows = []
    pair_hits = []

    for qid, g in d.groupby("question_id"):
        if len(g) < 2:
            continue
        g = g.copy()
        uncertainty = pd.to_numeric(g[col], errors="coerce")
        best_gain = float(g["repair_gain"].max())
        best_idx = set(g.index[np.isclose(g["repair_gain"], best_gain, atol=1e-9)])

        max_u = float(uncertainty.max())
        conf_indices = list(g.index[np.isclose(uncertainty, max_u, atol=1e-12)])
        earliest_idx = g["repair_hop"].idxmin()
        latest_idx = g["repair_hop"].idxmax()

        spread = float(g["repair_gain"].max() - g["repair_gain"].min())

        # Tie-aware expected policy value under uniform tie-breaking.
        selected_gains = g.loc[conf_indices, "repair_gain"].astype(float)
        confidence_hit = float(np.mean([idx in best_idx for idx in conf_indices]))
        confidence_selected_gain = float(selected_gains.mean())
        regret = float(best_gain - confidence_selected_gain)
        confidence_rescue = float(
            (g.loc[conf_indices, "repaired_final_label"] == "correct").mean()
        )

        # First-error baseline: earliest wrong repair candidate in this set.
        first_error_idx = earliest_idx

        qrows.append({
            "question_id": qid,
            "n_candidates": int(len(g)),
            "n_tied_confidence_choices": int(len(conf_indices)),
            "confidence_hit": confidence_hit,
            "random_expected_hit": float(len(best_idx) / len(g)),
            "earliest_hit": float(earliest_idx in best_idx),
            "latest_hit": float(latest_idx in best_idx),
            "first_error_hit": float(first_error_idx in best_idx),
            "confidence_regret": regret,
            "normalized_regret": regret / spread if spread > 1e-12 else 0.0,
            "gain_spread": spread,
            "confidence_repair_gain": confidence_selected_gain,
            "oracle_repair_gain": best_gain,
            "confidence_rescue": confidence_rescue,
            "oracle_rescue": float((g["repaired_final_label"] == "correct").any()),
        })

        ids = list(g.index)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ia, ib = ids[a], ids[b]
                du = float(g.loc[ia, col] - g.loc[ib, col])
                dg = float(g.loc[ia, "repair_gain"] - g.loc[ib, "repair_gain"])
                if abs(dg) <= 1e-12:
                    continue
                if abs(du) <= 1e-12:
                    pair_hits.append(0.5)
                else:
                    pair_hits.append(float((du > 0) == (dg > 0)))

    qdf = pd.DataFrame(qrows)
    if len(qdf) == 0:
        return {"eligible_questions": 0}, qdf

    rho, p = spearmanr(
        pd.to_numeric(d[col], errors="coerce"),
        pd.to_numeric(d["repair_gain"], errors="coerce"),
    ) if len(d) >= 3 else (np.nan, np.nan)

    return {
        "eligible_questions": int(len(qdf)),
        "candidate_count_distribution": {
            str(int(k)): int(v) for k, v in qdf["n_candidates"].value_counts().sort_index().items()
        },
        "confidence_top1_hit": float(qdf["confidence_hit"].mean()),
        "random_expected_top1_hit": float(qdf["random_expected_hit"].mean()),
        "confidence_minus_random": float((qdf["confidence_hit"] - qdf["random_expected_hit"]).mean()),
        "earliest_top1_hit": float(qdf["earliest_hit"].mean()),
        "latest_top1_hit": float(qdf["latest_hit"].mean()),
        "pairwise_ranking_accuracy": float(np.mean(pair_hits)) if pair_hits else None,
        "pooled_spearman_uncertainty_vs_gain": float(rho) if np.isfinite(rho) else None,
        "pooled_spearman_p": float(p) if np.isfinite(p) else None,
        "mean_confidence_regret": float(qdf["confidence_regret"].mean()),
        "mean_normalized_regret": float(qdf["normalized_regret"].mean()),
        "mean_gain_spread": float(qdf["gain_spread"].mean()),
        "mean_confidence_selected_gain": float(qdf["confidence_repair_gain"].mean()),
        "mean_oracle_gain": float(qdf["oracle_repair_gain"].mean()),
        "confidence_rescue_rate": float(qdf["confidence_rescue"].mean()),
        "oracle_one_repair_rescue_rate": float(qdf["oracle_rescue"].mean()),
    }, qdf


def bootstrap(qdf: pd.DataFrame, n_boot: int, seed: int) -> dict:
    if len(qdf) == 0:
        return {}
    rng = np.random.default_rng(seed)
    stats = {"confidence_top1_hit": [], "confidence_minus_random": [], "normalized_regret": []}
    n = len(qdf)
    for _ in range(n_boot):
        b = qdf.iloc[rng.integers(0, n, size=n)]
        stats["confidence_top1_hit"].append(float(b["confidence_hit"].mean()))
        stats["confidence_minus_random"].append(float((b["confidence_hit"] - b["random_expected_hit"]).mean()))
        stats["normalized_regret"].append(float(b["normalized_regret"].mean()))
    return {
        k: {
            "mean": float(np.mean(v)),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
        }
        for k, v in stats.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/qwen17b_sequential_seed42")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "07_repairs.csv")

    all_summaries = {}
    for name, col in SIGNALS.items():
        if col not in df.columns:
            all_summaries[name] = {"error": f"missing column {col}"}
            continue
        summary, qdf = analyze(df, col)
        summary["bootstrap"] = bootstrap(qdf, args.bootstrap, args.seed)
        all_summaries[name] = summary
        qdf.to_csv(run_dir / f"08_policy_{name}.csv", index=False)

    all_summaries["interpretation_note"] = (
        "Within-trace z-scoring is monotone within each question, so raw vs z-scored "
        "top-1 repair ranking should usually be identical when both are finite. Its useful "
        "role is primarily in pooled cross-question detection/calibration analyses, not changing "
        "the within-question argmax."
    )

    write_json(run_dir / "08_policy_summary.json", all_summaries)
    print("\n================ 08 SEQUENTIAL POLICY ================")
    print(json.dumps(all_summaries, indent=2))

    # Tiny-sample warning.
    candidates = [x.get("eligible_questions", 0) for x in all_summaries.values() if isinstance(x, dict)]
    max_n = max(candidates) if candidates else 0
    if max_n < 20:
        print("\nWARNING: fewer than 20 eligible multi-error questions. Treat ranking estimates as pilot evidence only.")


if __name__ == "__main__":
    main()
