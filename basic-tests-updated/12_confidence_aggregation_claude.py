#!/usr/bin/env python3
from __future__ import annotations

"""
12_confidence_aggregation_claude.py

ANALYSIS ONLY. Does not load Qwen and does not regenerate any model output.
Reads existing Test 06 artifacts and writes into its own --run-dir.

QUESTION (mentor item 2)
------------------------
Which per-step confidence score, and which method of aggregating those scores
across a reasoning trace, actually helps predict the final result?

TARGETS
-------
  final_correct   binary, baseline_final_label == "correct"   -> AUROC / AUPRC
  final_gold_lp   baseline_final_gold_logprob                 -> Spearman / Pearson

`final_gold_lp` is the quantity the repo's `repair_gain` metric is a difference
of (README_confidence_repair.md):
    repair_gain = mean logP(gold final | repaired) - mean logP(gold final | original)
so correlating a trace-level confidence score against it asks whether that score
tracks the same axis repair_gain is measured on.

Questions whose final label is `needs_review` are excluded from the automatic
metrics, following the repo convention. They are never silently dropped: the
count is reported.

ORIENTATION
-----------
Every signal is converted so that HIGHER = MORE CONFIDENT:
    mean_logprob, min_logprob, margin   as-is
    entropy                             negated
    verbal_confidence                   as-is (raw 0-100)

VARIANTS
--------
    raw       the signal as recorded
    z         within-trace z-score
    minmax    within-trace min-max
The repo already documents that within-trace normalization LOSES to raw for
pooled hop-level error detection (mean-logprob 0.735 raw vs 0.596 z on
valid-parent hops). z/minmax are included here to test whether that also holds
for this different, trace-level target.

AGGREGATIONS
------------
On the native confidence scale:
    mean, min, max, median, first, last, range, std,
    weight_late, weight_early, token_weighted,
    min_terminal_ancestors  (min over hops the terminal hop transitively
                             depends on, via the `#n` reference graph)
On a [0,1]-mapped scale:
    product, noisy_or, harmonic

The [0,1] map is a GLOBAL min-max across all hops of the dataset. It exists so
that product / noisy-or / harmonic are defined. It is a monotone rescaling for
aggregation only. It is NOT calibration, these are NOT probabilities, and no
aggregate here should be described as a calibrated metric.

THE MULTIPLE-COMPARISONS PROBLEM (the point of this script)
------------------------------------------------------------
The grid is (signals x variants x aggregations) cells scored on ~56 questions.
Picking the best cell from a grid that large on a sample that small will find a
strong-looking winner from pure noise. Per-cell bootstrap CIs do NOT fix this,
because they say nothing about the selection.

So the script runs a BEST-OF-GRID PERMUTATION NULL: shuffle the outcome across
questions, recompute EVERY cell, record the best cell's score, repeat B times.
That yields the distribution of "best cell you would see by chance" and a
p-value for the observed winner against it. If the observed best does not clear
that bar, the correct conclusion is that no signal/aggregation is
distinguishable at this sample size.

Because label permutation does not change the ranks of the score columns, the
ranks are precomputed once and every permutation is a matrix product, which is
what makes a full-grid null cheap.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr
from sklearn.metrics import average_precision_score

REF_RE = re.compile(r"#(\d+)")

# signal -> (column, sign mapping raw column to confidence orientation)
SIGNALS = {
    "mean_logprob": ("mean_logprob", +1.0),
    "min_logprob": ("min_logprob", +1.0),
    "entropy": ("entropy", -1.0),
    "margin": ("margin", +1.0),
    "verbal": ("verbal_confidence", +1.0),
}

VARIANTS = ["raw", "z", "minmax"]

NATIVE_AGGS = [
    "mean", "min", "max", "median", "first", "last", "range", "std",
    "weight_late", "weight_early", "token_weighted", "min_terminal_ancestors",
]
PROB_AGGS = ["product", "noisy_or", "harmonic"]
AGGS = NATIVE_AGGS + PROB_AGGS


# ---------------------------------------------------------------- dependencies

def terminal_ancestors(templates: list[str]) -> list[int]:
    """Indices the terminal hop transitively depends on, plus the terminal hop."""
    n = len(templates)
    parents = []
    for j, t in enumerate(templates):
        refs = {int(m.group(1)) - 1 for m in REF_RE.finditer(str(t))}
        parents.append({p for p in refs if 0 <= p < j})
    keep = {n - 1}
    frontier = [n - 1]
    while frontier:
        cur = frontier.pop()
        for p in parents[cur]:
            if p not in keep:
                keep.add(p)
                frontier.append(p)
    return sorted(keep)


# ---------------------------------------------------------------- aggregation

def aggregate(vals: np.ndarray, agg: str, weights: np.ndarray | None,
              ancestors: list[int]) -> float:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)] if agg not in ("first", "last", "min_terminal_ancestors") else v
    if len(v) == 0:
        return np.nan
    n = len(v)

    if agg == "mean":
        return float(np.mean(v))
    if agg == "min":
        return float(np.min(v))
    if agg == "max":
        return float(np.max(v))
    if agg == "median":
        return float(np.median(v))
    if agg == "first":
        return float(v[0])
    if agg == "last":
        return float(v[-1])
    if agg == "range":
        return float(np.max(v) - np.min(v))
    if agg == "std":
        return float(np.std(v, ddof=0))
    if agg == "weight_late":
        w = np.arange(1, n + 1, dtype=float)
        return float(np.sum(v * w) / np.sum(w))
    if agg == "weight_early":
        w = np.arange(n, 0, -1, dtype=float)
        return float(np.sum(v * w) / np.sum(w))
    if agg == "token_weighted":
        if weights is None or not np.isfinite(weights).all() or np.sum(weights) <= 0:
            return float(np.mean(v))
        return float(np.sum(v * weights) / np.sum(weights))
    if agg == "min_terminal_ancestors":
        sel = [v[i] for i in ancestors if 0 <= i < len(v) and np.isfinite(v[i])]
        return float(np.min(sel)) if sel else np.nan

    # [0,1]-space aggregations; v is already globally min-max mapped by caller
    p = np.clip(v, 1e-6, 1 - 1e-6)
    if agg == "product":
        return float(np.sum(np.log(p)))          # log of the product; monotone in it
    if agg == "noisy_or":
        return float(1.0 - np.prod(1.0 - p))
    if agg == "harmonic":
        return float(n / np.sum(1.0 / p))
    raise ValueError(agg)


# ---------------------------------------------------------------- fast metrics

def auroc_from_ranks(ranks: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    ranks: (n_questions, n_cells) column-wise ranks of the score, computed once.
    y: (n_questions,) binary. Returns AUROC per cell.
    Label permutation does not change ranks, so this is the whole permutation
    null in one matrix product.
    """
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    if npos == 0 or nneg == 0:
        return np.full(ranks.shape[1], np.nan)
    s = y.astype(float) @ ranks
    return (s - npos * (npos + 1.0) / 2.0) / (npos * nneg)


def spearman_from_ranks(ranks: np.ndarray, yrank: np.ndarray) -> np.ndarray:
    a = ranks - ranks.mean(axis=0, keepdims=True)
    b = yrank - yrank.mean()
    denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b[:, None]).sum(axis=0) / denom


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="outputs/qwen17b_sequential_seed42")
    ap.add_argument("--run-dir", default="outputs/qwen17b_aggregation_claude")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = Path(args.source_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    hops = pd.read_csv(src / "06_sequential_hops.csv")
    questions = pd.read_csv(src / "06_sequential_questions.csv")

    # ---- build per-question aggregated scores ---------------------------
    # Global [0,1] map per (signal, variant), for the probability-space aggs only.
    oriented = {}
    for sig, (col, sign) in SIGNALS.items():
        base = sign * pd.to_numeric(hops[col], errors="coerce")
        oriented[(sig, "raw")] = base
        z = pd.Series(np.nan, index=hops.index)
        mm = pd.Series(np.nan, index=hops.index)
        for _, idx in hops.groupby("question_id").groups.items():
            idx = list(idx)
            v = base.loc[idx].astype(float)
            f = v[np.isfinite(v)]
            if len(f) == 0:
                continue
            sd = float(f.std(ddof=0))
            lo, hi = float(f.min()), float(f.max())
            z.loc[idx] = (v - float(f.mean())) / sd if sd > 1e-12 else 0.0
            mm.loc[idx] = (v - lo) / (hi - lo) if hi - lo > 1e-12 else 0.5
        oriented[(sig, "z")] = z
        oriented[(sig, "minmax")] = mm

    gmap = {}
    for k, s in oriented.items():
        f = s[np.isfinite(s)]
        lo, hi = (float(f.min()), float(f.max())) if len(f) else (0.0, 1.0)
        gmap[k] = (lo, hi if hi - lo > 1e-12 else lo + 1.0)

    qids = list(questions["question_id"])
    rows = []
    for qid in qids:
        g = hops[hops["question_id"] == qid].sort_values("hop_idx")
        idx = list(g.index)
        anc = terminal_ancestors(list(g["template_question"]))
        w = pd.to_numeric(g["generated_tokens"], errors="coerce").to_numpy(dtype=float)
        rec = {"question_id": qid, "n_hops": int(len(g))}
        for (sig, var), s in oriented.items():
            v = s.loc[idx].to_numpy(dtype=float)
            lo, hi = gmap[(sig, var)]
            p = (v - lo) / (hi - lo)
            for agg in AGGS:
                use = p if agg in PROB_AGGS else v
                rec[f"{sig}|{var}|{agg}"] = aggregate(use, agg, w, anc)
        rows.append(rec)
    trace = pd.DataFrame(rows).merge(
        questions[["question_id", "n_hops", "baseline_final_label",
                   "baseline_final_gold_logprob", "n_clear_wrong_hops"]],
        on=["question_id", "n_hops"], how="left",
    )
    trace.to_csv(run_dir / "12_trace_scores.csv", index=False)

    cells = [c for c in trace.columns if "|" in c]

    # ---- targets --------------------------------------------------------
    n_review = int((trace["baseline_final_label"] == "needs_review").sum())
    scored = trace[trace["baseline_final_label"].isin(["correct", "incorrect"])].copy()
    y_bin = (scored["baseline_final_label"] == "correct").astype(int).to_numpy()
    y_lp = pd.to_numeric(scored["baseline_final_gold_logprob"], errors="coerce").to_numpy()

    X = scored[cells].to_numpy(dtype=float)
    # Drop non-finite cells and cells that are CONSTANT BY CONSTRUCTION. Within-trace
    # min-max forces min=0 and max=1 in every trace, so minmax|min, minmax|max and
    # minmax|range carry no information at all and would otherwise be scored as if
    # they were real cells.
    finite_ok = np.isfinite(X).all(axis=0)
    varies = np.nanmax(X, axis=0) - np.nanmin(X, axis=0) > 1e-12
    keep = finite_ok & varies
    dropped_nonfinite = [cells[i] for i in np.where(~finite_ok)[0]]
    dropped_constant = [cells[i] for i in np.where(finite_ok & ~varies)[0]]
    dropped = dropped_nonfinite + dropped_constant
    cells = [c for c, k in zip(cells, keep) if k]
    X = X[:, keep]

    R = np.apply_along_axis(rankdata, 0, X)
    obs_auroc = auroc_from_ranks(R, y_bin)
    yr = rankdata(y_lp)
    obs_spear = spearman_from_ranks(R, yr)

    auprc = np.array([average_precision_score(y_bin, X[:, j]) for j in range(X.shape[1])])
    pear = np.array([pearsonr(X[:, j], y_lp)[0] for j in range(X.shape[1])])

    # ---- per-cell bootstrap over questions -------------------------------
    n = len(y_bin)
    boot_au = np.empty((args.bootstrap, len(cells)))
    boot_sp = np.empty((args.bootstrap, len(cells)))
    for b in range(args.bootstrap):
        s = rng.integers(0, n, n)
        Rb = np.apply_along_axis(rankdata, 0, X[s])
        boot_au[b] = auroc_from_ranks(Rb, y_bin[s])
        boot_sp[b] = spearman_from_ranks(Rb, rankdata(y_lp[s]))
    au_lo, au_hi = np.nanpercentile(boot_au, [2.5, 97.5], axis=0)
    sp_lo, sp_hi = np.nanpercentile(boot_sp, [2.5, 97.5], axis=0)

    # ---- best-of-grid permutation null -----------------------------------
    perm_best_au = np.empty(args.permutations)
    perm_best_sp = np.empty(args.permutations)
    for b in range(args.permutations):
        yp = y_bin[rng.permutation(n)]
        perm_best_au[b] = np.nanmax(np.abs(auroc_from_ranks(R, yp) - 0.5))
        ypl = yr[rng.permutation(n)]
        perm_best_sp[b] = np.nanmax(np.abs(spearman_from_ranks(R, ypl)))

    obs_best_au = float(np.nanmax(np.abs(obs_auroc - 0.5)))
    obs_best_sp = float(np.nanmax(np.abs(obs_spear)))
    p_au = float((np.sum(perm_best_au >= obs_best_au) + 1) / (args.permutations + 1))
    p_sp = float((np.sum(perm_best_sp >= obs_best_sp) + 1) / (args.permutations + 1))

    # Is the winner just tracking question difficulty rather than anything
    # confidence-specific? Report its relation to the number of clearly wrong hops
    # and its stability across the 3-hop / 4-hop subgroups.
    best_j = int(np.nanargmax(np.abs(obs_auroc - 0.5)))
    best_cell = cells[best_j]
    bx = X[:, best_j]
    nwrong = pd.to_numeric(scored["n_clear_wrong_hops"], errors="coerce").to_numpy(float)
    nh = pd.to_numeric(scored["n_hops"], errors="coerce").to_numpy(float)
    winner_diag = {
        "cell": best_cell,
        "auroc_all": float(obs_auroc[best_j]),
        "spearman_vs_n_clear_wrong_hops": float(spearmanr(bx, nwrong).statistic),
        "note_difficulty": (
            "A strong negative correlation with n_clear_wrong_hops means the winning "
            "aggregate is largely reporting how many steps went wrong, i.e. question "
            "difficulty, rather than anything specific to confidence semantics."
        ),
        "by_n_hops": {},
    }
    for k in sorted(set(nh[np.isfinite(nh)])):
        m = nh == k
        if m.sum() >= 5 and len(set(y_bin[m])) == 2:
            winner_diag["by_n_hops"][f"{int(k)}hop"] = {
                "n": int(m.sum()),
                "n_correct": int(y_bin[m].sum()),
                "auroc": float(auroc_from_ranks(
                    np.apply_along_axis(rankdata, 0, X[m][:, [best_j]]), y_bin[m])[0]),
            }

    grid = pd.DataFrame({
        "cell": cells,
        "signal": [c.split("|")[0] for c in cells],
        "variant": [c.split("|")[1] for c in cells],
        "aggregation": [c.split("|")[2] for c in cells],
        "auroc_final_correct": obs_auroc,
        "auroc_ci_lo": au_lo, "auroc_ci_hi": au_hi,
        "auprc_final_correct": auprc,
        "spearman_final_gold_lp": obs_spear,
        "spearman_ci_lo": sp_lo, "spearman_ci_hi": sp_hi,
        "pearson_final_gold_lp": pear,
        "abs_auroc_minus_half": np.abs(obs_auroc - 0.5),
    }).sort_values("abs_auroc_minus_half", ascending=False).reset_index(drop=True)
    grid.to_csv(run_dir / "12_aggregation_grid.csv", index=False)

    top_au = grid.head(10)
    top_sp = grid.reindex(grid["spearman_final_gold_lp"].abs()
                          .sort_values(ascending=False).index).head(10)

    summary = {
        "script": "12_confidence_aggregation_claude.py",
        "analysis_only": True,
        "source_dir": str(src),
        "n_questions_total": int(len(trace)),
        "n_questions_scored": int(len(scored)),
        "n_questions_needs_review_excluded": n_review,
        "n_grid_cells": len(cells),
        "n_cells_dropped_nonfinite": len(dropped_nonfinite),
        "n_cells_dropped_constant_by_construction": len(dropped_constant),
        "cells_dropped_constant_by_construction": dropped_constant,
        "cells_dropped_nonfinite": dropped_nonfinite,
        "grid_shape": {
            "signals": list(SIGNALS), "variants": VARIANTS, "aggregations": AGGS,
        },
        "target_definitions": {
            "final_correct": "baseline_final_label == 'correct' (AUROC/AUPRC)",
            "final_gold_lp": (
                "baseline_final_gold_logprob; the quantity repo repair_gain is a "
                "difference of (Spearman/Pearson)"
            ),
        },
        "best_of_grid_permutation_null": {
            "permutations": args.permutations,
            "observed_best_abs_auroc_minus_half": obs_best_au,
            "observed_best_auroc_cell": str(grid.iloc[0]["cell"]),
            "observed_best_auroc": float(grid.iloc[0]["auroc_final_correct"]),
            "null_mean_best_abs_auroc_minus_half": float(perm_best_au.mean()),
            "null_p95_best_abs_auroc_minus_half": float(np.percentile(perm_best_au, 95)),
            "p_value_auroc": p_au,
            "observed_best_abs_spearman": obs_best_sp,
            "observed_best_spearman_cell": str(top_sp.iloc[0]["cell"]),
            "observed_best_spearman": float(top_sp.iloc[0]["spearman_final_gold_lp"]),
            "null_mean_best_abs_spearman": float(perm_best_sp.mean()),
            "null_p95_best_abs_spearman": float(np.percentile(perm_best_sp, 95)),
            "p_value_spearman": p_sp,
            "interpretation": (
                "p is the probability that a grid this size produces a winner at least "
                "this strong when the outcome is unrelated to every score. p > 0.05 means "
                "no signal/aggregation is distinguishable at this sample size."
            ),
        },
        "top10_by_auroc": top_au.drop(columns=["abs_auroc_minus_half"]).to_dict(orient="records"),
        "top10_by_abs_spearman": top_sp.drop(columns=["abs_auroc_minus_half"]).to_dict(orient="records"),
        "raw_vs_normalized_variant_means": {
            v: {
                "mean_auroc": float(grid[grid["variant"] == v]["auroc_final_correct"].mean()),
                "mean_abs_spearman": float(
                    grid[grid["variant"] == v]["spearman_final_gold_lp"].abs().mean()),
            } for v in VARIANTS
        },
        "per_signal_best": {
            s: {
                "best_cell": str(g.sort_values("abs_auroc_minus_half", ascending=False)
                                 .iloc[0]["cell"]),
                "best_auroc": float(g.sort_values("abs_auroc_minus_half", ascending=False)
                                    .iloc[0]["auroc_final_correct"]),
            } for s, g in grid.groupby("signal")
        },
        "winner_diagnostics": winner_diag,
        "caveats": [
            "Analysis-only; no model was run and no existing output was modified.",
            "N is 56 scored questions against a grid this size; per-cell CIs do not "
            "correct for selecting the winner. Read the permutation null first.",
            "The [0,1] map used by product/noisy_or/harmonic is a monotone rescale for "
            "aggregation only. It is not calibration and these are not probabilities.",
            "No aggregate here is a calibrated metric.",
            "This answers the trace-level aggregation question. Per-candidate repair "
            "selection is 08_sequential_policy_analysis_v2.py.",
        ],
    }
    Path(run_dir / "12_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ 12 CONFIDENCE AGGREGATION ================")
    print(f"scored questions: {len(scored)}  (needs_review excluded: {n_review})")
    print(f"grid cells: {len(cells)}\n")
    print("--- top 10 by |AUROC - 0.5| for final correctness ---")
    print(top_au[["cell", "auroc_final_correct", "auroc_ci_lo", "auroc_ci_hi",
                  "auprc_final_correct"]].to_string(index=False))
    print("\n--- top 10 by |Spearman| vs final gold logprob ---")
    print(top_sp[["cell", "spearman_final_gold_lp", "spearman_ci_lo",
                  "spearman_ci_hi"]].to_string(index=False))
    print("\n--- best-of-grid permutation null ---")
    print(json.dumps(summary["best_of_grid_permutation_null"], indent=2))
    print("\n--- mean performance by variant ---")
    print(json.dumps(summary["raw_vs_normalized_variant_means"], indent=2))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
