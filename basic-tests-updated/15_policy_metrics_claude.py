#!/usr/bin/env python3
from __future__ import annotations

"""
15_policy_metrics_claude.py

TODO 2, closing the gap: run the ACTUAL repair-policy metric suite defined in
README_confidence_repair.md, rather than an approximation of it.

Metrics implemented exactly as the docs define them:
    repair_gain            = logP(gold final | repaired) - logP(gold final | original)
    confidence top-1 hit   selected candidate is an oracle best-gain candidate,
                           using EXPECTED value under uniform tie-breaking
    random expected top-1  K / N   (K tied best-gain candidates of N)
    pairwise accuracy      ties count 0.5
    repair regret          best_gain - selected_gain
    normalized regret      regret / (best_gain - worst_gain)
    mean selected gain vs mean oracle gain
    rescue rate
    bootstrap CIs
Plus the doc's structural baselines: earliest-wrong and latest-wrong.

TWO LEVELS
----------
A. PER-STEP (replicates and extends 08): among repair candidates inside a
   question, which per-step confidence signal picks the highest-gain repair?

B. TRACE-LEVEL AGGREGATION (the new part Xiang asked for): does an aggregate of
   per-step confidence predict question-level repair outcomes -- how much repair
   value is available, and whether the question is rescuable? Aggregation is a
   trace-level operation, so it cannot select WITHIN a question; this is the
   coherent way to apply it to the repair metric.

CLUSTERING WARNING
------------------
4 of the 8 multi-candidate questions share hops 2-4 (the `88460_30152_20999`
chain). They are MuSiQue variants of one reasoning chain, not independent
samples. Every per-step result is reported both raw and after collapsing that
cluster to one item.

Analysis only. No model. Writes only into --run-dir.
"""

import argparse, json, re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SIGNALS = {   # sign maps raw column -> UNCERTAINTY (higher = less confident)
    "mean_logprob": -1.0, "min_logprob": -1.0, "entropy": +1.0,
    "margin": -1.0, "verbal_confidence": -1.0,
}
AGGS = ["mean", "min", "max", "median", "first", "last", "std", "range"]


def chain_key(qid: str) -> str:
    """MuSiQue ids look like 4hop3__A_B_C_D; the shared tail identifies a chain."""
    m = re.match(r"(\d)hop\d*__(.+)", str(qid))
    return "_".join(m.group(2).split("_")[1:]) if m else str(qid)


def expected_top1(u: np.ndarray, gain: np.ndarray) -> float:
    """Expected top-1 hit under uniform tie-breaking among max-uncertainty candidates."""
    sel = np.flatnonzero(u >= u.max() - 1e-12)
    best = np.flatnonzero(gain >= gain.max() - 1e-12)
    return float(len(set(sel) & set(best)) / len(sel))


def expected_selected_gain(u: np.ndarray, gain: np.ndarray) -> float:
    sel = np.flatnonzero(u >= u.max() - 1e-12)
    return float(np.mean(gain[sel]))


def policy_for(df: pd.DataFrame, ucol: str) -> dict:
    hits, rand, pair, regret, nregret, selg, oracle, rescue = [], [], [], [], [], [], [], []
    early, late = [], []
    for _, g in df.groupby("question_id"):
        if len(g) < 2:
            continue
        u = pd.to_numeric(g[ucol], errors="coerce").to_numpy(float)
        gain = g["repair_gain"].to_numpy(float)
        if not np.isfinite(u).all():
            continue
        hop = g["repair_hop"].to_numpy(float)
        best = float(gain.max()); worst = float(gain.min())
        K = int((gain >= best - 1e-12).sum())
        hits.append(expected_top1(u, gain))
        rand.append(K / len(gain))
        sg = expected_selected_gain(u, gain)
        selg.append(sg); oracle.append(best)
        regret.append(best - sg)
        nregret.append((best - sg) / (best - worst) if best - worst > 1e-12 else 0.0)
        early.append(float(gain[np.argmin(hop)] >= best - 1e-12))
        late.append(float(gain[np.argmax(hop)] >= best - 1e-12))
        lab = g["repaired_final_label"].to_numpy()
        sel = np.flatnonzero(u >= u.max() - 1e-12)
        rescue.append(float(np.mean([lab[i] == "correct" for i in sel])))
        for i, j in combinations(range(len(gain)), 2):
            if abs(gain[i] - gain[j]) < 1e-12:
                continue
            if abs(u[i] - u[j]) < 1e-12:
                pair.append(0.5)
            else:
                pair.append(float((u[i] > u[j]) == (gain[i] > gain[j])))
    n = len(hits)
    if n == 0:
        return {"n_questions": 0}
    rng = np.random.default_rng(0)
    h = np.array(hits); r = np.array(rand)
    boot = [float((h[s] - r[s]).mean())
            for s in (rng.integers(0, n, n) for _ in range(5000))]
    return {
        "n_questions": n,
        "confidence_top1": float(h.mean()),
        "random_expected_top1": float(r.mean()),
        "confidence_minus_random": float((h - r).mean()),
        "confidence_minus_random_ci95": [float(np.percentile(boot, 2.5)),
                                         float(np.percentile(boot, 97.5))],
        "earliest_wrong_top1": float(np.mean(early)),
        "latest_wrong_top1": float(np.mean(late)),
        "pairwise_accuracy": float(np.mean(pair)) if pair else None,
        "n_pairs": len(pair),
        "mean_regret": float(np.mean(regret)),
        "mean_normalized_regret": float(np.mean(nregret)),
        "mean_selected_gain": float(np.mean(selg)),
        "mean_oracle_gain": float(np.mean(oracle)),
        "selected_rescue_rate": float(np.mean(rescue)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="outputs/qwen17b_sequential_seed42")
    ap.add_argument("--run-dir", default="outputs/qwen17b_policy_claude")
    args = ap.parse_args()

    src, run_dir = Path(args.source_dir), Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rep = pd.read_csv(src / "07_repairs.csv")
    hops = pd.read_csv(src / "06_sequential_hops.csv")

    rep["chain"] = rep.question_id.map(chain_key)
    for s, sign in SIGNALS.items():
        rep[f"u_{s}"] = sign * pd.to_numeric(rep[s], errors="coerce")

    out = {"source": str(src),
           "n_candidates": int(len(rep)),
           "n_questions": int(rep.question_id.nunique())}

    # ---------- A. per-step repair selection ----------
    multi = rep.groupby("question_id").filter(lambda g: len(g) >= 2)
    out["multi_candidate"] = {
        "n_questions": int(multi.question_id.nunique()),
        "n_distinct_chains": int(multi.chain.nunique()),
        "chain_sizes": {k: int(v) for k, v in
                        multi.groupby("chain").question_id.nunique().items()},
        "clustering_warning": (
            "Questions sharing a chain key share sub-questions and are not "
            "independent samples."),
    }

    # one representative question per chain, deterministically chosen
    dedup_ids = (multi.sort_values("question_id").groupby("chain")
                 .question_id.first().tolist())
    dedup = multi[multi.question_id.isin(dedup_ids)]

    out["per_step_policy"] = {}
    out["per_step_policy_chain_deduplicated"] = {}
    for s in SIGNALS:
        out["per_step_policy"][s] = policy_for(multi, f"u_{s}")
        out["per_step_policy_chain_deduplicated"][s] = policy_for(dedup, f"u_{s}")

    # ---------- B. trace-level aggregation vs question-level repair ----------
    qagg = []
    for qid, g in hops.groupby("question_id"):
        g = g.sort_values("hop_idx")
        row = {"question_id": qid}
        for s, sign in SIGNALS.items():
            v = pd.to_numeric(g[s], errors="coerce").astype(float)
            conf = -sign * v          # back to confidence orientation
            f = conf[np.isfinite(conf)]
            if not len(f):
                continue
            row.update({
                f"{s}|mean": f.mean(), f"{s}|min": f.min(), f"{s}|max": f.max(),
                f"{s}|median": f.median(), f"{s}|first": float(conf.iloc[0]),
                f"{s}|last": float(conf.iloc[-1]), f"{s}|std": f.std(ddof=0),
                f"{s}|range": f.max() - f.min(),
            })
        qagg.append(row)
    qagg = pd.DataFrame(qagg)

    qout = rep.groupby("question_id").agg(
        max_repair_gain=("repair_gain", "max"),
        mean_repair_gain=("repair_gain", "mean"),
        n_candidates=("repair_gain", "size"),
        any_rescue=("repaired_final_label", lambda s: float((s == "correct").any())),
    ).reset_index()
    merged = qagg.merge(qout, on="question_id", how="inner")
    merged.to_csv(run_dir / "15_question_level.csv", index=False)

    cells = [c for c in merged.columns if "|" in c]
    rows = []
    for c in cells:
        x = pd.to_numeric(merged[c], errors="coerce")
        m = np.isfinite(x)
        if m.sum() < 10:
            continue
        for tgt in ["max_repair_gain", "mean_repair_gain"]:
            r = spearmanr(x[m], merged[tgt][m])
            rows.append({"cell": c, "target": tgt, "n": int(m.sum()),
                         "spearman": float(r.statistic), "p": float(r.pvalue)})
    agg_df = pd.DataFrame(rows).sort_values("p")
    agg_df.to_csv(run_dir / "15_aggregation_vs_repair.csv", index=False)
    out["trace_aggregation_vs_repair_value"] = {
        "n_questions": int(len(merged)),
        "n_cells": int(agg_df.cell.nunique()),
        "top10_by_p": agg_df.head(10).to_dict("records"),
        "n_cells_p_below_05": int((agg_df.p < 0.05).sum()),
        "n_cells_expected_below_05_by_chance": float(0.05 * len(agg_df)),
        "note": ("With this many cells, compare the count of p<0.05 against the "
                 "chance expectation before reading any single row."),
    }

    (run_dir / "15_policy_summary.json").write_text(json.dumps(out, indent=2))

    print("\n============ REPAIR-POLICY METRICS (repo definitions) ============")
    print(f"candidates={out['n_candidates']}  questions={out['n_questions']}  "
          f"multi-candidate={out['multi_candidate']['n_questions']} "
          f"across {out['multi_candidate']['n_distinct_chains']} distinct chains")
    for name, key in [("ALL 8 multi-candidate questions", "per_step_policy"),
                      ("CHAIN-DEDUPLICATED", "per_step_policy_chain_deduplicated")]:
        print(f"\n--- {name} ---")
        t = pd.DataFrame([{
            "signal": s, "n": v.get("n_questions"),
            "top1": v.get("confidence_top1"), "random": v.get("random_expected_top1"),
            "minus_random": v.get("confidence_minus_random"),
            "ci95": v.get("confidence_minus_random_ci95"),
            "pairwise": v.get("pairwise_accuracy"),
            "regret": v.get("mean_regret"), "norm_regret": v.get("mean_normalized_regret"),
            "sel_gain": v.get("mean_selected_gain"), "oracle": v.get("mean_oracle_gain"),
            "rescue": v.get("selected_rescue_rate"),
        } for s, v in out[key].items()])
        print(t.to_string(index=False))
        any_v = next(iter(out[key].values()))
        print(f"  structural baselines: earliest={any_v.get('earliest_wrong_top1')}  "
              f"latest={any_v.get('latest_wrong_top1')}")
    print("\n--- trace-level aggregation vs question repair value ---")
    print(agg_df.head(10).to_string(index=False))
    print(f"\ncells with p<0.05: {out['trace_aggregation_vs_repair_value']['n_cells_p_below_05']}"
          f" | expected by chance: "
          f"{out['trace_aggregation_vs_repair_value']['n_cells_expected_below_05_by_chance']:.1f}")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
