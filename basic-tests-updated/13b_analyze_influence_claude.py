#!/usr/bin/env python3
from __future__ import annotations

"""
13b_analyze_influence_claude.py

Analysis of 13_step_influence_claude.py. ANALYSIS ONLY, no model loaded.

Answers, in order:

 1. Is influence measurable at all? Compare the |d_gold_logprob| distribution
    against the sham control, which is byte-identical and must give exactly 0.
    If redaction moves the final answer no more than sham does, the final-answer
    step is ignoring the reasoning state and nothing else here is interpretable.

 2. Are steps load-bearing or inert? Report the influence distribution and the
    share of hops whose removal changes the final answer. "Inert" is defined
    against the observed distribution rather than an arbitrary cutoff.

 3. Does influence depend on POSITION? Position was the strongest baseline in
    the repair pilot (latest wrong hop won 8/8). With descendants frozen, this
    asks whether later steps are intrinsically more load-bearing.

 4. Does CONFIDENCE predict influence? This is the load-bearing question. A
    signal that detects errors at ~0.73 AUROC may still be blind to whether a
    step matters. Reported as Spearman, and as AUROC for the top-tercile of
    influence, both marginally and PARTIALLED ON POSITION, since position is
    the competing explanation.

 5. The 2x2. Crossing benchmark correctness with load-bearing/inert gives the
    four cells the project set out to find, and asks whether confidence
    separates them.

Multiple comparisons: several signals x several measures are tested. A
best-of-grid permutation null is run over the confidence-vs-influence family so
the strongest correlation can be read against what the grid produces by chance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score

# confidence orientation: higher value = model is MORE confident
SIGNALS = {
    "mean_logprob": +1.0,
    "min_logprob": +1.0,
    "entropy": -1.0,
    "margin": +1.0,
    "verbal_confidence": +1.0,
}


def partial_spearman(x, y, z):
    """Spearman of x,y controlling for z, via ranks and residualization."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz, dtype=float), rz])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex, ey = rx - Z @ bx, ry - Z @ by
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/qwen17b_influence_claude")
    ap.add_argument("--primary", default="redact",
                    help="Substitution used as the primary influence measure.")
    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir)
    d = pd.read_csv(run_dir / "13_influence.csv")

    out = {"run_dir": str(run_dir), "primary_substitution": args.primary,
           "n_rows": int(len(d))}

    # ---- 1. is influence measurable at all -----------------------------
    sham = d[d.substitution == "sham"]
    prim = d[d.substitution == args.primary].copy()
    out["measurability"] = {
        "n_sham": int(len(sham)),
        "sham_max_abs_d_logprob": float(sham.influence.max()) if len(sham) else None,
        "sham_final_change_rate": float(sham.final_changed.mean()) if len(sham) else None,
        "sham_note": (
            "Sham substitutes the answer with itself, so the prompt is byte-identical. "
            "Anything above 0 here is decoding nondeterminism and is the noise floor."
        ),
    }
    for s, g in d.groupby("substitution"):
        out["measurability"][f"{s}_influence"] = {
            "n": int(len(g)),
            "mean": float(g.influence.mean()),
            "median": float(g.influence.median()),
            "p90": float(g.influence.quantile(0.90)),
            "final_change_rate": float(g.final_changed.mean()),
            "mean_d_logprob_signed": float(g.d_gold_logprob.mean()),
        }

    # ---- 2. load-bearing vs inert --------------------------------------
    hi = prim.influence.quantile(2 / 3)
    lo = prim.influence.quantile(1 / 3)
    prim["load_class"] = np.where(prim.influence >= hi, "load_bearing",
                          np.where(prim.influence <= lo, "inert", "middle"))
    out["load_bearing"] = {
        "tercile_cutoffs": {"inert_max": float(lo), "load_bearing_min": float(hi)},
        "counts": {k: int(v) for k, v in prim.load_class.value_counts().items()},
        "final_change_rate_by_class": {
            k: float(g.final_changed.mean()) for k, g in prim.groupby("load_class")},
        "share_of_hops_whose_removal_changes_final": float(prim.final_changed.mean()),
    }

    # ---- 3. position ----------------------------------------------------
    pos_rows = {}
    for col in ["target_hop", "hops_from_end", "hop_position_frac"]:
        r = spearmanr(prim[col], prim.influence)
        pos_rows[col] = {"spearman": float(r.statistic), "p": float(r.pvalue)}
    out["position"] = {
        "correlations_with_influence": pos_rows,
        "influence_by_hop": {
            str(int(k)): {"n": int(len(g)), "mean_influence": float(g.influence.mean()),
                          "median_influence": float(g.influence.median()),
                          "final_change_rate": float(g.final_changed.mean())}
            for k, g in prim.groupby("target_hop")},
        "influence_by_hops_from_end": {
            str(int(k)): {"n": int(len(g)), "mean_influence": float(g.influence.mean()),
                          "median_influence": float(g.influence.median())}
            for k, g in prim.groupby("hops_from_end")},
    }

    # ---- 4. does confidence predict influence --------------------------
    conf_rows = []
    for sig, sign in SIGNALS.items():
        x = sign * pd.to_numeric(prim[sig], errors="coerce")
        m = np.isfinite(x) & np.isfinite(prim.influence)
        if m.sum() < 10:
            continue
        xv, iv = x[m].to_numpy(), prim.influence[m].to_numpy()
        posv = prim.target_hop[m].to_numpy(dtype=float)
        r = spearmanr(xv, iv)
        # AUROC for "is this hop in the top influence tercile"
        ylab = (iv >= hi).astype(int)
        au = roc_auc_score(ylab, -xv) if len(set(ylab)) == 2 else np.nan
        # and the error-detection AUROC on the same rows, for contrast
        lab = prim.hop_label[m]
        ok = lab.isin(["correct", "incorrect"]).to_numpy()
        yerr = (lab[ok] == "incorrect").astype(int).to_numpy()
        au_err = (roc_auc_score(yerr, -xv[ok])
                  if len(set(yerr)) == 2 and ok.sum() > 5 else np.nan)
        conf_rows.append({
            "signal": sig, "n": int(m.sum()),
            "spearman_conf_vs_influence": float(r.statistic),
            "p_value": float(r.pvalue),
            "partial_spearman_controlling_position": partial_spearman(xv, iv, posv),
            "auroc_predicting_top_influence_tercile": float(au),
            "auroc_predicting_hop_is_wrong": float(au_err),
        })
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(run_dir / "13b_confidence_vs_influence.csv", index=False)
    out["confidence_vs_influence"] = conf.to_dict("records")
    out["confidence_vs_influence_note"] = (
        "auroc_predicting_hop_is_wrong vs auroc_predicting_top_influence_tercile is the "
        "central contrast: a signal can detect that a step is WRONG while being blind to "
        "whether the step MATTERS."
    )

    # best-of-grid permutation null over the confidence-vs-influence family
    if len(conf):
        from scipy.stats import rankdata
        X = np.column_stack([
            (SIGNALS[s] * pd.to_numeric(prim[s], errors="coerce")).to_numpy()
            for s in conf.signal])
        keep = np.isfinite(X).all(axis=1) & np.isfinite(prim.influence.to_numpy())
        Xk = np.apply_along_axis(rankdata, 0, X[keep])
        yk = rankdata(prim.influence.to_numpy()[keep])
        a = Xk - Xk.mean(0, keepdims=True)
        obs = np.abs((a * (yk - yk.mean())[:, None]).sum(0) /
                     np.sqrt((a ** 2).sum(0) * ((yk - yk.mean()) ** 2).sum()))
        best = np.empty(args.permutations)
        for b in range(args.permutations):
            yp = yk[rng.permutation(len(yk))]
            best[b] = np.nanmax(np.abs((a * (yp - yp.mean())[:, None]).sum(0) /
                     np.sqrt((a ** 2).sum(0) * ((yp - yp.mean()) ** 2).sum())))
        out["confidence_vs_influence_permutation"] = {
            "observed_best_abs_spearman": float(np.nanmax(obs)),
            "best_signal": str(conf.signal.iloc[int(np.nanargmax(obs))]),
            "null_p95": float(np.percentile(best, 95)),
            "p_value": float((np.sum(best >= np.nanmax(obs)) + 1) / (args.permutations + 1)),
        }

    # ---- 5. the 2x2 -----------------------------------------------------
    sc = prim[prim.hop_label.isin(["correct", "incorrect"])].copy()
    sc = sc[sc.load_class != "middle"]
    cells = {}
    for (lab, cls), g in sc.groupby(["hop_label", "load_class"]):
        cells[f"{lab}|{cls}"] = {
            "n": int(len(g)),
            "mean_influence": float(g.influence.mean()),
            "mean_verbal_confidence": float(
                pd.to_numeric(g.verbal_confidence, errors="coerce").mean()),
            "mean_mean_logprob": float(pd.to_numeric(g.mean_logprob, errors="coerce").mean()),
            "final_change_rate": float(g.final_changed.mean()),
        }
    out["two_by_two"] = {
        "cells": cells,
        "definition": ("rows = benchmark correctness of the step; columns = influence "
                       "tercile with the middle tercile dropped"),
    }
    # do wrong load-bearing steps get lower confidence than wrong inert steps?
    for sig in SIGNALS:
        try:
            a = pd.to_numeric(sc[(sc.hop_label == "incorrect") &
                                 (sc.load_class == "load_bearing")][sig], errors="coerce").dropna()
            b = pd.to_numeric(sc[(sc.hop_label == "incorrect") &
                                 (sc.load_class == "inert")][sig], errors="coerce").dropna()
            if len(a) > 3 and len(b) > 3:
                u = mannwhitneyu(a, b)
                out["two_by_two"].setdefault("wrong_loadbearing_vs_wrong_inert", {})[sig] = {
                    "n_load_bearing": int(len(a)), "n_inert": int(len(b)),
                    "mean_load_bearing": float(a.mean()), "mean_inert": float(b.mean()),
                    "mannwhitney_p": float(u.pvalue),
                }
        except Exception:
            pass

    out["caveats"] = [
        "Frozen substitution: descendants are NOT regenerated, so this measures how much "
        "the final read-out depends on the state, not the full causal effect in a rollout.",
        "One model, one dataset, one seed.",
        "Influence terciles are relative to this sample, not an absolute standard.",
        "Read the permutation p-value before any single correlation.",
    ]

    (run_dir / "13b_influence_analysis.json").write_text(json.dumps(out, indent=2))

    print("\n============== STEP INFLUENCE ANALYSIS ==============")
    print(json.dumps({k: out[k] for k in
                      ["measurability", "load_bearing", "position"]}, indent=2))
    print("\n--- confidence vs influence (the load-bearing question) ---")
    if len(conf):
        print(conf.to_string(index=False))
        print("\npermutation:", json.dumps(
            out.get("confidence_vs_influence_permutation", {}), indent=2))
    print("\n--- 2x2 ---")
    print(json.dumps(out["two_by_two"], indent=2))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
