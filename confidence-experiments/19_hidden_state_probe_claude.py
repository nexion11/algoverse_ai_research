#!/usr/bin/env python3
from __future__ import annotations

"""
19_hidden_state_probe_claude.py

Is the load-bearing information present in the model's internals at all?

MOTIVATION
----------
Every experiment so far asks whether the model USES a confidence signal it is
handed. All are null: supplying per-step confidence bare, demonstrating its use
in-context, and permuting it all give a delta accuracy of exactly 0.000.

And experiment 13 showed the deeper problem: confidence predicts whether a step
is WRONG at AUROC 0.702, but whether a step is LOAD-BEARING at 0.517 -- chance.
No amount of prompting or training can extract information a signal does not
carry.

But nothing has yet tested whether that information exists ANYWHERE in the
model. Verbalized and token-level confidence are both narrow read-outs. This
probes the hidden states directly.

METHOD
------
One forward pass per hop over the same hop prompt Test 06 used. Take the
last-position hidden state at several layers. Train L2 logistic regression to
predict:

    is_wrong        the hop's benchmark label is `incorrect`
    is_load_bearing the hop is in the top influence tercile from experiment 13

Evaluated with GroupKFold grouped by QUESTION, because hops from the same
question share documents and a random split would leak.

BASELINES ON IDENTICAL ROWS AND FOLDS
-------------------------------------
The same cross-validation is run on the four white-box confidence signals and
on verbalized confidence, so probe and baseline are directly comparable. A
label-permutation null is also run per target, since a 2048-dimensional probe
on ~150 rows can fit noise -- the permuted-label AUROC is what says whether an
above-chance number is real.

READING
-------
    probe recovers influence >> 0.517   the model knows which steps matter but
                                        does not express it in confidence; there
                                        is a trainable signal to extract
    probe also at chance                the information is not in the model
                                        either, which explains every null so far

Imports common.py from the pilot; writes only into --run-dir.
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _paths
from common import (
    context_block, environment_metadata, hop_prompt, load_musique,
    load_subject_model, read_jsonl, seed_everything, write_json,
)

SIGNALS = {"mean_logprob": +1.0, "min_logprob": +1.0, "entropy": -1.0,
           "margin": +1.0, "verbal_confidence": +1.0}


@torch.inference_mode()
def hidden_at_last_token(model, tok, device, prompt, layers):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    # slice immediately: keeping every layer's full sequence would be ~GBs
    vecs = {L: out.hidden_states[L][0, -1, :].float().cpu().numpy() for L in layers}
    del out, enc
    if device.type == "mps":
        torch.mps.empty_cache()
    return vecs


def cv_auroc(X, y, groups, n_splits=5, seed=0, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return np.nan
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=2000, random_state=seed))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    m = np.isfinite(oof)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return np.nan
    return float(roc_auc_score(y[m], oof[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/probe_claude")
    ap.add_argument("--influence-csv",
                    default="../basic-tests-updated/outputs/qwen17b_influence_claude/13_influence.csv")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", default="", help="comma-separated; default = 5 spread layers")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--cache", default="")
    args = ap.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    traces = read_jsonl(_paths.TRACES)
    model_name = args.model or json.loads(_paths.SUMMARY.read_text())["environment"]["model"]

    cache = Path(args.cache) if args.cache else run_dir / "19_hidden.npz"
    meta_path = run_dir / "19_hop_meta.csv"

    if cache.exists() and meta_path.exists():
        print(f"loading cached activations from {cache}")
        z = np.load(cache); meta = pd.read_csv(meta_path)
        H = {int(k.split("_")[1]): z[k] for k in z.files}
        model = tok = device = dtype = None
    else:
        ds = load_musique()
        model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)
        nlayer = model.config.num_hidden_layers
        layers = ([int(x) for x in args.layers.split(",") if x.strip()] if args.layers
                  else sorted({1, nlayer // 4, nlayer // 2, 3 * nlayer // 4, nlayer}))
        print(f"model has {nlayer} layers; probing {layers}")

        rows, feats = [], {L: [] for L in layers}
        t0 = time.perf_counter()
        for qn, tr in enumerate(traces, start=1):
            ex = ds[int(tr["dataset_idx"])]
            ctx = context_block(ex)
            for h in sorted(tr["hops"], key=lambda x: int(x["hop_idx"])):
                p = hop_prompt(tok, ctx, str(h["model_resolved_question"]))
                v = hidden_at_last_token(model, tok, device, p, layers)
                for L in layers:
                    feats[L].append(v[L])
                rows.append({"question_id": tr["question_id"], "hop": int(h["hop"]),
                             "hop_idx": int(h["hop_idx"]), "n_hops": int(tr["n_hops"]),
                             "label": h["label"],
                             **{k: h.get(k) for k in SIGNALS}})
            if qn % 10 == 0:
                print(f"  {qn}/{len(traces)} questions | {time.perf_counter()-t0:.0f}s",
                      flush=True)
        meta = pd.DataFrame(rows); meta.to_csv(meta_path, index=False)
        H = {L: np.stack(feats[L]) for L in layers}
        np.savez_compressed(cache, **{f"layer_{L}": H[L] for L in H})
        print(f"cached activations -> {cache}  ({time.perf_counter()-t0:.0f}s)")

    # ---- labels -------------------------------------------------------
    inf = pd.read_csv(args.influence_csv)
    inf = inf[inf.substitution == "redact"][["question_id", "target_hop", "influence"]]
    inf = inf.rename(columns={"target_hop": "hop"})
    meta = meta.merge(inf, on=["question_id", "hop"], how="left")
    hi = meta.influence.quantile(2 / 3)
    meta["is_wrong"] = (meta.label == "incorrect").astype(int)
    meta["is_load_bearing"] = (meta.influence >= hi).astype(float)
    meta.loc[~np.isfinite(meta.influence), "is_load_bearing"] = np.nan

    targets = {
        "is_wrong": meta.label.isin(["correct", "incorrect"]).to_numpy(),
        "is_load_bearing": np.isfinite(meta.influence).to_numpy(),
    }
    groups_all = meta.question_id.to_numpy()

    results, baselines, nulls = {}, {}, {}
    for tgt, mask in targets.items():
        y = meta.loc[mask, tgt].to_numpy().astype(int)
        g = groups_all[mask]
        results[tgt] = {"n": int(mask.sum()), "n_positive": int(y.sum()), "by_layer": {}}
        for L in sorted(H):
            X = H[L][mask]
            results[tgt]["by_layer"][f"layer_{L}"] = cv_auroc(X, y, g, seed=args.seed)
        best_layer = max(results[tgt]["by_layer"],
                         key=lambda k: (results[tgt]["by_layer"][k]
                                        if np.isfinite(results[tgt]["by_layer"][k]) else -1))
        results[tgt]["best_layer"] = best_layer
        results[tgt]["best_auroc"] = results[tgt]["by_layer"][best_layer]

        # baselines: the confidence signals, same rows, same CV
        b = {}
        for s, sign in SIGNALS.items():
            x = (sign * pd.to_numeric(meta.loc[mask, s], errors="coerce")).to_numpy()
            ok = np.isfinite(x)
            if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
                continue
            from sklearn.metrics import roc_auc_score
            b[s] = float(roc_auc_score(y[ok], -x[ok]))   # higher confidence -> lower risk
        baselines[tgt] = b

        # permutation null on the best layer (a 2048-dim probe on ~150 rows can fit noise)
        Lbest = int(best_layer.split("_")[1])
        Xb = H[Lbest][mask]
        perm = []
        for _ in range(args.permutations):
            perm.append(cv_auroc(Xb, y[rng.permutation(len(y))], g, seed=args.seed))
        perm = np.array([p for p in perm if np.isfinite(p)])
        obs = results[tgt]["best_auroc"]
        nulls[tgt] = {
            "permutations": int(len(perm)),
            "null_mean_auroc": float(perm.mean()),
            "null_p95_auroc": float(np.percentile(perm, 95)),
            "observed": float(obs),
            "p_value": float((np.sum(perm >= obs) + 1) / (len(perm) + 1)),
        }

    summary = {
        "environment": (environment_metadata(model_name, device, dtype)
                        if model is not None else {"model": model_name, "note": "cached"}),
        "n_hops": int(len(meta)),
        "probe": "L2 logistic regression on last-token hidden state, GroupKFold by question",
        "results": results,
        "confidence_baselines_same_rows": baselines,
        "permutation_nulls_best_layer": nulls,
        "reading": (
            "Compare probe best_auroc against confidence_baselines_same_rows for the same "
            "target. For is_load_bearing the confidence baselines sit at chance (~0.52). A "
            "probe meaningfully above that, and above its permutation null, means the "
            "information IS present in the model but is not expressed in confidence."
        ),
    }
    write_json(run_dir / "19_probe_summary.json", summary)
    meta.to_csv(run_dir / "19_hop_meta_labeled.csv", index=False)

    print("\n================ HIDDEN-STATE PROBE ================")
    for tgt in results:
        print(f"\n--- {tgt}  (n={results[tgt]['n']}, positives={results[tgt]['n_positive']}) ---")
        for k, v in results[tgt]["by_layer"].items():
            print(f"   probe {k:10s} AUROC = {v:.3f}" if np.isfinite(v) else f"   {k}: n/a")
        print(f"   BEST: {results[tgt]['best_layer']} = {results[tgt]['best_auroc']:.3f}")
        print("   confidence baselines (same rows):")
        for s, v in sorted(baselines[tgt].items(), key=lambda x: -x[1]):
            print(f"      {s:20s} {v:.3f}")
        print(f"   permutation null: mean={nulls[tgt]['null_mean_auroc']:.3f} "
              f"p95={nulls[tgt]['null_p95_auroc']:.3f} p={nulls[tgt]['p_value']:.4f}")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
