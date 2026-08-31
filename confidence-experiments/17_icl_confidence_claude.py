#!/usr/bin/env python3
from __future__ import annotations

"""
17_icl_confidence_claude.py

Can IN-CONTEXT LEARNING teach the model to use step-level confidence?

BACKGROUND
----------
A prior experiment supplied per-step confidence directly to the final-answer
step and found no effect: accuracy was identical whether the confidences were
correctly assigned or randomly permuted across steps (delta accuracy exactly
0.000). That tested whether a bare number is usable. It did not test whether
DEMONSTRATIONS can teach the model what to do with one, which is a different
mechanism and the one worth testing next.

CONFIDENCE SCORE
----------------
Per-step top1-top2 logit `margin`, the signal that best predicted final
correctness in the aggregation grid (AUROC 0.866 as a trace mean). Margin is
mapped to a 0-100 display scale by a GLOBAL min-max over all hops in the
dataset, so the number means the same thing across questions. That is a
monotone rescaling for display; it is NOT calibration and these are NOT
probabilities. The trace-level score shown is the mean of the per-step values.

CONDITIONS (hop answers identical in all five; only the final prompt differs)
----------------------------------------------------------------------------
  zero_shot          plain final prompt, no confidence           [replicates baseline]
  zero_shot_conf     + per-step confidence, no demonstrations    [replicates prior null]
  icl_noconf         + demonstrations WITHOUT confidence
  icl_conf           + demonstrations WITH confidence, showing a low-confidence
                     step being re-checked rather than taken at face value
  icl_conf_shuffled  same as icl_conf, but the TEST item's confidences are
                     permuted across its steps

THE THREE CONTRASTS THAT MATTER
-------------------------------
  icl_conf vs icl_noconf         does confidence add anything beyond the
                                 demonstrations themselves?
  icl_conf vs icl_conf_shuffled  does the model use WHICH step is uncertain?
  icl_conf vs zero_shot_conf     do demonstrations teach use of a signal that
                                 was inert when supplied bare?

Without `icl_noconf`, a gain over zero-shot cannot be separated from the
generic benefit of having worked examples in the prompt. Without the shuffled
arm, a gain cannot be attributed to confidence values at all.

DEMONSTRATIONS
--------------
Drawn from real traces, chosen deterministically and HELD OUT of evaluation:
one where the lowest-confidence step is genuinely wrong (teaching the model to
discount it) and one where every step is confident and correct (teaching it not
to over-correct). Demonstrations omit their source documents -- only the
reasoning state and the correct final answer are shown -- since including four
document sets would exceed the context window.

Imports `common.py` from the original pilot rather than forking it, so every
number here stays comparable to previously published results.
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import _paths
from common import (
    batched_generate_with_confidence, batched_target_logprob, context_block,
    environment_metadata, final_prompt, grade_answer, load_musique,
    load_subject_model, read_jsonl, seed_everything, write_json,
)

SYS_PLAIN = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Return only the shortest factual answer span to the main question. Do not explain."
)
SYS_CONF = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Each step carries a CONFIDENCE value from 0 to 100 indicating how reliable that "
    "step's answer is. Rely on high-confidence steps. Where a step has low confidence, "
    "do not take its answer at face value -- re-check it against the documents before "
    "using it. Return only the shortest factual answer span to the main question. "
    "Do not explain."
)


def chat(tok, system, user):
    m = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def state_lines(qs, ans, conf=None):
    if conf is None:
        return "\n".join(f"- {q} -> {a}" for q, a in zip(qs, ans))
    return "\n".join(f"- {q} -> {a}  [CONFIDENCE: {c:.0f}]"
                     for q, a, c in zip(qs, ans, conf))


def demo_block(demos, with_conf):
    """Worked examples. Documents are omitted; only state and answer are shown."""
    out = ["Here are worked examples of answering from a reasoning state."]
    for d in demos:
        out.append("")
        out.append(f"MAIN QUESTION:\n{d['main_question']}")
        body = state_lines(d["questions"], d["answers"],
                           d["conf"] if with_conf else None)
        out.append(f"REASONING STATE:\n{body}")
        if with_conf:
            out.append(f"TRACE CONFIDENCE: {np.mean(d['conf']):.0f}")
            out.append(f"REASONING: {d['rationale']}")
        out.append(f"SHORT ANSWER: {d['gold_final']}")
    return "\n".join(out)


def build_prompt(tok, ctx, main_q, qs, ans, conf, cond, demos):
    if cond == "zero_shot":
        return final_prompt(tok, ctx, main_q, list(zip(qs, ans)))
    with_conf = cond != "icl_noconf"
    sysmsg = SYS_CONF if with_conf else SYS_PLAIN
    parts = [f"DOCUMENTS:\n{ctx}", ""]
    if cond.startswith("icl"):
        parts += [demo_block(demos, with_conf), "", "Now answer this one.", ""]
    parts.append("REASONING STATE:\n" + state_lines(qs, ans, conf if with_conf else None))
    if with_conf:
        parts.append(f"TRACE CONFIDENCE: {np.mean(conf):.0f}")
    parts += [f"\nMAIN QUESTION:\n{main_q}", "\nSHORT ANSWER:"]
    return chat(tok, sysmsg, "\n".join(parts))


def pick_demos(traces, conf_of):
    """
    Deterministic, illustrative, held out of evaluation:
      demo A -- the lowest-confidence step is benchmark-WRONG (discount it)
      demo B -- every step confident and correct (do not over-correct)
    """
    a = b = None
    for tr in traces:
        hops = sorted(tr["hops"], key=lambda x: int(x["hop_idx"]))
        if len(hops) < 3:
            continue
        c = conf_of(hops)
        lab = [h["label"] for h in hops]
        lo = int(np.argmin(c))
        if a is None and lab[lo] == "incorrect" and (sorted(c)[1] - min(c)) > 8:
            a = (tr, hops, c, lo)
        if b is None and all(x == "correct" for x in lab) and min(c) > 55:
            b = (tr, hops, c, None)
        if a and b:
            break
    demos = []
    for item, kind in [(a, "low"), (b, "high")]:
        if item is None:
            continue
        tr, hops, c, lo = item
        rationale = (
            f"Step {lo+1} has the lowest confidence ({c[lo]:.0f}), so its answer "
            f"'{hops[lo]['pred']}' is unreliable. Re-checking the documents gives "
            f"'{hops[lo]['gold']}' instead, which changes the final answer."
            if kind == "low" else
            "Every step is high confidence, so the reasoning state can be used as given."
        )
        demos.append({
            "question_id": tr["question_id"],
            "main_question": tr["main_question"],
            "questions": [h["model_resolved_question"] for h in hops],
            "answers": [h["pred"] for h in hops],
            "conf": list(c), "rationale": rationale,
            "gold_final": tr["gold_final"], "kind": kind,
        })
    return demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/icl_claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--conditions",
                    default="zero_shot,zero_shot_conf,icl_noconf,icl_conf,icl_conf_shuffled")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    traces = read_jsonl(_paths.TRACES)
    model_name = args.model or json.loads(_paths.SUMMARY.read_text())["environment"]["model"]

    # global min-max map of margin -> 0..100 (monotone display rescale, not calibration)
    allm = [float(h["margin"]) for tr in traces for h in tr["hops"]
            if np.isfinite(float(h.get("margin", np.nan)))]
    LO, HI = float(np.min(allm)), float(np.max(allm))

    def conf_of(hops):
        v = np.array([float(h["margin"]) for h in hops], dtype=float)
        return 100.0 * (v - LO) / (HI - LO)

    demos = pick_demos(traces, conf_of)
    demo_ids = {d["question_id"] for d in demos}
    write_json(run_dir / "17_demonstrations.json", demos)
    print(f"demonstrations ({len(demos)}), held out of evaluation:")
    for d in demos:
        print(f"  [{d['kind']}] {d['question_id']}  conf={[round(x) for x in d['conf']]}")

    eval_traces = [t for t in traces if t["question_id"] not in demo_ids]
    if args.limit_questions:
        eval_traces = eval_traces[:args.limit_questions]
    print(f"evaluating on {len(eval_traces)} held-out questions\n")

    ds = load_musique()
    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    csv_path = run_dir / "17_icl.csv"; jl = run_dir / "17_icl.jsonl"
    done = set()
    if args.resume and jl.exists():
        done = {(r["question_id"], r["condition"]) for r in read_jsonl(jl)}
    sink = open(jl, "a" if args.resume else "w", encoding="utf-8")
    rows = pd.read_csv(csv_path).to_dict("records") if (args.resume and csv_path.exists()) else []
    t0 = time.perf_counter()

    try:
        for qnum, tr in enumerate(eval_traces, start=1):
            ex = ds[int(tr["dataset_idx"])]
            ctx = context_block(ex)
            hops = sorted(tr["hops"], key=lambda x: int(x["hop_idx"]))
            qs = [str(h["model_resolved_question"]) for h in hops]
            ans = [str(h["pred"]) for h in hops]
            conf = conf_of(hops)
            shuf = conf[rng.permutation(len(conf))]
            gold_final = str(tr["gold_final"])
            aliases = list(tr.get("answer_aliases", []) or [])
            print(f"\n=== q{qnum}/{len(eval_traces)} {tr['question_id']} "
                  f"| conf={[round(x) for x in conf]} ===", flush=True)

            for cond in conds:
                if (str(tr["question_id"]), cond) in done:
                    continue
                c = shuf if cond == "icl_conf_shuffled" else conf
                p = build_prompt(tok, ctx, str(tr["main_question"]), qs, ans, c, cond, demos)
                lp = float(batched_target_logprob(
                    model, tok, device, [p], [gold_final], batch_size=1,
                    progress_label=f"q{qnum}-{cond}-lp")[0]["mean_logprob"])
                r = batched_generate_with_confidence(
                    model, tok, device, [p], batch_size=1,
                    max_new_tokens=args.max_new_tokens,
                    progress_label=f"q{qnum}-{cond}-gen")[0]
                pred = str(r["answer"])
                label, f1, _ = grade_answer(pred, gold_final, aliases)
                row = {
                    "question_id": str(tr["question_id"]), "n_hops": int(tr["n_hops"]),
                    "condition": cond, "confidences": ",".join(f"{x:.0f}" for x in c),
                    "trace_confidence": float(np.mean(c)),
                    "gold_final_logprob": lp, "final_pred": pred, "final_label": label,
                    "final_correct": bool(label == "correct"), "final_f1": float(f1),
                    "gold_final": gold_final, "prompt_tokens": int(r["prompt_tokens"]),
                    "test06_label": str(tr["baseline_final_label"]),
                }
                rows.append(row); sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    finally:
        sink.close()

    df = pd.DataFrame(rows); df.to_csv(csv_path, index=False)

    piv_lp = df.pivot_table(index="question_id", columns="condition", values="gold_final_logprob")
    piv_ok = df.pivot_table(index="question_id", columns="condition", values="final_correct")
    rngb = np.random.default_rng(0)

    def paired(a, b):
        if a not in piv_lp.columns or b not in piv_lp.columns:
            return None
        d = (piv_lp[a] - piv_lp[b]).dropna().to_numpy()
        k = (piv_ok[a] - piv_ok[b]).dropna().astype(float).to_numpy()
        bl = [float(d[s].mean()) for s in (rngb.integers(0, len(d), len(d)) for _ in range(5000))]
        bk = [float(k[s].mean()) for s in (rngb.integers(0, len(k), len(k)) for _ in range(5000))]
        return {"n": int(len(d)),
                "d_gold_logprob": float(d.mean()),
                "d_logprob_ci95": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
                "d_accuracy": float(k.mean()),
                "d_accuracy_ci95": [float(np.percentile(bk, 2.5)), float(np.percentile(bk, 97.5))]}

    summary = {
        "environment": environment_metadata(model_name, device, dtype),
        "confidence_signal": "per-step top1-top2 margin, global min-max to 0-100",
        "n_eval_questions": int(df.question_id.nunique()),
        "demonstrations_held_out": sorted(demo_ids),
        "wall_clock_min": (time.perf_counter() - t0) / 60.0,
        "accuracy_by_condition": {c: float(piv_ok[c].mean()) for c in piv_ok.columns},
        "mean_gold_logprob_by_condition": {c: float(piv_lp[c].mean()) for c in piv_lp.columns},
        "n_needs_review": int((df.final_label == "needs_review").sum()),
        "key_contrasts": {
            "icl_conf_vs_icl_noconf": paired("icl_conf", "icl_noconf"),
            "icl_conf_vs_icl_conf_shuffled": paired("icl_conf", "icl_conf_shuffled"),
            "icl_conf_vs_zero_shot_conf": paired("icl_conf", "zero_shot_conf"),
            "icl_conf_vs_zero_shot": paired("icl_conf", "zero_shot"),
            "icl_noconf_vs_zero_shot": paired("icl_noconf", "zero_shot"),
            "zero_shot_conf_vs_zero_shot": paired("zero_shot_conf", "zero_shot"),
        },
        "reading": (
            "icl_conf vs icl_noconf isolates the value of the confidence numbers on top of "
            "the demonstrations. icl_conf vs icl_conf_shuffled isolates whether the model "
            "uses WHICH step is uncertain. icl_noconf vs zero_shot measures the generic "
            "benefit of worked examples. A gain in icl_conf over zero_shot that is fully "
            "explained by icl_noconf is a demonstration effect, not a confidence effect."
        ),
    }
    write_json(run_dir / "17_summary.json", summary)
    print("\n============ IN-CONTEXT LEARNING ============")
    print("accuracy:", json.dumps(summary["accuracy_by_condition"], indent=1))
    print("key contrasts:", json.dumps(summary["key_contrasts"], indent=1))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
