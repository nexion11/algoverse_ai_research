#!/usr/bin/env python3
from __future__ import annotations

"""
18_icl_proper_claude.py

Proper in-context-learning test of whether demonstrations can teach the model
to use step-level confidence.

WHY THIS REPLACES 17
--------------------
The first attempt (`17_icl_confidence_claude.py`) used demonstrations that did
not demonstrate the target behaviour:

  * In both demos the final answer equalled the LAST STEP's answer, so the demo
    set was perfectly explained by "copy the last step" -- a rule that is
    confidence-independent by construction.
  * The low-confidence demo's rationale claimed that correcting a step "changes
    the final answer", but the answer shown was exactly what ignoring the
    correction produces. The demonstration contradicted its own rationale.
  * Demonstrations carried no documents while the test item carried ~2400
    tokens of them, so the demonstrated task differed from the asked task.
  * Everything was packed into one user turn rather than multi-turn.

Any of those could produce a false null. This version fixes all four.

DEMONSTRATIONS
--------------
Sourced from Test 07 repair results, which contain real cases where replacing a
wrong hop with gold and regenerating produced a DIFFERENT and CORRECT final
answer. That gives demonstrations where discounting a step provably changes the
answer, rather than demonstrations that merely assert it.

  Type A (2)  the lowest-confidence step is wrong; the correct final answer
              differs from what the state naively implies
  Type B (2)  every step is confident and correct; the state should be used
              as given

The script ASSERTS that "copy the last step" fails on the demo set. If a demo
set is ever selected where that heuristic explains every answer, the run aborts
rather than silently repeating the flaw in 17.

Each demonstration includes the real MuSiQue supporting paragraphs for its
steps (via `paragraph_support_idx`), so demo and test share the same
documents -> state -> question -> answer shape. Demos are held out of
evaluation. Conversation is multi-turn: (user, assistant) pairs then the test
item.

CONDITIONS
----------
  zero_shot          plain final prompt, no confidence
  zero_shot_conf     + confidence, no demonstrations
  icl_noconf         + the SAME demonstrations and the SAME answers, confidence
                     stripped. Isolates the demonstrations from the signal:
                     the model sees that the answer sometimes departs from the
                     chain, but not why.
  icl_conf           + demonstrations with confidence
  icl_conf_shuffled  icl_conf with the TEST item's confidences permuted

`icl_conf` vs `icl_noconf` is the contrast that matters: same examples, same
targets, confidence present or absent.
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import _paths
from common import (
    batched_generate_with_confidence, batched_target_logprob, context_block,
    environment_metadata, final_prompt, grade_answer, load_musique,
    load_subject_model, normalize, read_jsonl, seed_everything, write_json,
)

SYS_PLAIN = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Return only the shortest factual answer span to the main question. Do not explain."
)
SYS_CONF = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Each step carries a CONFIDENCE value from 0 to 100 indicating how reliable that "
    "step's answer is. Rely on high-confidence steps. Where a step has low confidence, "
    "do not take its answer at face value -- re-check it against the documents and use "
    "what the documents say instead. "
    "Return only the shortest factual answer span to the main question. Do not explain."
)


def state_lines(qs, ans, conf=None):
    if conf is None:
        return "\n".join(f"- {q} -> {a}" for q, a in zip(qs, ans))
    return "\n".join(f"- {q} -> {a}  [CONFIDENCE: {c:.0f}]"
                     for q, a, c in zip(qs, ans, conf))


def user_turn(docs, main_q, qs, ans, conf):
    parts = [f"DOCUMENTS:\n{docs}", ""]
    parts.append("REASONING STATE:\n" + state_lines(qs, ans, conf))
    if conf is not None:
        parts.append(f"TRACE CONFIDENCE: {np.mean(conf):.0f}")
    parts += [f"\nMAIN QUESTION:\n{main_q}", "\nSHORT ANSWER:"]
    return "\n".join(parts)


def build_messages(ex_docs, main_q, qs, ans, conf, cond, demos):
    with_conf = cond in ("zero_shot_conf", "icl_conf", "icl_conf_shuffled")
    msgs = [{"role": "system", "content": SYS_CONF if with_conf else SYS_PLAIN}]
    if cond.startswith("icl"):
        for d in demos:
            msgs.append({"role": "user", "content": user_turn(
                d["docs"], d["main_question"], d["questions"], d["answers"],
                d["conf"] if with_conf else None)})
            msgs.append({"role": "assistant", "content": d["target_answer"]})
    msgs.append({"role": "user", "content": user_turn(
        ex_docs, main_q, qs, ans, conf if with_conf else None)})
    return msgs


def render(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def support_docs(ex, hops, max_chars=340):
    """Real MuSiQue supporting paragraphs for this trace's steps."""
    by_idx = {p.get("idx"): p for p in ex["paragraphs"]}
    out, seen = [], set()
    for step in ex["question_decomposition"]:
        i = step.get("paragraph_support_idx")
        if i is None or i in seen or i not in by_idx:
            continue
        seen.add(i)
        p = by_idx[i]
        out.append(f"[Document {i}] {p.get('title','')}\n{p['paragraph_text'][:max_chars]}")
    return "\n\n".join(out)


def pick_demos(traces, ds, conf_of, n_a=2, n_b=2):
    rep = pd.read_csv(_paths.SEQ / "07_repairs.csv")
    tr = {t["question_id"]: t for t in traces}
    demos = []

    # Type A: lowest-confidence step is wrong; repairing it yields a DIFFERENT,
    # CORRECT final answer. The demo shows the ORIGINAL state and the corrected answer.
    for _, r in rep.iterrows():
        if len(demos) >= n_a:
            break
        if r.repaired_final_label != "correct":
            continue
        t = tr[r.question_id]
        hops = sorted(t["hops"], key=lambda x: int(x["hop_idx"]))
        c = conf_of(hops)
        i = int(r.repair_hop_idx)
        if int(np.argmin(c)) != i:
            continue                     # the low-confidence step must BE the culprit
        if normalize(str(r.repaired_final_pred)) == normalize(str(t["baseline_final_pred"])):
            continue                     # the correction must change the answer
        ex = ds[int(t["dataset_idx"])]
        demos.append({
            "kind": "A_discount_low_confidence", "question_id": r.question_id,
            "main_question": t["main_question"],
            "questions": [h["model_resolved_question"] for h in hops],
            "answers": [h["pred"] for h in hops], "conf": list(c),
            "target_answer": str(r.repaired_final_pred),
            "naive_last_step": str(hops[-1]["pred"]),
            "docs": support_docs(ex, hops),
            "why": (f"step {i+1} has the lowest confidence ({c[i]:.0f}) and is wrong; "
                    f"the documents give '{r.hop_gold}', changing the answer from "
                    f"'{t['baseline_final_pred']}' to '{r.repaired_final_pred}'"),
        })

    # Type B: every step confident and correct; use the state as given.
    used = {d["question_id"] for d in demos}
    for t in traces:
        if len([d for d in demos if d["kind"].startswith("B")]) >= n_b:
            break
        if t["question_id"] in used or t["baseline_final_label"] != "correct":
            continue
        hops = sorted(t["hops"], key=lambda x: int(x["hop_idx"]))
        c = conf_of(hops)
        if min(c) < 45 or any(h["label"] != "correct" for h in hops):
            continue
        ex = ds[int(t["dataset_idx"])]
        demos.append({
            "kind": "B_trust_high_confidence", "question_id": t["question_id"],
            "main_question": t["main_question"],
            "questions": [h["model_resolved_question"] for h in hops],
            "answers": [h["pred"] for h in hops], "conf": list(c),
            "target_answer": str(t["baseline_final_pred"]),
            "naive_last_step": str(hops[-1]["pred"]),
            "docs": support_docs(ex, hops),
            "why": "every step is high confidence and the state can be used as given",
        })
    return demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/icl_proper_claude")
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
    allm = [float(h["margin"]) for t in traces for h in t["hops"]]
    LO, HI = float(np.min(allm)), float(np.max(allm))

    def conf_of(hops):
        v = np.array([float(h["margin"]) for h in hops], dtype=float)
        return 100.0 * (v - LO) / (HI - LO)

    ds = load_musique()
    demos = pick_demos(traces, ds, conf_of)

    # --- guard against repeating the flaw in 17 -------------------------
    copy_last = sum(normalize(d["target_answer"]) == normalize(d["naive_last_step"])
                    for d in demos)
    print(f"\nDemonstrations: {len(demos)}")
    for d in demos:
        marker = "COPY-LAST" if normalize(d["target_answer"]) == normalize(
            d["naive_last_step"]) else "differs"
        print(f"  [{d['kind']}] {d['question_id']}")
        print(f"     conf={[round(x) for x in d['conf']]}  target='{d['target_answer']}' "
              f"(last step='{d['naive_last_step']}' -> {marker})")
        print(f"     {d['why']}")
    if copy_last == len(demos):
        raise SystemExit(
            "ABORT: every demonstration's answer equals its last step, so the demo set is "
            "explained by 'copy the last step' and cannot teach confidence use. "
            "This is the exact flaw that invalidated experiment 17.")
    print(f"\n'copy the last step' explains {copy_last}/{len(demos)} demos -> "
          f"heuristic FAILS on the set, so it cannot account for a result.\n")
    write_json(run_dir / "18_demonstrations.json", demos)

    demo_ids = {d["question_id"] for d in demos}
    eval_traces = [t for t in traces if t["question_id"] not in demo_ids]
    if args.limit_questions:
        eval_traces = eval_traces[:args.limit_questions]
    print(f"evaluating on {len(eval_traces)} held-out questions")

    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    csv_path = run_dir / "18_icl.csv"; jl = run_dir / "18_icl.jsonl"
    done = set()
    if args.resume and jl.exists():
        done = {(r["question_id"], r["condition"]) for r in read_jsonl(jl)}
    sink = open(jl, "a" if args.resume else "w", encoding="utf-8")
    rows = pd.read_csv(csv_path).to_dict("records") if (args.resume and csv_path.exists()) else []
    t0 = time.perf_counter()

    try:
        for qnum, t in enumerate(eval_traces, start=1):
            ex = ds[int(t["dataset_idx"])]
            ctx = context_block(ex)
            hops = sorted(t["hops"], key=lambda x: int(x["hop_idx"]))
            qs = [str(h["model_resolved_question"]) for h in hops]
            ans = [str(h["pred"]) for h in hops]
            conf = conf_of(hops); shuf = conf[rng.permutation(len(conf))]
            gold_final = str(t["gold_final"])
            aliases = list(t.get("answer_aliases", []) or [])
            print(f"\n=== q{qnum}/{len(eval_traces)} {t['question_id']} ===", flush=True)

            for cond in conds:
                if (str(t["question_id"]), cond) in done:
                    continue
                c = shuf if cond == "icl_conf_shuffled" else conf
                if cond == "zero_shot":
                    p = final_prompt(tok, ctx, str(t["main_question"]), list(zip(qs, ans)))
                else:
                    p = render(tok, build_messages(ctx, str(t["main_question"]),
                                                   qs, ans, c, cond, demos))
                lp = float(batched_target_logprob(
                    model, tok, device, [p], [gold_final], batch_size=1,
                    progress_label=f"q{qnum}-{cond}-lp")[0]["mean_logprob"])
                r = batched_generate_with_confidence(
                    model, tok, device, [p], batch_size=1,
                    max_new_tokens=args.max_new_tokens,
                    progress_label=f"q{qnum}-{cond}-gen")[0]
                pred = str(r["answer"])
                label, f1, _ = grade_answer(pred, gold_final, aliases)
                row = {"question_id": str(t["question_id"]), "n_hops": int(t["n_hops"]),
                       "condition": cond, "confidences": ",".join(f"{x:.0f}" for x in c),
                       "gold_final_logprob": lp, "final_pred": pred, "final_label": label,
                       "final_correct": bool(label == "correct"), "final_f1": float(f1),
                       "gold_final": gold_final, "prompt_tokens": int(r["prompt_tokens"])}
                rows.append(row); sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    finally:
        sink.close()

    df = pd.DataFrame(rows); df.to_csv(csv_path, index=False)
    piv_lp = df.pivot_table(index="question_id", columns="condition", values="gold_final_logprob")
    piv_ok = df.pivot_table(index="question_id", columns="condition", values="final_correct")
    piv_pr = df.pivot_table(index="question_id", columns="condition",
                            values="final_pred", aggfunc="first")
    rngb = np.random.default_rng(0)

    def paired(a, b):
        if a not in piv_lp.columns or b not in piv_lp.columns:
            return None
        d = (piv_lp[a] - piv_lp[b]).dropna().to_numpy()
        k = (piv_ok[a] - piv_ok[b]).dropna().astype(float).to_numpy()
        bl = [float(d[s].mean()) for s in (rngb.integers(0, len(d), len(d)) for _ in range(5000))]
        bk = [float(k[s].mean()) for s in (rngb.integers(0, len(k), len(k)) for _ in range(5000))]
        return {"n": int(len(d)), "d_gold_logprob": float(d.mean()),
                "d_logprob_ci95": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
                "d_accuracy": float(k.mean()),
                "d_accuracy_ci95": [float(np.percentile(bk, 2.5)), float(np.percentile(bk, 97.5))],
                "n_answers_changed": int((piv_pr[a].astype(str) != piv_pr[b].astype(str)).sum())}

    summary = {
        "environment": environment_metadata(model_name, device, dtype),
        "supersedes": "17_icl_confidence_claude.py (invalid demonstrations)",
        "n_eval_questions": int(df.question_id.nunique()),
        "n_demonstrations": len(demos),
        "demonstrations_held_out": sorted(demo_ids),
        "copy_last_step_explains_n_demos": int(copy_last),
        "wall_clock_min": (time.perf_counter() - t0) / 60.0,
        "n_needs_review": int((df.final_label == "needs_review").sum()),
        "accuracy_by_condition": {c: float(piv_ok[c].mean()) for c in piv_ok.columns},
        "mean_gold_logprob_by_condition": {c: float(piv_lp[c].mean()) for c in piv_lp.columns},
        "key_contrasts": {
            "icl_conf_vs_icl_noconf": paired("icl_conf", "icl_noconf"),
            "icl_conf_vs_icl_conf_shuffled": paired("icl_conf", "icl_conf_shuffled"),
            "icl_conf_vs_zero_shot_conf": paired("icl_conf", "zero_shot_conf"),
            "icl_noconf_vs_zero_shot": paired("icl_noconf", "zero_shot"),
            "icl_conf_vs_zero_shot": paired("icl_conf", "zero_shot"),
        },
    }
    write_json(run_dir / "18_summary.json", summary)
    print("\n============ IN-CONTEXT LEARNING (proper) ============")
    print("accuracy:", json.dumps(summary["accuracy_by_condition"], indent=1))
    print("contrasts:", json.dumps(summary["key_contrasts"], indent=1))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
