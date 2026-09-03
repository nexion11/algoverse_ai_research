#!/usr/bin/env python3
from __future__ import annotations

"""
20_icl_variants_claude.py

Two ICL variants targeting the leading explanation for why every
confidence-in-prompt experiment has been null.

THE HYPOTHESIS
--------------
The model reads the confidence numbers -- logP shifts on 45 of 58 questions when
they change -- but does not act on them. One explanation fits every result so
far: IT HAS NO ROOM TO ACT. In every experiment it receives documents plus a
fixed reasoning state and emits ~16 tokens. Even if it fully believed step 2 was
unreliable, re-deriving step 2 from the documents and recomputing the chain is
not something that fits in a bare short answer. We have been telling the model
to distrust a step while giving it no mechanism to do anything about it.

Two interventions follow, each with its own control.

VARIANT 1 -- CHAIN OF THOUGHT (room to act)
    Demonstrations where the assistant REASONS before answering: it names the
    lowest-confidence step, re-checks it against the documents, and only then
    answers. At test time the model may do the same.
      icl_cot_noconf         CoT demos, no confidence            [control]
      icl_cot_conf           CoT demos with confidence
      icl_cot_conf_shuffled  confidences permuted across steps   [control]

VARIANT 2 -- SALIENT FLAG (one word, not five numbers)
    Numbers spread over every step may read as noise. Here exactly one step is
    marked `[UNRELIABLE - verify against the documents]` and no other step
    carries a value.
      icl_flag_lowest        the flag is on the lowest-confidence step
      icl_flag_random        the flag is on a RANDOM step             [control]

    `lowest` vs `random` is the sharpest test in the whole project: identical
    prompt shape, identical number of flags, only WHICH step is flagged differs.
    If confidence-based selection carries usable information, flagging the right
    step must beat flagging an arbitrary one.

REFERENCE
    icl_noconf   demonstrations without confidence, no CoT -- pairs to
                 experiment 18 so CoT's own contribution is measurable.

SCORING NOTE
------------
CoT conditions generate reasoning then `ANSWER: <span>`; the span is parsed out.
logP(gold) for those arms is scored after the model's own generated reasoning,
so it is conditioned differently from the non-CoT arms and is NOT comparable
across the two families. ACCURACY is the primary metric here and is comparable
throughout.
"""

import argparse, json, re, time
from pathlib import Path

import numpy as np
import pandas as pd

import _paths
from common import (
    batched_generate_with_confidence, batched_target_logprob, context_block,
    environment_metadata, grade_answer, load_musique, load_subject_model,
    normalize, read_jsonl, seed_everything, write_json,
)

SYS_PLAIN = ("Use only the supplied documents and the provided structured reasoning state. "
             "Return only the shortest factual answer span to the main question. Do not explain.")
SYS_CONF = ("Use only the supplied documents and the provided structured reasoning state. "
            "Each step carries a CONFIDENCE value from 0 to 100. Rely on high-confidence "
            "steps; where a step has low confidence, re-check it against the documents and "
            "use what the documents say. Return only the shortest factual answer span. "
            "Do not explain.")
SYS_COT = ("Use only the supplied documents and the provided structured reasoning state. "
           "Each step carries a CONFIDENCE value from 0 to 100. First identify the least "
           "reliable step and re-check it against the documents. Then give the answer. "
           "End your reply with a line of the form 'ANSWER: <shortest factual answer span>'.")
SYS_COT_NOCONF = ("Use only the supplied documents and the provided structured reasoning "
                  "state. First re-check the reasoning state against the documents. Then "
                  "give the answer. End your reply with a line of the form "
                  "'ANSWER: <shortest factual answer span>'.")
SYS_FLAG = ("Use only the supplied documents and the provided structured reasoning state. "
            "One step is marked UNRELIABLE. Do not take that step's answer at face value -- "
            "verify it against the documents and use what the documents say. "
            "Return only the shortest factual answer span. Do not explain.")

FLAG = "  [UNRELIABLE - verify against the documents]"


def state_lines(qs, ans, conf=None, flag_idx=None):
    out = []
    for i, (q, a) in enumerate(zip(qs, ans)):
        line = f"- {q} -> {a}"
        if conf is not None:
            line += f"  [CONFIDENCE: {conf[i]:.0f}]"
        if flag_idx is not None and i == flag_idx:
            line += FLAG
        out.append(line)
    return "\n".join(out)


def user_turn(docs, main_q, qs, ans, conf=None, flag_idx=None, show_trace=True):
    p = [f"DOCUMENTS:\n{docs}", "",
         "REASONING STATE:\n" + state_lines(qs, ans, conf, flag_idx)]
    if conf is not None and show_trace:
        p.append(f"TRACE CONFIDENCE: {np.mean(conf):.0f}")
    p += [f"\nMAIN QUESTION:\n{main_q}", "\nSHORT ANSWER:"]
    return "\n".join(p)


FAMILIES = {
    "icl_noconf":            dict(sys=SYS_PLAIN,       conf=False, cot=False, flag=None),
    "icl_cot_noconf":        dict(sys=SYS_COT_NOCONF,  conf=False, cot=True,  flag=None),
    "icl_cot_conf":          dict(sys=SYS_COT,         conf=True,  cot=True,  flag=None),
    "icl_cot_conf_shuffled": dict(sys=SYS_COT,         conf=True,  cot=True,  flag=None),
    "icl_flag_lowest":       dict(sys=SYS_FLAG,        conf=False, cot=False, flag="lowest"),
    "icl_flag_random":       dict(sys=SYS_FLAG,        conf=False, cot=False, flag="random"),
}


def build_messages(cond, docs, main_q, qs, ans, conf, flag_idx, demos):
    f = FAMILIES[cond]
    msgs = [{"role": "system", "content": f["sys"]}]
    for d in demos:
        dflag = d["low_idx"] if f["flag"] else None
        msgs.append({"role": "user", "content": user_turn(
            d["docs"], d["main_question"], d["questions"], d["answers"],
            d["conf"] if f["conf"] else None, dflag)})
        if f["cot"]:
            msgs.append({"role": "assistant",
                         "content": d["cot_conf"] if f["conf"] else d["cot_noconf"]})
        else:
            msgs.append({"role": "assistant", "content": d["target_answer"]})
    msgs.append({"role": "user", "content": user_turn(
        docs, main_q, qs, ans, conf if f["conf"] else None, flag_idx)})
    return msgs


def render(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_cot(text):
    m = re.findall(r"ANSWER\s*:\s*(.+)", str(text), flags=re.I)
    if m:
        return m[-1].strip().strip("`\"' .")
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    return lines[-1].strip("`\"' .") if lines else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/icl_variants_claude")
    ap.add_argument("--demos", default="outputs/icl_proper_claude/18_demonstrations.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--cot-max-new-tokens", type=int, default=110)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--conditions", default=",".join(FAMILIES))
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

    # reuse the validated demonstrations from experiment 18, add CoT targets
    demos = json.load(open(args.demos))
    for d in demos:
        c = np.array(d["conf"], dtype=float)
        lo = int(np.argmin(c)); d["low_idx"] = lo
        if d["kind"].startswith("A"):
            d["cot_conf"] = (
                f"Step {lo+1} has the lowest confidence ({c[lo]:.0f}), so I re-check it "
                f"against the documents. {d['why'].split(';')[-1].strip().capitalize()}. "
                f"ANSWER: {d['target_answer']}")
            d["cot_noconf"] = (
                f"I re-check the reasoning state against the documents. Step {lo+1} does not "
                f"match the documents, so the chain must be corrected. "
                f"ANSWER: {d['target_answer']}")
        else:
            d["cot_conf"] = (
                f"The lowest confidence is step {lo+1} at {c[lo]:.0f}, which is still high, "
                f"and the documents agree with the state. "
                f"ANSWER: {d['target_answer']}")
            d["cot_noconf"] = (
                f"The documents agree with every step of the reasoning state. "
                f"ANSWER: {d['target_answer']}")
    write_json(run_dir / "20_demonstrations.json", demos)
    demo_ids = {d["question_id"] for d in demos}

    eval_traces = [t for t in traces if t["question_id"] not in demo_ids]
    if args.limit_questions:
        eval_traces = eval_traces[:args.limit_questions]
    print(f"evaluating {len(eval_traces)} held-out questions x {len(conds)} conditions")

    ds = load_musique()
    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    csv_path = run_dir / "20_variants.csv"; jl = run_dir / "20_variants.jsonl"
    done = set()
    if args.resume and jl.exists():
        done = {(r["question_id"], r["condition"]) for r in read_jsonl(jl)}
    sink = open(jl, "a" if args.resume else "w", encoding="utf-8")
    rows = pd.read_csv(csv_path).to_dict("records") if (args.resume and csv_path.exists()) else []
    t0 = time.perf_counter()

    try:
        for qn, t in enumerate(eval_traces, start=1):
            ex = ds[int(t["dataset_idx"])]
            ctx = context_block(ex)
            hops = sorted(t["hops"], key=lambda x: int(x["hop_idx"]))
            qs = [str(h["model_resolved_question"]) for h in hops]
            ans = [str(h["pred"]) for h in hops]
            conf = conf_of(hops)
            shuf = conf[rng.permutation(len(conf))]
            lowest = int(np.argmin(conf))
            others = [i for i in range(len(conf)) if i != lowest]
            rand_idx = int(rng.choice(others)) if others else lowest
            gold_final = str(t["gold_final"])
            aliases = list(t.get("answer_aliases", []) or [])
            print(f"\n=== q{qn}/{len(eval_traces)} {t['question_id']} | "
                  f"conf={[round(x) for x in conf]} lowest=step{lowest+1} "
                  f"random=step{rand_idx+1} ===", flush=True)

            for cond in conds:
                if (str(t["question_id"]), cond) in done:
                    continue
                f = FAMILIES[cond]
                c = shuf if cond.endswith("shuffled") else conf
                fi = {"lowest": lowest, "random": rand_idx}.get(f["flag"])
                p = render(tok, build_messages(cond, ctx, str(t["main_question"]),
                                               qs, ans, c, fi, demos))
                mnt = args.cot_max_new_tokens if f["cot"] else args.max_new_tokens
                r = batched_generate_with_confidence(
                    model, tok, device, [p], batch_size=1, max_new_tokens=mnt,
                    progress_label=f"q{qn}-{cond}")[0]
                raw = str(r.get("raw_answer", ""))
                pred = parse_cot(raw) if f["cot"] else str(r["answer"])
                # logP(gold): for CoT, conditioned on the model's own reasoning
                score_prompt = p + (raw.rsplit("ANSWER:", 1)[0] + "ANSWER: "
                                    if f["cot"] and "ANSWER:" in raw else "")
                lp = float(batched_target_logprob(
                    model, tok, device, [score_prompt], [gold_final], batch_size=1,
                    progress_label=f"q{qn}-{cond}-lp")[0]["mean_logprob"])
                label, f1, _ = grade_answer(pred, gold_final, aliases)
                row = {"question_id": str(t["question_id"]), "n_hops": int(t["n_hops"]),
                       "condition": cond, "is_cot": bool(f["cot"]),
                       "flagged_step": (fi + 1) if fi is not None else None,
                       "lowest_conf_step": lowest + 1,
                       "confidences": ",".join(f"{x:.0f}" for x in c),
                       "final_pred": pred, "raw_output": raw[:400],
                       "gold_final_logprob": lp, "final_label": label,
                       "final_correct": bool(label == "correct"), "final_f1": float(f1),
                       "gold_final": gold_final}
                rows.append(row); sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    finally:
        sink.close()

    df = pd.DataFrame(rows); df.to_csv(csv_path, index=False)
    ok = df.pivot_table(index="question_id", columns="condition", values="final_correct")
    pr = df.pivot_table(index="question_id", columns="condition",
                        values="final_pred", aggfunc="first")
    rngb = np.random.default_rng(0)

    def paired(a, b):
        if a not in ok.columns or b not in ok.columns:
            return None
        k = (ok[a] - ok[b]).dropna().astype(float).to_numpy()
        bk = [float(k[s].mean()) for s in (rngb.integers(0, len(k), len(k)) for _ in range(5000))]
        return {"n": int(len(k)), "d_accuracy": float(k.mean()),
                "d_accuracy_ci95": [float(np.percentile(bk, 2.5)),
                                    float(np.percentile(bk, 97.5))],
                "n_answers_changed": int((pr[a].astype(str) != pr[b].astype(str)).sum())}

    summary = {
        "environment": environment_metadata(model_name, device, dtype),
        "n_eval_questions": int(df.question_id.nunique()),
        "wall_clock_min": (time.perf_counter() - t0) / 60.0,
        "n_needs_review": int((df.final_label == "needs_review").sum()),
        "accuracy_by_condition": {c: float(ok[c].mean()) for c in ok.columns},
        "key_contrasts": {
            "cot_conf_vs_cot_noconf": paired("icl_cot_conf", "icl_cot_noconf"),
            "cot_conf_vs_cot_conf_shuffled": paired("icl_cot_conf", "icl_cot_conf_shuffled"),
            "cot_noconf_vs_noconf": paired("icl_cot_noconf", "icl_noconf"),
            "flag_lowest_vs_flag_random": paired("icl_flag_lowest", "icl_flag_random"),
            "flag_lowest_vs_noconf": paired("icl_flag_lowest", "icl_noconf"),
        },
        "reading": (
            "flag_lowest vs flag_random is the sharpest test: identical prompt shape and "
            "one flag in both, only WHICH step is flagged differs. cot_conf vs "
            "cot_noconf asks whether confidence helps once the model has room to reason. "
            "cot_noconf vs noconf measures what chain-of-thought contributes on its own."
        ),
    }
    write_json(run_dir / "20_summary.json", summary)
    print("\n============ ICL VARIANTS ============")
    print("accuracy:", json.dumps(summary["accuracy_by_condition"], indent=1))
    print("contrasts:", json.dumps(summary["key_contrasts"], indent=1))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
