#!/usr/bin/env python3
from __future__ import annotations

"""
21_cot_icl_proper_claude.py

Chain-of-thought in-context learning, rebuilt after the CoT arm of experiment 20
was found invalid.

WHY 20's CoT ARM WAS INVALID
----------------------------
Its demonstration rationales were generated from two string templates, so the
prompt contained only two surface forms. The model copied the form rather than
the behaviour. The smoking gun, from a real output:

    "The lowest confidence is step 1 at 17, which is still high, and the
     documents agree with the state. ANSWER: 2010"

17 is near the bottom of the scale, not "still high". The model reproduced the
Type-B template verbatim and slotted in a number that contradicts it. Mean
output length was 147 characters -- almost exactly the template length. No
re-checking occurred, so the arm measured sentence-pattern copying, not
confidence use.

WHAT IS FIXED HERE
------------------
1. RATIONALES ARE HAND-WRITTEN AND STRUCTURALLY DISTINCT. Each of the four
   demonstrations reasons in a different shape: forward verification naming a
   specific factual error; backward reasoning from an implausible answer type;
   brief per-step confirmation; noting the weakest link is nonetheless adequate.
   No shared skeleton, no "step N has the lowest confidence (X)" formula, and
   deliberately varied lengths and openings, so surface-form copying cannot
   succeed.

2. DOCUMENTS ARE MATCHED. Demonstrations carry the SAME full paragraph set the
   test item carries, distractors included. In 20 the demos held only supporting
   paragraphs (~2k chars) while the test held all twenty (~9.5k), so
   "re-check against the documents" meant something easier in the demos than at
   test.

3. TWO NEW VALIDITY CHECKS, because the previous failure was invisible in the
   headline numbers:

   PARROT RATE   maximum similarity between each generated rationale and the
                 demonstration rationales. A high rate means the model is
                 copying and the arm is invalid, exactly as in 20.

   STEP-ID CHECK for confidence conditions, parse which step number the model
                 names as least reliable and compare it against the true argmin
                 of the supplied confidences. This directly asks whether the
                 model READS the numbers, independently of whether acting on
                 them helps. It is the diagnostic that would have caught 20.

CONDITIONS
----------
    cot_zeroshot        CoT instruction, no demonstrations, no confidence
    cot_noconf          CoT demonstrations, no confidence
    cot_conf            CoT demonstrations with confidence
    cot_conf_shuffled   as cot_conf, test-item confidences permuted across steps

    cot_conf vs cot_noconf        does confidence help once there is room to reason
    cot_conf vs cot_conf_shuffled does the model use WHICH step is uncertain
    cot_noconf vs cot_zeroshot    what the demonstrations contribute on their own
"""

import argparse, json, re, time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

import _paths
from common import (
    batched_generate_with_confidence, batched_target_logprob, context_block,
    environment_metadata, grade_answer, load_musique, load_subject_model,
    read_jsonl, seed_everything, write_json,
)

SYS_CONF = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Each step carries a CONFIDENCE value from 0 to 100 indicating how reliable that "
    "step's answer is. Work out which step is least reliable, check it against the "
    "documents, and correct the chain if the documents disagree with it. "
    "Then end your reply with a line of the form 'ANSWER: <shortest factual answer span>'."
)
SYS_NOCONF = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Check the reasoning state against the documents and correct the chain where the "
    "documents disagree with it. "
    "Then end your reply with a line of the form 'ANSWER: <shortest factual answer span>'."
)

# Hand-written, deliberately heterogeneous. Keyed by question_id.
# Each pairs a confidence-aware rationale with a confidence-free counterpart that
# reaches the same answer, so `cot_noconf` is a fair control rather than a weaker prompt.
RATIONALES = {
    # forward verification, names a concrete factual error
    "4hop2__71753_158279_70784_61381": dict(
        conf=("Steps 1 and 2 line up with the documents. Step 3 is the weakest link at 29, "
              "and it is wrong: Arabia Terra is a region on Mars, not a region lying north "
              "of the Middle East. The documents put the Kingdom of Saudi Arabia there. "
              "Saudi Arabia was founded in 1932.\nANSWER: 1932"),
        noconf=("Steps 1 and 2 line up with the documents. Step 3 does not: Arabia Terra is a "
                "region on Mars, not a region lying north of the Middle East. The documents "
                "put the Kingdom of Saudi Arabia there. Saudi Arabia was founded in 1932."
                "\nANSWER: 1932")),
    # backward reasoning from an implausible answer type
    "3hop1__409517_547811_80702": dict(
        conf=("A bridge was asked for, yet the chain ends on 'Mandolin', which is an "
              "instrument -- something went wrong upstream. The least trustworthy step is "
              "step 2, at 10, and indeed the documents place Vivaldi's birth in Venice "
              "rather than Paris. The famous bridge in Venice is the Rialto."
              "\nANSWER: Rialto Bridge"),
        noconf=("A bridge was asked for, yet the chain ends on 'Mandolin', which is an "
                "instrument -- something went wrong upstream. The documents place Vivaldi's "
                "birth in Venice rather than Paris. The famous bridge in Venice is the "
                "Rialto.\nANSWER: Rialto Bridge")),
    # terse per-step confirmation
    "3hop1__857_846_7795": dict(
        conf=("The edict names the Karmapa, the Yongle Emperor received him at Nanjing, and "
              "Nanjing's 2011 population is stated directly in the documents. Nothing here "
              "is weak enough to overturn.\nANSWER: 8.11 million"),
        noconf=("The edict names the Karmapa, the Yongle Emperor received him at Nanjing, and "
                "Nanjing's 2011 population is stated directly in the documents. Nothing "
                "conflicts.\nANSWER: 8.11 million")),
    # weakest-link framing, concludes no correction is needed
    "4hop1__152146_5274_458768_33677": dict(
        conf=("Even the shakiest step here, the second one at 57, survives checking -- the "
              "documents do name Universal Music Group as the only group larger than Sony "
              "Music Entertainment. With the chain intact, the award year stands as given."
              "\nANSWER: 2013"),
        noconf=("Each link survives checking -- the documents do name Universal Music Group as "
                "the only group larger than Sony Music Entertainment, and the headquarters "
                "and award year follow. The chain is intact.\nANSWER: 2013")),
}


def demo_docs(ex, n_paragraphs, seed=0):
    """
    Demonstration document set: every supporting paragraph plus distractors up to
    n_paragraphs, in original order.

    The mismatch being corrected from experiment 20 is that demos held ONLY clean
    supporting paragraphs while the test item held twenty with distractors, so
    "re-check against the documents" was an easier task in the demos than at test.
    What matters is that evidence must be LOCATED among distractors, not that the
    count matches exactly -- a full 20-paragraph set in all four demos yields a
    ~12.8k-token prompt and ~75s per call on MPS.
    """
    rng = np.random.default_rng(seed)
    paras = list(ex["paragraphs"])
    sup = [p for p in paras if p.get("is_supporting")]
    other = [p for p in paras if not p.get("is_supporting")]
    take = min(max(0, n_paragraphs - len(sup)), len(other))
    keep_idx = {id(p) for p in sup}
    if take:
        for p in rng.choice(np.array(other, dtype=object), size=take, replace=False):
            keep_idx.add(id(p))
    kept = [p for p in paras if id(p) in keep_idx]
    return "\n\n".join(
        f"[Document {p.get('idx','?')}] {p.get('title','')}\n{p['paragraph_text']}"
        for p in kept)


def state_lines(qs, ans, conf=None):
    if conf is None:
        return "\n".join(f"- {q} -> {a}" for q, a in zip(qs, ans))
    return "\n".join(f"- {q} -> {a}  [CONFIDENCE: {c:.0f}]"
                     for q, a, c in zip(qs, ans, conf))


def user_turn(docs, main_q, qs, ans, conf):
    p = [f"DOCUMENTS:\n{docs}", "", "REASONING STATE:\n" + state_lines(qs, ans, conf)]
    if conf is not None:
        p.append(f"TRACE CONFIDENCE: {np.mean(conf):.0f}")
    p += [f"\nMAIN QUESTION:\n{main_q}", "\nSHORT ANSWER:"]
    return "\n".join(p)


def build_messages(cond, docs, main_q, qs, ans, conf, demos):
    use_conf = cond in ("cot_conf", "cot_conf_shuffled")
    msgs = [{"role": "system", "content": SYS_CONF if use_conf else SYS_NOCONF}]
    if cond != "cot_zeroshot":
        for d in demos:
            msgs.append({"role": "user", "content": user_turn(
                d["docs"], d["main_question"], d["questions"], d["answers"],
                d["conf"] if use_conf else None)})
            msgs.append({"role": "assistant",
                         "content": d["rationale_conf"] if use_conf else d["rationale_noconf"]})
    msgs.append({"role": "user", "content": user_turn(
        docs, main_q, qs, ans, conf if use_conf else None)})
    return msgs


def render(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_answer(text):
    m = re.findall(r"ANSWER\s*:\s*(.+)", str(text), flags=re.I)
    if m:
        return m[-1].strip().strip("`\"' .")
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    return lines[-1].strip("`\"' .") if lines else ""


def named_step(text):
    """Which step does the model call least reliable? Returns 1-based index or None."""
    t = str(text).split("ANSWER:")[0]
    pats = [r"step\s+(\d+)\s+(?:is|has|at|,)", r"(?:weakest|least|lowest|shakiest)[^.]{0,40}?step\s+(\d+)",
            r"step\s+(\d+)[^.]{0,40}?(?:weakest|least reliable|lowest)"]
    for p in pats:
        m = re.search(p, t, flags=re.I)
        if m:
            return int(m.group(1))
    m = re.search(r"step\s+(\d+)", t, flags=re.I)
    return int(m.group(1)) if m else None


def parrot_score(text, rationales):
    body = str(text).split("ANSWER:")[0].strip()
    if len(body) < 15:
        return 0.0
    return max(SequenceMatcher(None, body,
                               r.split("ANSWER:")[0].strip()).ratio() for r in rationales)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/cot_proper_claude")
    ap.add_argument("--base-demos", default="outputs/icl_proper_claude/18_demonstrations.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--demo-paragraphs", type=int, default=8,
                    help="Paragraphs per demonstration: all supporting, plus distractors.")
    ap.add_argument("--score-logprob", action="store_true",
                    help="Also teacher-force logP(gold). Doubles runtime. Accuracy is the "
                         "primary metric here because CoT logP is conditioned on the "
                         "model's own generated reasoning and is not comparable to "
                         "non-CoT arms.")
    ap.add_argument("--conditions", default="cot_zeroshot,cot_noconf,cot_conf,cot_conf_shuffled")
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
    by_id = {t["question_id"]: t for t in traces}
    demos = []
    for base in json.load(open(args.base_demos)):
        qid = base["question_id"]
        if qid not in RATIONALES:
            raise SystemExit(f"no hand-written rationale for demo {qid}")
        t = by_id[qid]
        demos.append({**base,
                      "docs": demo_docs(ds[int(t["dataset_idx"])],
                                        args.demo_paragraphs, seed=args.seed),
                      "rationale_conf": RATIONALES[qid]["conf"],
                      "rationale_noconf": RATIONALES[qid]["noconf"]})

    # verify the rationales are genuinely heterogeneous
    sims = [SequenceMatcher(None, a["rationale_conf"].split("ANSWER:")[0],
                            b["rationale_conf"].split("ANSWER:")[0]).ratio()
            for i, a in enumerate(demos) for b in demos[i + 1:]]
    print(f"demonstration rationales: {len(demos)}")
    print(f"  pairwise similarity  max={max(sims):.3f}  mean={np.mean(sims):.3f}")
    if max(sims) > 0.60:
        raise SystemExit("ABORT: demonstration rationales are too similar to each other; "
                         "the model could copy a shared template, which is what invalidated "
                         "the CoT arm of experiment 20.")
    print("  -> heterogeneous, no shared template to copy\n")
    write_json(run_dir / "21_demonstrations.json", demos)
    rat_conf = [d["rationale_conf"] for d in demos]
    rat_noconf = [d["rationale_noconf"] for d in demos]

    demo_ids = {d["question_id"] for d in demos}
    eval_traces = [t for t in traces if t["question_id"] not in demo_ids]
    if args.limit_questions:
        eval_traces = eval_traces[:args.limit_questions]
    print(f"evaluating {len(eval_traces)} held-out questions x {len(conds)} conditions")

    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    csv_path = run_dir / "21_cot.csv"; jl = run_dir / "21_cot.jsonl"
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
            conf = conf_of(hops); shuf = conf[rng.permutation(len(conf))]
            gold_final = str(t["gold_final"])
            aliases = list(t.get("answer_aliases", []) or [])
            print(f"\n=== q{qn}/{len(eval_traces)} {t['question_id']} ===", flush=True)

            for cond in conds:
                if (str(t["question_id"]), cond) in done:
                    continue
                use_conf = cond in ("cot_conf", "cot_conf_shuffled")
                c = shuf if cond == "cot_conf_shuffled" else conf
                p = render(tok, build_messages(cond, ctx, str(t["main_question"]),
                                               qs, ans, c, demos))
                r = batched_generate_with_confidence(
                    model, tok, device, [p], batch_size=1,
                    max_new_tokens=args.max_new_tokens, progress_label=f"q{qn}-{cond}")[0]
                raw = str(r.get("raw_answer", ""))
                pred = parse_answer(raw)
                lp = float("nan")
                if args.score_logprob:
                    score_prompt = (p + raw.rsplit("ANSWER:", 1)[0] + "ANSWER: "
                                    if "ANSWER:" in raw else p)
                    lp = float(batched_target_logprob(
                        model, tok, device, [score_prompt], [gold_final], batch_size=1,
                        progress_label=f"q{qn}-{cond}-lp")[0]["mean_logprob"])
                label, f1, _ = grade_answer(pred, gold_final, aliases)
                ns = named_step(raw)
                true_low = int(np.argmin(c)) + 1
                rows.append({
                    "question_id": str(t["question_id"]), "n_hops": int(t["n_hops"]),
                    "condition": cond, "final_pred": pred, "raw_output": raw[:600],
                    "gold_final_logprob": lp, "final_label": label,
                    "final_correct": bool(label == "correct"), "final_f1": float(f1),
                    "gold_final": gold_final,
                    "confidences": ",".join(f"{x:.0f}" for x in c),
                    "named_step": ns, "true_lowest_step": true_low,
                    "named_matches_lowest": (None if ns is None else bool(ns == true_low)),
                    "parrot_score": parrot_score(raw, rat_conf if use_conf else rat_noconf),
                    "rationale_chars": len(raw.split("ANSWER:")[0].strip()),
                })
                sink.write(json.dumps(rows[-1], ensure_ascii=False) + "\n"); sink.flush()
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

    validity = {}
    for c, g in df.groupby("condition"):
        named = g[g.named_step.notna()]
        chance = float((1.0 / g.n_hops).mean())
        validity[c] = {
            "mean_parrot_score": float(g.parrot_score.mean()),
            "frac_high_parrot_over_0.6": float((g.parrot_score > 0.6).mean()),
            "mean_rationale_chars": float(g.rationale_chars.mean()),
            "frac_naming_a_step": float(len(named) / len(g)),
            "named_matches_true_lowest": (float(named.named_matches_lowest.mean())
                                          if len(named) else None),
            "chance_rate_for_naming": chance,
        }

    summary = {
        "environment": environment_metadata(model_name, device, dtype),
        "supersedes": "CoT arm of 20_icl_variants_claude.py (templated rationales)",
        "n_eval_questions": int(df.question_id.nunique()),
        "wall_clock_min": (time.perf_counter() - t0) / 60.0,
        "n_needs_review": int((df.final_label == "needs_review").sum()),
        "demo_rationale_pairwise_similarity_max": float(max(sims)),
        "demo_paragraphs_per_demonstration": int(args.demo_paragraphs),
        "logprob_scored": bool(args.score_logprob),
        "accuracy_by_condition": {c: float(ok[c].mean()) for c in ok.columns},
        "validity_checks": validity,
        "validity_reading": (
            "mean_parrot_score near 1 means the model is copying demonstration wording and "
            "the arm is invalid, which is what happened in experiment 20. "
            "named_matches_true_lowest above chance_rate_for_naming means the model is "
            "actually READING the confidence values, independently of whether doing so helps."
        ),
        "key_contrasts": {
            "cot_conf_vs_cot_noconf": paired("cot_conf", "cot_noconf"),
            "cot_conf_vs_cot_conf_shuffled": paired("cot_conf", "cot_conf_shuffled"),
            "cot_noconf_vs_cot_zeroshot": paired("cot_noconf", "cot_zeroshot"),
        },
    }
    write_json(run_dir / "21_summary.json", summary)
    print("\n============ CoT ICL (rebuilt) ============")
    print("accuracy:", json.dumps(summary["accuracy_by_condition"], indent=1))
    print("validity:", json.dumps(validity, indent=1))
    print("contrasts:", json.dumps(summary["key_contrasts"], indent=1))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
