#!/usr/bin/env python3
from __future__ import annotations

"""
11b_sequential_confidence_intervention_claude.py

Causal step-level confidence intervention on the Test 06 sequential traces.

SCIENTIFIC QUESTION
-------------------
If an intermediate answer is held EXACTLY fixed, does changing only the
confidence assigned to that answer causally alter the reasoning that follows it?

WHAT THIS SCRIPT DOES (per non-terminal hop Hi of each Test 06 trace)
---------------------------------------------------------------------
  1. Answers for hops H1..Hi are pinned to the Test 06 model predictions.
  2. Verbalized step confidence from Test 06 is normalized within each trace
     to [0, 1] (default: min-max; see NORMALIZATION below).
  3. A confidence-aware downstream prompt is built. It lists, for every
     completed step: SUB-QUESTION, ANSWER, CONFIDENCE, LOG INDEX.
  4. BASELINE arm: hop Hi carries its ORIGINAL normalized confidence.
  5. INTERVENTION arms: ONLY the confidence number for Hi changes. Answers and
     every other input field are byte-identical.
  6. Hops H(i+1)..Hn are regenerated sequentially, then the final answer.
  7. The gold final answer is scored by mean token log-probability.

The intervened value for Hi persists in every downstream prompt in which Hi
appears (including the final prompt). Every non-target confidence is pinned at
its ORIGINAL normalized value and downstream confidence is NEVER re-elicited,
so this is a strict one-coordinate intervention.

INTERVENTION REGIMES
--------------------
  strong    Ci = 0.10  vs  Ci = 0.90
  local     Ci = clip(C0 - 0.10, 0, 1)  vs  clip(C0 + 0.10, 0, 1)
  sham      Ci = C0 exactly (CONTROL: prompt is byte-identical to baseline;
            any downstream difference is decoding nondeterminism, not signal)
  placebo   LOG INDEX for Hi = 0.10 vs 0.90, confidence pinned at C0
            (CONTROL: a length-matched, format-matched, semantically inert
            number in the same structural slot. See CONFOUNDS below.)

NORMALIZATION (--normalization)
-------------------------------
  minmax  (default, as specified)  c_norm = (c - lo) / (hi - lo) within trace.
          Ties map to identical values (no tie-breaking is applied).
          FLAT trace (hi - lo <= 1e-12) or a trace with no parseable
          confidence: every hop is assigned FLAT_VALUE = 0.50 and the trace is
          flagged `trace_is_flat` / `trace_norm_status`. Flat traces are NOT
          dropped, but every summary is reported both including and excluding
          them, because for those traces the "original" confidence is a
          convention, not a measurement.
  raw100  c_norm = c / 100, no within-trace normalization. Provided as a
          robustness check against min-max artifacts; NOT the default.

Within-trace min-max is a MONOTONE RESCALING FOR PROMPT CONSTRUCTION. It is not
calibration, it does not make the numbers probabilities, and averaging these
values does not produce a calibrated metric.

CONFOUNDS AND CONTROLS (read before interpreting results)
---------------------------------------------------------
  C1 PROTOCOL, NOT NATIVE BEHAVIOUR. Test 06 answered each hop with documents +
     sub-question only. This script adds a structured confidence-aware state
     block. Any effect measured here is an effect of THIS PROMPTING PROTOCOL.
     It is not evidence that the unmodified model natively consumes confidence.
     The baseline arm is therefore RE-RUN under the same protocol; Test 06
     numbers are never used as the paired baseline. `protocol_drift_*` reports
     how far the protocol baseline moved from the Test 06 trajectory.
  C2 DECODING NOISE. Without a null arm, any downstream change could be
     float16/MPS nondeterminism accumulating over a regenerated chain. The
     `sham` arm replays the pipeline through a byte-identical prompt and is the
     empirical noise floor. Non-zero sham change rates invalidate effects of
     comparable size.
  C3 NUMBER-PERTURBATION SENSITIVITY (the confound that actually blocks a
     causal-about-CONFIDENCE reading). A small LM may react to ANY digit
     changing in a structured prompt slot, not to the semantics of confidence.
     The `placebo` arm perturbs a same-slot, same-format, same-magnitude,
     explicitly-inert LOG INDEX field 0.10 -> 0.90 while confidence is held
     fixed. If placebo effects match strong-confidence effects, the result is
     prompt sensitivity, not confidence sensitivity. This is the least invasive
     control that separates the two, so it is ON by default.
  C4 PROMPT LENGTH. All numbers render as fixed-width "%.2f", so the perturbed
     span is always 4 characters. Token counts of baseline vs intervention
     prompts are compared and any mismatch is recorded in the audit stream.
  C5 STALE DOWNSTREAM CONFIDENCE. A regenerated hop keeps its ORIGINAL
     normalized confidence even when its answer changed. This is required for a
     one-coordinate intervention but means downstream confidence metadata can
     describe an answer that no longer exists. `n_stale_conf_hops` counts it.
  C6 MIN-MAX MANUFACTURES EXTREMES. Every non-flat trace is forced to contain a
     0.00 and a 1.00. Local perturbation at those hops clips to a zero-size
     step; such arms are flagged `is_degenerate`, excluded from per-unit
     statistics, and reported separately (they behave as extra sham controls).
  C7 COARSE, TIED CONFIDENCE. The Test 06 pilot uses few distinct verbalized
     values with heavy ties. The raw distribution is reported in the summary.
  C8 MEDIATION. With a confidence-aware FINAL prompt, confidence reaches the
     final answer both directly and through changed intermediate answers. The
     gold final answer is therefore scored TWICE per arm: once under the
     confidence-aware final prompt and once under the ORIGINAL confidence-free
     `common.final_prompt`. The confidence-free score isolates the path that
     runs through changed answers only. When no answer changed, the
     confidence-free score must equal baseline exactly; this is asserted and
     doubles as an independent determinism check.

PROMPT-IDENTITY VERIFICATION
----------------------------
For the first regenerated hop (H(i+1)) the baseline and intervention prompts
are verified to differ ONLY in the intervened number, three independent ways:
  (a) both prompts are re-rendered with the target field replaced by a
      placeholder token and asserted byte-identical;
  (b) the two real prompts are required to differ in exactly ONE contiguous
      character span, and reconstruction across that span is asserted;
  (c) tokenized lengths are compared.
Failures raise unless --audit-soft-fail is passed; every check is logged to
11b_prompt_audit.jsonl.

This script only READS existing artifacts. It writes exclusively into its own
--run-dir and modifies no existing script or output.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    batched_generate_with_confidence,
    batched_target_logprob,
    context_block,
    environment_metadata,
    final_prompt,
    grade_answer,
    load_musique,
    load_subject_model,
    normalize,
    read_jsonl,
    resolve_refs,
    seed_everything,
    write_json,
)

REF_RE = re.compile(r"#(\d+)")

CONF_PLACEHOLDER = "<<<CONF_TARGET_PLACEHOLDER>>>"
PLACEBO_PLACEHOLDER = "<<<LOGIDX_TARGET_PLACEHOLDER>>>"

FLAT_VALUE = 0.50          # convention for flat / unparseable traces
PLACEBO_BASELINE = 0.50    # constant inert value on every step in every arm

SYSTEM_CONF_HOP = (
    "You are solving a document-grounded multi-hop QA benchmark. "
    "You are shown the reasoning steps already completed. Each completed step lists its "
    "SUB-QUESTION, its fixed ANSWER, a CONFIDENCE value between 0.00 and 1.00 stating how "
    "reliable that answer is (1.00 means fully reliable, 0.00 means unreliable), and a "
    "LOG INDEX, which is an internal bookkeeping number that carries no information about "
    "correctness. Take the stated CONFIDENCE of each prior answer into account: rely on "
    "high-confidence prior answers, and treat low-confidence prior answers with caution, "
    "re-checking them against the documents. "
    "Use only the supplied documents and this reasoning state. "
    "Return only the shortest factual answer span. Do not explain."
)

SYSTEM_CONF_FINAL = (
    "Use only the supplied documents and the provided structured reasoning state. "
    "Each completed step lists its SUB-QUESTION, its fixed ANSWER, a CONFIDENCE value "
    "between 0.00 and 1.00 stating how reliable that answer is (1.00 means fully reliable, "
    "0.00 means unreliable), and a LOG INDEX, which is an internal bookkeeping number that "
    "carries no information about correctness. Take the stated CONFIDENCE of each step into "
    "account when producing the final answer. "
    "Return only the shortest factual answer span to the main question. Do not explain."
)


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def fmt_num(x: float) -> str:
    """Fixed-width rendering so every perturbed span is exactly 4 characters."""
    return f"{float(x):.2f}"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def apply_chat_template(tok, messages) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def single_diff_span(a: str, b: str):
    """
    Return the unique contiguous differing span between a and b, or None if
    equal. Reconstruction is verified by the caller: if exactly one contiguous
    region differs, splicing b_mid into a must reproduce b exactly.
    """
    if a == b:
        return None
    n = min(len(a), len(b))
    p = 0
    while p < n and a[p] == b[p]:
        p += 1
    s = 0
    while s < (n - p) and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    return {
        "prefix_len": p,
        "suffix_len": s,
        "a_mid": a[p:len(a) - s],
        "b_mid": b[p:len(b) - s],
    }


def dependency_parents(hops: list[dict]) -> list[set[int]]:
    parents: list[set[int]] = []
    for j, h in enumerate(hops):
        refs = set()
        for m in REF_RE.finditer(str(h["template_question"])):
            p = int(m.group(1)) - 1
            if 0 <= p < j:
                refs.add(p)
        parents.append(refs)
    return parents


def descendants_of(node: int, parents: list[set[int]]) -> set[int]:
    desc: set[int] = set()
    for j in range(node + 1, len(parents)):
        if node in parents[j] or any(p in desc for p in parents[j]):
            desc.add(j)
    return desc


def same_answer(a: str, b: str) -> bool:
    """Normalized comparison so whitespace/punctuation noise is not a 'change'."""
    return normalize(str(a)) == normalize(str(b))


# --------------------------------------------------------------------------
# within-trace confidence normalization
# --------------------------------------------------------------------------

def normalize_trace_confidence(hops: list[dict], mode: str) -> dict:
    """
    Map Test 06 verbalized confidence (0-100) to [0, 1] within one trace.

    Returns dict with per-hop normalized values plus explicit status flags.
    Status values:
      ok           at least two distinct finite values; min-max applied
      flat         all finite values identical -> every hop set to FLAT_VALUE
      all_missing  no parseable confidence    -> every hop set to FLAT_VALUE
      raw100       --normalization raw100 was used (no within-trace rescaling)
    """
    raw = []
    for h in hops:
        v = h.get("verbal_confidence", None)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        raw.append(v if np.isfinite(v) else np.nan)

    finite = [v for v in raw if np.isfinite(v)]
    n_missing = len(raw) - len(finite)
    counts = Counter(finite)
    n_tied = sum(c for c in counts.values() if c > 1)

    if mode == "raw100":
        norm = [(v / 100.0 if np.isfinite(v) else FLAT_VALUE) for v in raw]
        status = "raw100"
        is_flat = bool(len(set(finite)) <= 1)
    elif not finite:
        norm = [FLAT_VALUE] * len(raw)
        status = "all_missing"
        is_flat = True
    else:
        lo, hi = float(min(finite)), float(max(finite))
        if hi - lo <= 1e-12:
            norm = [FLAT_VALUE] * len(raw)
            status = "flat"
            is_flat = True
        else:
            norm = [
                ((v - lo) / (hi - lo)) if np.isfinite(v) else FLAT_VALUE
                for v in raw
            ]
            status = "ok"
            is_flat = False

    norm = [float(min(1.0, max(0.0, x))) for x in norm]
    return {
        "raw": raw,
        "norm": norm,
        "status": status,
        "is_flat": bool(is_flat),
        "n_missing": int(n_missing),
        "n_distinct_raw": int(len(set(finite))),
        "n_tied_raw": int(n_tied),
        "raw_min": float(min(finite)) if finite else None,
        "raw_max": float(max(finite)) if finite else None,
    }


# --------------------------------------------------------------------------
# confidence-aware prompt construction
# --------------------------------------------------------------------------

def state_block(
    steps: list[dict],
    placeholder_hop: int | None = None,
    placeholder_field: str | None = None,
) -> str:
    """
    steps: [{hop, question, answer, conf, logidx}, ...] in hop order.

    placeholder_field in {None, 'conf', 'logidx'}: when set, the named field of
    hop `placeholder_hop` renders as an opaque placeholder token instead of a
    number. Rendering both arms with the placeholder and asserting byte
    equality is the airtight proof that nothing but that number differs.
    """
    lines = []
    for s in steps:
        if placeholder_field == "conf" and s["hop"] == placeholder_hop:
            conf = CONF_PLACEHOLDER
        else:
            conf = fmt_num(s["conf"])
        if placeholder_field == "logidx" and s["hop"] == placeholder_hop:
            logidx = PLACEBO_PLACEHOLDER
        else:
            logidx = fmt_num(s["logidx"])
        lines.append(
            f"STEP {s['hop']}\n"
            f"  SUB-QUESTION: {s['question']}\n"
            f"  ANSWER: {s['answer']}\n"
            f"  CONFIDENCE: {conf}\n"
            f"  LOG INDEX: {logidx}"
        )
    return "\n".join(lines)


def conf_hop_prompt(tok, ctx, steps, subq, placeholder_hop=None, placeholder_field=None) -> str:
    user = (
        f"DOCUMENTS:\n{ctx}\n\n"
        f"COMPLETED REASONING STATE:\n"
        f"{state_block(steps, placeholder_hop, placeholder_field)}\n\n"
        f"NEXT SUB-QUESTION:\n{subq}\n\n"
        "SHORT ANSWER:"
    )
    return apply_chat_template(
        tok,
        [{"role": "system", "content": SYSTEM_CONF_HOP},
         {"role": "user", "content": user}],
    )


def conf_final_prompt(tok, ctx, steps, main_q, placeholder_hop=None, placeholder_field=None) -> str:
    user = (
        f"DOCUMENTS:\n{ctx}\n\n"
        f"STRUCTURED REASONING STATE:\n"
        f"{state_block(steps, placeholder_hop, placeholder_field)}\n\n"
        f"MAIN QUESTION:\n{main_q}\n\n"
        "SHORT ANSWER:"
    )
    return apply_chat_template(
        tok,
        [{"role": "system", "content": SYSTEM_CONF_FINAL},
         {"role": "user", "content": user}],
    )


# --------------------------------------------------------------------------
# model helpers (batch size 1 throughout for MPS stability)
# --------------------------------------------------------------------------

def gen_one(model, tok, device, prompt, max_new_tokens, label):
    return batched_generate_with_confidence(
        model, tok, device, [prompt],
        batch_size=1,
        max_new_tokens=max_new_tokens,
        progress_label=label,
    )[0]


def score_one(model, tok, device, prompt, target, label) -> float:
    out = batched_target_logprob(
        model, tok, device, [prompt], [target],
        batch_size=1, progress_label=label,
    )[0]
    x = float(out["mean_logprob"])
    if not np.isfinite(x):
        raise RuntimeError(f"Non-finite target logprob in {label}")
    return x


# --------------------------------------------------------------------------
# one arm = one full downstream regeneration under one confidence assignment
# --------------------------------------------------------------------------

def run_arm(
    model, tok, device, args,
    ctx, main_q, gold_final, aliases,
    hops, target_idx,
    conf_vec, logidx_vec,
    arm_name,
    capture_placeholder: str | None = None,
):
    """
    Answers for hops 0..target_idx are pinned to the Test 06 predictions.
    Hops target_idx+1..n-1 are regenerated sequentially, then the final answer.

    conf_vec / logidx_vec are the FULL per-hop metadata vectors used in every
    downstream prompt (already containing the arm's intervened value).

    capture_placeholder in {None,'conf','logidx'} additionally renders the FIRST
    regenerated hop's prompt with the target field masked, for identity proof.
    """
    n = len(hops)
    answers = [str(h["pred"]) for h in hops]
    questions = [str(h["model_resolved_question"]) for h in hops]

    regen = []
    first_prompt = None
    first_prompt_masked = None

    for j in range(target_idx + 1, n):
        resolved_q = resolve_refs(str(hops[j]["template_question"]), answers[:j])
        questions[j] = resolved_q

        steps = [{
            "hop": k + 1,
            "question": questions[k],
            "answer": answers[k],
            "conf": conf_vec[k],
            "logidx": logidx_vec[k],
        } for k in range(j)]

        p = conf_hop_prompt(tok, ctx, steps, resolved_q)
        if j == target_idx + 1:
            first_prompt = p
            if capture_placeholder is not None:
                first_prompt_masked = conf_hop_prompt(
                    tok, ctx, steps, resolved_q,
                    placeholder_hop=target_idx + 1,
                    placeholder_field=capture_placeholder,
                )

        r = gen_one(
            model, tok, device, p, args.max_new_tokens,
            f"{arm_name}-h{target_idx+1}-regen-h{j+1}",
        )
        answers[j] = str(r["answer"])
        regen.append({
            "hop": j + 1,
            "question": resolved_q,
            "answer": answers[j],
            "raw_answer": str(r.get("raw_answer", "")),
            "mean_logprob": float(r["mean_logprob"]),
            "entropy": float(r["entropy"]),
            "generated_tokens": int(r["generated_tokens"]),
            "prompt_tokens": int(r["prompt_tokens"]),
            "prompt_sha": sha(p),
        })

    all_steps = [{
        "hop": k + 1,
        "question": questions[k],
        "answer": answers[k],
        "conf": conf_vec[k],
        "logidx": logidx_vec[k],
    } for k in range(n)]

    fp_conf = conf_final_prompt(tok, ctx, all_steps, main_q)
    fr = gen_one(model, tok, device, fp_conf, args.max_new_tokens,
                 f"{arm_name}-h{target_idx+1}-final")
    final_pred = str(fr["answer"])
    final_label, final_f1, _ = grade_answer(final_pred, gold_final, aliases)

    lp_conf = score_one(model, tok, device, fp_conf, gold_final,
                        f"{arm_name}-h{target_idx+1}-goldlp-confaware")

    # C8: confidence-free rescoring isolates the answer-mediated path and acts
    # as an independent determinism probe when no answer changed.
    fp_free = final_prompt(tok, ctx, main_q, [(questions[k], answers[k]) for k in range(n)])
    lp_free = score_one(model, tok, device, fp_free, gold_final,
                        f"{arm_name}-h{target_idx+1}-goldlp-conffree")

    return {
        "arm": arm_name,
        "answers": answers,
        "questions": questions,
        "regenerated": regen,
        "final_pred": final_pred,
        "final_raw": str(fr.get("raw_answer", "")),
        "final_label": final_label,
        "final_f1": float(final_f1),
        "final_gold_logprob_confaware": lp_conf,
        "final_gold_logprob_conffree": lp_free,
        "final_prompt_sha_confaware": sha(fp_conf),
        "final_prompt_sha_conffree": sha(fp_free),
        "first_regen_prompt": first_prompt,
        "first_regen_prompt_masked": first_prompt_masked,
        "first_regen_prompt_sha": sha(first_prompt) if first_prompt else None,
    }


# --------------------------------------------------------------------------
# prompt-identity audit
# --------------------------------------------------------------------------

def audit_prompt_pair(tok, base_arm, int_arm, field, expect_base, expect_int, meta, args, audit_sink):
    """
    Verify baseline vs intervention first-regenerated-hop prompts differ ONLY in
    the intervened number. Three independent checks; see module docstring.
    """
    rec = dict(meta)
    rec.update({"field": field, "checks": {}, "ok": True, "notes": []})

    a = base_arm["first_regen_prompt"]
    b = int_arm["first_regen_prompt"]
    ma = base_arm["first_regen_prompt_masked"]
    mb = int_arm["first_regen_prompt_masked"]

    # (a) masked-render byte equality
    ok_mask = (ma is not None and mb is not None and ma == mb)
    rec["checks"]["masked_render_identical"] = bool(ok_mask)
    if not ok_mask:
        rec["ok"] = False
        rec["notes"].append("masked renders differ: something other than the target number changed")

    # (b) exactly one contiguous differing span, with reconstruction
    d = single_diff_span(a, b)
    if d is None:
        rec["checks"]["single_contiguous_diff"] = True
        rec["checks"]["diff_is_empty"] = True
        rec["diff"] = None
        if abs(float(expect_int) - float(expect_base)) > 1e-12:
            rec["ok"] = False
            rec["notes"].append("prompts identical though the intervened value differs")
    else:
        recon = a[:d["prefix_len"]] + d["b_mid"] + a[len(a) - d["suffix_len"]:]
        ok_span = (recon == b)
        rec["checks"]["single_contiguous_diff"] = bool(ok_span)
        rec["checks"]["diff_is_empty"] = False
        rec["diff"] = {
            "baseline_span": d["a_mid"],
            "intervention_span": d["b_mid"],
            "prefix_len": d["prefix_len"],
            "suffix_len": d["suffix_len"],
        }
        if not ok_span:
            rec["ok"] = False
            rec["notes"].append("more than one region of the prompt differs")
        # the differing span must be a substring of the two rendered numbers
        sb, si = fmt_num(expect_base), fmt_num(expect_int)
        if d["a_mid"] not in sb or d["b_mid"] not in si:
            rec["ok"] = False
            rec["notes"].append(
                f"diff span {d['a_mid']!r}->{d['b_mid']!r} not explained by {sb}->{si}"
            )

    # (c) token-length parity (C4)
    na = len(tok.encode(a, add_special_tokens=False))
    nb = len(tok.encode(b, add_special_tokens=False))
    rec["checks"]["token_len_equal"] = bool(na == nb)
    rec["baseline_prompt_tokens"] = int(na)
    rec["intervention_prompt_tokens"] = int(nb)
    if na != nb:
        rec["ok"] = False
        rec["notes"].append(f"token length changed {na} -> {nb}")

    # pinned answers must be identical through the target hop
    ti = int(meta["target_hop_idx"])
    pinned_ok = base_arm["answers"][:ti + 1] == int_arm["answers"][:ti + 1]
    rec["checks"]["pinned_answers_identical"] = bool(pinned_ok)
    if not pinned_ok:
        rec["ok"] = False
        rec["notes"].append("pinned answers through the target hop are not identical")

    rec["baseline_value"] = float(expect_base)
    rec["intervention_value"] = float(expect_int)
    rec["baseline_prompt_sha"] = base_arm["first_regen_prompt_sha"]
    rec["intervention_prompt_sha"] = int_arm["first_regen_prompt_sha"]

    audit_sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    audit_sink.flush()

    if not rec["ok"]:
        msg = f"PROMPT AUDIT FAILED {meta} :: {rec['notes']}"
        if args.audit_soft_fail:
            print("WARNING: " + msg, file=sys.stderr, flush=True)
        else:
            raise AssertionError(msg)
    return rec


# --------------------------------------------------------------------------
# summarization
# --------------------------------------------------------------------------

def _rate(s):
    s = pd.Series(s).dropna()
    return float(s.mean()) if len(s) else None


def _mean(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else None


def summarize_block(g: pd.DataFrame) -> dict:
    if not len(g):
        return {"n": 0}
    return {
        "n": int(len(g)),
        "n_degenerate": int(g["is_degenerate"].sum()),
        "immediate_hop_change_rate": _rate(g["immediate_hop_changed"]),
        "any_descendant_change_rate": _rate(g["any_descendant_changed"]),
        "mean_n_changed_descendants": _mean(g["n_changed_descendants"]),
        "mean_first_changed_hop": _mean(g["first_changed_hop"]),
        "any_dep_descendant_change_rate": _rate(g["any_dep_descendant_changed"]),
        "final_answer_change_rate": _rate(g["final_changed"]),
        "final_accuracy": _rate(g["final_correct"]),
        "n_final_needs_review": int((g["final_label"] == "needs_review").sum()),
        "mean_d_logprob_confaware": _mean(g["d_logprob_confaware"]),
        "median_d_logprob_confaware": (
            float(pd.to_numeric(g["d_logprob_confaware"], errors="coerce").median())
            if len(g) else None
        ),
        "mean_abs_d_logprob_confaware": _mean(g["d_logprob_confaware"].abs()),
        "mean_d_logprob_conffree": _mean(g["d_logprob_conffree"]),
        "mean_d_logprob_per_unit_conf": _mean(g["d_logprob_per_unit_conf"]),
        "mean_conf_delta": _mean(g["conf_delta"]),
    }


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    non_base = df[df["arm"] != "baseline"].copy()

    out["by_arm"] = {a: summarize_block(g) for a, g in non_base.groupby("arm")}
    out["by_regime"] = {r: summarize_block(g) for r, g in non_base.groupby("regime")}

    nf = non_base[~non_base["trace_is_flat"].astype(bool)]
    out["by_arm_excluding_flat_traces"] = {a: summarize_block(g) for a, g in nf.groupby("arm")}

    nd = non_base[~non_base["is_degenerate"].astype(bool)]
    out["by_arm_excluding_degenerate"] = {a: summarize_block(g) for a, g in nd.groupby("arm")}

    out["by_arm_and_target_hop_position"] = {
        f"{a}|hop{int(h)}": summarize_block(g)
        for (a, h), g in non_base.groupby(["arm", "target_hop"])
    }
    out["by_arm_and_target_hop_benchmark_label"] = {
        f"{a}|{lab}": summarize_block(g)
        for (a, lab), g in non_base.groupby(["arm", "target_hop_label"])
    }
    return out


def paired_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Direct high-vs-low comparison for the same (question, target hop)."""
    pairs = [
        ("strong", "strong_low", "strong_high"),
        ("local", "local_down", "local_up"),
        ("placebo", "placebo_low", "placebo_high"),
    ]
    key = ["question_id", "target_hop"]
    rows = []
    for regime, lo_arm, hi_arm in pairs:
        lo = df[df["arm"] == lo_arm].set_index(key)
        hi = df[df["arm"] == hi_arm].set_index(key)
        base = df[df["arm"] == "baseline"].set_index(key)
        common = lo.index.intersection(hi.index)
        for k in common:
            l, h = lo.loc[k], hi.loc[k]
            if isinstance(l, pd.DataFrame):
                l = l.iloc[0]
            if isinstance(h, pd.DataFrame):
                h = h.iloc[0]
            b = base.loc[k] if k in base.index else None
            if isinstance(b, pd.DataFrame):
                b = b.iloc[0]
            dc = float(h["conf_arm_norm"]) - float(l["conf_arm_norm"]) if regime != "placebo" else \
                 float(h["placebo_arm"]) - float(l["placebo_arm"])
            dlp = float(h["final_gold_logprob_confaware"]) - float(l["final_gold_logprob_confaware"])
            rows.append({
                "question_id": k[0],
                "target_hop": int(k[1]),
                "regime": regime,
                "n_hops": int(l["n_hops"]),
                "target_hop_label": l["target_hop_label"],
                "trace_is_flat": bool(l["trace_is_flat"]),
                "low_value": float(l["conf_arm_norm"]) if regime != "placebo" else float(l["placebo_arm"]),
                "high_value": float(h["conf_arm_norm"]) if regime != "placebo" else float(h["placebo_arm"]),
                "value_gap": dc,
                "degenerate_pair": bool(abs(dc) < 1e-12),
                "low_final_pred": l["final_pred"],
                "high_final_pred": h["final_pred"],
                "baseline_final_pred": b["final_pred"] if b is not None else None,
                "high_vs_low_final_changed": (not same_answer(l["final_pred"], h["final_pred"])),
                "low_final_correct": bool(l["final_correct"]),
                "high_final_correct": bool(h["final_correct"]),
                "low_gold_logprob": float(l["final_gold_logprob_confaware"]),
                "high_gold_logprob": float(h["final_gold_logprob_confaware"]),
                "high_minus_low_gold_logprob": dlp,
                "high_minus_low_per_unit": (dlp / dc) if abs(dc) > 1e-12 else None,
                "low_first_changed_hop": l["first_changed_hop"],
                "high_first_changed_hop": h["first_changed_hop"],
                "low_n_changed_descendants": int(l["n_changed_descendants"]),
                "high_n_changed_descendants": int(h["n_changed_descendants"]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Step-level confidence intervention on Test 06 sequential traces."
    )
    ap.add_argument("--trace-file",
                    default="outputs/qwen17b_sequential_seed42/06_sequential_trace.jsonl")
    ap.add_argument("--source-summary",
                    default="outputs/qwen17b_sequential_seed42/06_summary.json")
    ap.add_argument("--run-dir",
                    default="outputs/qwen17b_conf_intervention_claude_seed42",
                    help="Separate output directory. Existing outputs are never touched.")
    ap.add_argument("--model", default=None, help="Default: the model recorded in 06_summary.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--limit-questions", type=int, default=0,
                    help="Smoke test: use only the first N traces. 0 = all.")
    ap.add_argument("--normalization", choices=["minmax", "raw100"], default="minmax")
    ap.add_argument("--strong-low", type=float, default=0.10)
    ap.add_argument("--strong-high", type=float, default=0.90)
    ap.add_argument("--local-delta", type=float, default=0.10)
    ap.add_argument("--regimes", default="strong,local,sham,placebo",
                    help="Comma-separated subset of strong,local,sham,placebo.")
    ap.add_argument("--audit-soft-fail", action="store_true",
                    help="Log prompt-identity violations instead of raising.")
    ap.add_argument("--dump-prompts", action="store_true",
                    help="Persist full first-regenerated-hop prompts (large).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (question_id, target_hop) cases already present in the JSONL.")
    args = ap.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    unknown = set(regimes) - {"strong", "local", "sham", "placebo"}
    if unknown:
        raise SystemExit(f"Unknown regimes: {sorted(unknown)}")

    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    traces = read_jsonl(args.trace_file)
    if args.limit_questions:
        traces = traces[:args.limit_questions]

    src_summary = json.loads(Path(args.source_summary).read_text())
    model_name = args.model or src_summary["environment"]["model"]

    # ---- raw confidence distribution (C7), computed before any model load ----
    raw_all, flat_traces, norm_table = [], 0, []
    for tr in traces:
        hops = sorted(tr["hops"], key=lambda x: int(x["hop_idx"]))
        nz = normalize_trace_confidence(hops, args.normalization)
        flat_traces += int(nz["is_flat"])
        for k, h in enumerate(hops):
            raw_all.append(nz["raw"][k])
            norm_table.append({
                "question_id": tr["question_id"],
                "dataset_idx": int(tr["dataset_idx"]),
                "hop": k + 1,
                "n_hops": len(hops),
                "raw_verbal_confidence": nz["raw"][k],
                "conf_norm": nz["norm"][k],
                "trace_norm_status": nz["status"],
                "trace_is_flat": nz["is_flat"],
                "trace_n_distinct_raw": nz["n_distinct_raw"],
                "trace_n_tied_raw": nz["n_tied_raw"],
                "hop_label": h.get("label"),
            })
    pd.DataFrame(norm_table).to_csv(run_dir / "11b_normalized_confidence.csv", index=False)

    finite_raw = [v for v in raw_all if v is not None and np.isfinite(v)]
    raw_dist = {
        "n_hops": len(raw_all),
        "n_parsed": len(finite_raw),
        "n_missing": len(raw_all) - len(finite_raw),
        "n_distinct_values": int(len(set(finite_raw))),
        "value_counts": {str(k): int(v) for k, v in sorted(Counter(finite_raw).items())},
        "n_flat_traces": int(flat_traces),
        "n_traces": len(traces),
        "flat_trace_fraction": float(flat_traces / len(traces)) if traces else None,
        "note": (
            "Verbalized confidence in this pilot is coarse and heavily tied. Within-trace "
            "min-max is a monotone rescaling for prompt construction only; it is not "
            "calibration and these values are not probabilities."
        ),
    }
    print("\n---- raw verbalized confidence distribution ----")
    print(json.dumps(raw_dist, indent=2))

    ds = load_musique()
    model, tok, device, dtype = load_subject_model(model_name, args.device, args.dtype)

    rows_path = run_dir / "11b_interventions.jsonl"
    csv_path = run_dir / "11b_interventions.csv"
    audit_path = run_dir / "11b_prompt_audit.jsonl"

    done_keys = set()
    if args.resume and rows_path.exists():
        for r in read_jsonl(rows_path):
            done_keys.add((r["question_id"], int(r["target_hop"])))
        print(f"resume: {len(done_keys)} (question, target hop) cases already complete")

    rows_sink = open(rows_path, "a" if args.resume else "w", encoding="utf-8")
    audit_sink = open(audit_path, "a" if args.resume else "w", encoding="utf-8")

    all_rows = []
    if args.resume and csv_path.exists():
        all_rows = pd.read_csv(csv_path).to_dict(orient="records")

    t_start = time.perf_counter()
    n_cases = 0

    try:
        for qnum, tr in enumerate(traces, start=1):
            hops = sorted(tr["hops"], key=lambda x: int(x["hop_idx"]))
            n = len(hops)
            if n < 2:
                continue

            ex = ds[int(tr["dataset_idx"])]
            ctx = context_block(ex)
            main_q = str(tr["main_question"])
            gold_final = str(tr["gold_final"])
            aliases = list(tr.get("answer_aliases", []) or [])

            nz = normalize_trace_confidence(hops, args.normalization)
            conf0 = list(nz["norm"])
            logidx0 = [PLACEBO_BASELINE] * n
            parents = dependency_parents(hops)
            orig_answers = [str(h["pred"]) for h in hops]

            for i in range(n - 1):          # non-terminal hops only
                key = (str(tr["question_id"]), i + 1)
                if key in done_keys:
                    continue

                c0 = float(conf0[i])
                dep_desc = descendants_of(i, parents)

                # ---- arm definitions -------------------------------------
                arms = []
                if "strong" in regimes:
                    arms.append(("strong_low", "strong", "conf", args.strong_low, PLACEBO_BASELINE))
                    arms.append(("strong_high", "strong", "conf", args.strong_high, PLACEBO_BASELINE))
                if "local" in regimes:
                    arms.append(("local_down", "local", "conf",
                                 float(np.clip(c0 - args.local_delta, 0.0, 1.0)), PLACEBO_BASELINE))
                    arms.append(("local_up", "local", "conf",
                                 float(np.clip(c0 + args.local_delta, 0.0, 1.0)), PLACEBO_BASELINE))
                if "sham" in regimes:
                    arms.append(("sham", "sham", "conf", c0, PLACEBO_BASELINE))
                if "placebo" in regimes:
                    arms.append(("placebo_low", "placebo", "logidx", c0, args.strong_low))
                    arms.append(("placebo_high", "placebo", "logidx", c0, args.strong_high))

                print(
                    f"\n=== q{qnum}/{len(traces)} {tr['question_id']} | target H{i+1}/{n} "
                    f"| C0={c0:.2f} ({nz['status']}) | arms={len(arms)+1} ===",
                    flush=True,
                )

                # ---- BASELINE -------------------------------------------
                base = run_arm(
                    model, tok, device, args, ctx, main_q, gold_final, aliases,
                    hops, i, conf0, logidx0, "baseline", capture_placeholder="conf",
                )

                # baseline masked render for the placebo field, built without
                # any generation so the placebo audit has a counterpart
                base_masked_logidx = None
                if "placebo" in regimes and base["first_regen_prompt"] is not None:
                    j0 = i + 1
                    steps0 = [{
                        "hop": k + 1,
                        "question": base["questions"][k],
                        "answer": base["answers"][k],
                        "conf": conf0[k],
                        "logidx": logidx0[k],
                    } for k in range(j0)]
                    q0 = resolve_refs(str(hops[j0]["template_question"]), base["answers"][:j0])
                    base_masked_logidx = conf_hop_prompt(
                        tok, ctx, steps0, q0,
                        placeholder_hop=i + 1, placeholder_field="logidx",
                    )

                arm_results = {"baseline": base}

                base_row = make_row(
                    tr, hops, i, n, nz, c0, "baseline", "baseline",
                    c0, PLACEBO_BASELINE, base, base, dep_desc, orig_answers, args,
                )
                arm_rows = [base_row]

                # ---- INTERVENTIONS --------------------------------------
                for arm_name, regime, field, cval, pval in arms:
                    conf_vec = list(conf0)
                    logidx_vec = list(logidx0)
                    if field == "conf":
                        conf_vec[i] = float(cval)
                    else:
                        logidx_vec[i] = float(pval)

                    res = run_arm(
                        model, tok, device, args, ctx, main_q, gold_final, aliases,
                        hops, i, conf_vec, logidx_vec, arm_name,
                        capture_placeholder=field,
                    )
                    arm_results[arm_name] = res

                    meta = {
                        "question_id": str(tr["question_id"]),
                        "dataset_idx": int(tr["dataset_idx"]),
                        "target_hop": i + 1,
                        "target_hop_idx": i,
                        "arm": arm_name,
                        "regime": regime,
                    }
                    if field == "conf":
                        audit_prompt_pair(
                            tok, base, res, "CONFIDENCE", c0, cval, meta, args, audit_sink,
                        )
                    else:
                        shim = dict(base)
                        shim["first_regen_prompt_masked"] = base_masked_logidx
                        audit_prompt_pair(
                            tok, shim, res, "LOG INDEX",
                            PLACEBO_BASELINE, pval, meta, args, audit_sink,
                        )

                    arm_rows.append(make_row(
                        tr, hops, i, n, nz, c0, arm_name, regime,
                        cval if field == "conf" else c0,
                        pval if field == "logidx" else PLACEBO_BASELINE,
                        res, base, dep_desc, orig_answers, args,
                    ))

                # ---- persist ---------------------------------------------
                for r, name in zip(arm_rows, ["baseline"] + [a[0] for a in arms]):
                    detail = dict(r)
                    a = arm_results[name]
                    detail["state"] = {
                        "conf_vector": (list(conf0) if name == "baseline"
                                        else _conf_vec_for(conf0, i, r)),
                        "logidx_vector": _logidx_vec_for(logidx0, i, r),
                        "answers": a["answers"],
                        "questions": a["questions"],
                        "regenerated": a["regenerated"],
                        "original_answers": orig_answers,
                        "dependency_descendants": sorted(x + 1 for x in dep_desc),
                    }
                    if args.dump_prompts:
                        detail["first_regen_prompt"] = a["first_regen_prompt"]
                    rows_sink.write(json.dumps(detail, ensure_ascii=False) + "\n")
                rows_sink.flush()

                all_rows.extend(arm_rows)
                pd.DataFrame(all_rows).to_csv(csv_path, index=False)
                n_cases += 1

    finally:
        rows_sink.close()
        audit_sink.close()

    if not all_rows:
        raise SystemExit("No intervention cases were produced.")

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)

    contrasts = paired_contrasts(df)
    contrasts.to_csv(run_dir / "11b_paired_contrasts.csv", index=False)

    # ---- validity checks ------------------------------------------------
    audits = read_jsonl(audit_path) if audit_path.exists() else []
    sham = df[df["arm"] == "sham"]
    base_df = df[df["arm"] == "baseline"]

    validity = {
        "prompt_audit_records": len(audits),
        "prompt_audit_failures": int(sum(1 for a in audits if not a.get("ok", False))),
        "prompt_audit_token_len_mismatches": int(
            sum(1 for a in audits if not a.get("checks", {}).get("token_len_equal", True))
        ),
        "sham_control": {
            "n": int(len(sham)),
            "immediate_hop_change_rate": _rate(sham["immediate_hop_changed"]) if len(sham) else None,
            "any_descendant_change_rate": _rate(sham["any_descendant_changed"]) if len(sham) else None,
            "final_answer_change_rate": _rate(sham["final_changed"]) if len(sham) else None,
            "max_abs_d_logprob_confaware": (
                float(sham["d_logprob_confaware"].abs().max()) if len(sham) else None
            ),
            "interpretation": (
                "Sham replays a byte-identical prompt. Anything non-zero here is decoding "
                "nondeterminism and sets the floor below which confidence effects are noise."
            ),
        },
        "conffree_invariance_violations": int(df["conffree_invariance_violated"].sum()),
        "conffree_invariance_note": (
            "When an arm changed no answer, its confidence-free final prompt is identical to "
            "baseline's, so its gold log-probability must match baseline exactly. Violations "
            "indicate nondeterminism, not an intervention effect."
        ),
        "protocol_drift": {
            "note": (
                "Test 06 answered each hop without a reasoning-state block. The baseline arm "
                "re-runs the confidence-aware protocol, so its regenerated answers may differ "
                "from Test 06. This is protocol drift, not an intervention effect."
            ),
            "mean_fraction_regenerated_hops_differing_from_test06": (
                _mean(base_df["protocol_drift_fraction"]) if len(base_df) else None
            ),
            "mean_n_stale_conf_hops": _mean(base_df["n_stale_conf_hops"]) if len(base_df) else None,
        },
    }

    summary = {
        "script": "11b_sequential_confidence_intervention_claude.py",
        "environment": environment_metadata(model_name, device, dtype),
        "seed": args.seed,
        "source_trace_file": args.trace_file,
        "run_dir": str(run_dir),
        "config": {
            "normalization": args.normalization,
            "flat_trace_value": FLAT_VALUE,
            "placebo_baseline_value": PLACEBO_BASELINE,
            "regimes": regimes,
            "strong_low": args.strong_low,
            "strong_high": args.strong_high,
            "local_delta": args.local_delta,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy, do_sample=False, batch_size=1",
            "limit_questions": args.limit_questions,
        },
        "design": {
            "claim_scope": (
                "This tests a CONFIDENCE-AWARE PROMPTING PROTOCOL. It does not claim the "
                "unmodified model natively consumes step confidence."
            ),
            "intervention": (
                "One coordinate: only the confidence number attached to the target hop changes. "
                "The target hop's answer is never changed. Downstream confidence is never "
                "re-elicited; all non-target confidence is pinned at its original normalized value."
            ),
            "normalization_caveat": (
                "Within-trace min-max is a monotone rescaling, NOT statistical calibration. "
                "It forces a 0.00 and a 1.00 into every non-flat trace, and averaging these "
                "values does not yield a calibrated metric."
            ),
            "controls": [
                "sham: byte-identical prompt replay -> decoding-noise floor",
                "placebo: same-slot, same-format, same-magnitude inert LOG INDEX perturbation "
                "-> separates confidence semantics from generic number sensitivity",
                "confidence-free final rescoring -> isolates the answer-mediated path",
                "three-way prompt-identity audit on the first regenerated hop",
            ],
        },
        "raw_confidence_distribution": raw_dist,
        "counts": {
            "n_traces_used": len(traces),
            "n_target_hop_cases": int(df.groupby(["question_id", "target_hop"]).ngroups),
            "n_arm_rows": int(len(df)),
            "wall_clock_seconds": float(time.perf_counter() - t_start),
        },
        "validity_checks": validity,
        "results": summarize(df),
        "paired_high_vs_low": (
            {
                r: {
                    "n_pairs": int(len(g)),
                    "n_degenerate_pairs": int(g["degenerate_pair"].sum()),
                    "final_answer_differs_rate": _rate(g["high_vs_low_final_changed"]),
                    "mean_high_minus_low_gold_logprob": _mean(g["high_minus_low_gold_logprob"]),
                    "mean_high_minus_low_per_unit": _mean(g["high_minus_low_per_unit"]),
                    "high_final_accuracy": _rate(g["high_final_correct"]),
                    "low_final_accuracy": _rate(g["low_final_correct"]),
                }
                for r, g in contrasts.groupby("regime")
            } if len(contrasts) else {}
        ),
    }
    write_json(run_dir / "11b_summary.json", summary)

    print("\n================ 11b CONFIDENCE INTERVENTION ================")
    print(json.dumps({k: summary[k] for k in
                      ["counts", "validity_checks", "raw_confidence_distribution"]}, indent=2))
    print("\nResults by arm:")
    print(json.dumps(summary["results"]["by_arm"], indent=2))
    print("\nPaired high-vs-low:")
    print(json.dumps(summary["paired_high_vs_low"], indent=2))
    print("\nSaved:")
    for name in ["11b_interventions.csv", "11b_interventions.jsonl",
                 "11b_paired_contrasts.csv", "11b_prompt_audit.jsonl",
                 "11b_normalized_confidence.csv", "11b_summary.json"]:
        print(f"  {run_dir / name}")


def _conf_vec_for(conf0, i, row):
    v = list(conf0)
    v[i] = float(row["conf_arm_norm"])
    return v


def _logidx_vec_for(logidx0, i, row):
    v = list(logidx0)
    v[i] = float(row["placebo_arm"])
    return v


def make_row(tr, hops, i, n, nz, c0, arm_name, regime,
             conf_arm, placebo_arm, res, base, dep_desc, orig_answers, args):
    """Flatten one arm into the analysis row, measured against the paired baseline."""
    changed_hops, changed_dep_hops, stale = [], [], []
    for j in range(i + 1, n):
        if not same_answer(res["answers"][j], base["answers"][j]):
            changed_hops.append(j + 1)
            if j in dep_desc:
                changed_dep_hops.append(j + 1)
        if not same_answer(res["answers"][j], orig_answers[j]):
            stale.append(j + 1)

    n_regen = n - (i + 1)
    protocol_drift = [
        j + 1 for j in range(i + 1, n)
        if not same_answer(base["answers"][j], orig_answers[j])
    ]

    d_conf = float(conf_arm) - float(c0)
    d_lp = float(res["final_gold_logprob_confaware"]) - float(base["final_gold_logprob_confaware"])
    d_lp_free = float(res["final_gold_logprob_conffree"]) - float(base["final_gold_logprob_conffree"])

    answers_identical = all(
        same_answer(res["answers"][j], base["answers"][j]) for j in range(n)
    )
    conffree_violated = bool(answers_identical and abs(d_lp_free) > 1e-4)

    return {
        "question_id": str(tr["question_id"]),
        "dataset_idx": int(tr["dataset_idx"]),
        "n_hops": int(n),
        "target_hop": i + 1,
        "target_hop_idx": i,
        "arm": arm_name,
        "regime": regime,
        "normalization": args.normalization,

        "raw_verbal_confidence": nz["raw"][i],
        "conf_baseline_norm": float(c0),
        "conf_arm_norm": float(conf_arm),
        "conf_delta": d_conf,
        "placebo_baseline": float(PLACEBO_BASELINE),
        "placebo_arm": float(placebo_arm),
        "placebo_delta": float(placebo_arm) - float(PLACEBO_BASELINE),
        "trace_norm_status": nz["status"],
        "trace_is_flat": bool(nz["is_flat"]),
        "trace_n_distinct_raw": int(nz["n_distinct_raw"]),
        "is_degenerate": bool(
            abs(d_conf) < 1e-12 and abs(float(placebo_arm) - PLACEBO_BASELINE) < 1e-12
            and arm_name != "baseline"
        ),

        "target_hop_pred": str(hops[i]["pred"]),
        "target_hop_gold": str(hops[i]["gold"]),
        "target_hop_label": str(hops[i]["label"]),
        "target_hop_benchmark_correct": bool(hops[i]["label"] == "correct"),
        "target_hop_matches_gold_path": bool(hops[i].get("question_matches_gold_path", False)),
        "target_hop_answer_held_fixed": True,

        "n_regenerated": int(n_regen),
        "immediate_hop_changed": bool((i + 2) in changed_hops) if n_regen else None,
        "any_descendant_changed": bool(len(changed_hops) > 0),
        "first_changed_hop": int(min(changed_hops)) if changed_hops else None,
        "n_changed_descendants": int(len(changed_hops)),
        "changed_descendants": ",".join(str(x) for x in changed_hops),
        "n_dependency_descendants": int(len(dep_desc)),
        "any_dep_descendant_changed": bool(len(changed_dep_hops) > 0),
        "first_changed_dep_hop": int(min(changed_dep_hops)) if changed_dep_hops else None,
        "n_changed_dep_descendants": int(len(changed_dep_hops)),

        "final_pred": str(res["final_pred"]),
        "baseline_final_pred": str(base["final_pred"]),
        "final_changed": bool(not same_answer(res["final_pred"], base["final_pred"])),
        "final_changed_exact": bool(str(res["final_pred"]) != str(base["final_pred"])),
        "final_label": str(res["final_label"]),
        "final_correct": bool(res["final_label"] == "correct"),
        "final_f1": float(res["final_f1"]),
        "gold_final": str(tr["gold_final"]),

        "final_gold_logprob_confaware": float(res["final_gold_logprob_confaware"]),
        "final_gold_logprob_conffree": float(res["final_gold_logprob_conffree"]),
        "baseline_gold_logprob_confaware": float(base["final_gold_logprob_confaware"]),
        "baseline_gold_logprob_conffree": float(base["final_gold_logprob_conffree"]),
        "d_logprob_confaware": d_lp,
        "d_logprob_conffree": d_lp_free,
        "d_logprob_per_unit_conf": (d_lp / d_conf) if abs(d_conf) > 1e-12 else None,
        "conffree_invariance_violated": conffree_violated,

        "n_stale_conf_hops": int(len(stale)),
        "stale_conf_hops": ",".join(str(x) for x in stale),
        "protocol_drift_n": int(len(protocol_drift)),
        "protocol_drift_fraction": (len(protocol_drift) / n_regen) if n_regen else None,

        "test06_baseline_final_pred": str(tr["baseline_final_pred"]),
        "test06_baseline_final_label": str(tr["baseline_final_label"]),
        "test06_baseline_final_gold_logprob": float(tr["baseline_final_gold_logprob"]),

        "first_regen_prompt_sha": res["first_regen_prompt_sha"],
        "final_prompt_sha_confaware": res["final_prompt_sha_confaware"],
        "final_prompt_sha_conffree": res["final_prompt_sha_conffree"],
    }


if __name__ == "__main__":
    main()
