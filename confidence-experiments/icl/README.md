# In-Context Learning: can demonstrations teach a model to use step confidence?

Four experiments testing whether an LLM can be taught, in context, to act on
per-step confidence when answering multi-hop questions.

**Model:** Qwen3-1.7B, greedy, batch size 1.
**Data:** the 60-question MuSiQue selection and the Test 06 sequential traces.
**Confidence score:** per-step top1–top2 logit `margin`, mapped to 0–100 by a
global min-max, with the trace mean shown as `TRACE CONFIDENCE`. Margin was the
best predictor of final correctness in the aggregation grid (AUROC 0.866 as a
trace mean). The mapping is a monotone rescale for display — not calibration,
and not a probability.

---

## Answer

**No.** Demonstrations work. Demonstrations of *confidence use* do not.

| what changed | answers that changed (of 56) |
|---|---|
| adding worked examples | **17 (30%)** |
| adding confidence to those examples | 4 (7%) |
| permuting which step is flagged uncertain | **1 (2%)** |
| moving a single `UNRELIABLE` flag to a random step | **2 (4%)** |

The prompt has real leverage on this model — a third of answers move when you
add demonstrations. Confidence assignment moves one or two.

---

## Files

| script | what it does | status |
|---|---|---|
| `18_icl_proper_claude.py` | Few-shot ICL with confidence, multi-turn, 5 conditions | **valid** |
| `20_icl_variants_claude.py` | Chain-of-thought and salient-flag variants, 6 conditions | **flag arms valid, CoT arm invalid** |
| `17_icl_confidence_claude.py` | First attempt | **superseded — do not cite** |
| `21_cot_icl_proper_claude.py` | CoT rebuild | **written, never run** |

Outputs live in `outputs/<run>/` as CSV + JSONL + summary JSON, so every number
traces back to raw rows.

```bash
python 18_icl_proper_claude.py   --run-dir outputs/icl_proper_claude
python 20_icl_variants_claude.py --run-dir outputs/icl_variants_claude
```

`_paths.py` imports `common.py` from `basic-tests-updated/` rather than forking
it, so grading and confidence extraction are identical to the rest of the
project and all numbers stay comparable.

---

## 18 — Few-shot ICL with confidence

Four demonstrations, multi-turn `(user, assistant)` pairs, each carrying its real
MuSiQue supporting paragraphs. Two show the lowest-confidence step being wrong so
the answer must depart from the chain (confidence 29 and 10, both minima); two
show a confident chain that should be followed. The two classes require opposite
behaviour, and the script **asserts that "copy the last step" fails on the demo
set** and aborts otherwise.

| condition | accuracy |
|---|---|
| **icl_noconf** | **15/56 (26.8%)** |
| icl_conf | 14/56 (25.0%) |
| icl_conf_shuffled | 14/56 (25.0%) |
| zero_shot | 12/56 (21.4%) |
| zero_shot_conf | 12/56 (21.4%) |

- **Demonstrations work.** `icl_noconf` vs `zero_shot`: +5.4 points, 17/56
  answers changed. This is the positive control that makes the rest readable.
- **Confidence adds nothing on top.** `icl_conf` vs `icl_noconf`: Δ accuracy
  −0.018, 4/56 changed — the version *without* confidence scores higher.
- **Permuting confidence changes nothing.** `icl_conf` vs `icl_conf_shuffled`:
  Δ accuracy **0.0000**, CI [0.000, 0.000], 1/56 changed.

---

## 20 — Chain of thought, and a salient flag

Two attempts at the leading explanation for the nulls: that the model had **no
room to act**. It receives a fixed state and emits ~16 tokens, so even a believed
warning about step 2 cannot be acted on.

### Valid: the flag arms

One step marked `[UNRELIABLE - verify against the documents]`, no numbers
anywhere, against a control placing the flag on a **random** step.

```
icl_flag_lowest vs icl_flag_random
   delta accuracy = 0.0000   CI [0.000, 0.000]
   answers changed = 2 / 56

  (adding the flag at all changes 9 / 56)
```

**This is the sharpest confidence-targeting result in the project.** The model
demonstrably reads and acts on the flag, and acts identically whether it points
at the genuinely least-reliable step or an arbitrary one. It cannot be dismissed
as "the model ignores numeric fields" — there are no numbers. Both arms use
byte-identical demonstrations, so it is also immune to any criticism of them.

### Invalid: the CoT arm — do not cite

Its demonstration rationales came from two string templates, so the prompt
contained only two surface forms and the model copied the form rather than the
behaviour. From a real output:

```
"The lowest confidence is step 1 at 17, which is still high,
 and the documents agree with the state. ANSWER: 2010"
```

17 is near the bottom of the scale. The model reproduced the template verbatim
and slotted in a number contradicting it; mean output length matched the
template length. The arm measured sentence-pattern copying, so its contrasts —
including −0.054 for `cot_conf` vs `cot_noconf` — are not evidence about
confidence.

---

## 17 — Superseded, retained deliberately

Both of its demonstrations had a final answer equal to the **last step's**
answer, so the demo set was perfectly explained by "copy the last step" — a rule
that is confidence-independent by construction. Its low-confidence demo also
claimed in its rationale that correcting a step "changes the final answer" while
displaying the answer that ignoring the correction produces. Demonstrations
carried no documents while the test item carried ~2400 tokens of them.

Its null is uninterpretable. It is kept because the failure generalises:

> **An ICL null is worthless without a positive control showing the
> demonstrations do anything at all.**

Experiment 18 added that control, and 20 added the abort guard.

---

## 21 — Written, never run

Rebuilds the CoT arm properly: hand-written structurally heterogeneous
rationales (pairwise similarity max 0.335, with an abort guard above 0.60),
demonstration document sets containing distractors so "re-check the documents"
is the same task in demo and test, and two validity checks that would have
caught the failure in 20 — a **parrot-similarity score** against the
demonstration wording, and a **step-identification check** asking whether the
model names the true lowest-confidence step.

One known issue remains from its smoke test: the model reads confidence *values*
as step numbers (naming "step 17" in a three-step trace), which corrupts the
step-identification diagnostic. Steps need explicit `STEP n:` labels and the
parser needs to reject indices exceeding the trace length.

---

## Known weaknesses

- **Document asymmetry** in 18 and 20: demonstration turns carry ~2k characters
  of supporting paragraphs only, while the test turn carries ~9.5k including
  distractors. "Re-check against the documents" is therefore easier in the demos
  than at test. 21 fixes this; 18 and 20 do not.
- **Four demonstrations**, two of each type — thin for few-shot.
- 56 questions, and the demonstration gain in 18 is 3 questions with a CI whose
  lower bound is exactly 0. The behavioural gradient (30% / 7% / 2%) is the
  robust part, not the accuracy deltas.
- One model, one dataset, one seed.

---

## Why these nulls are explained rather than mysterious

A separate experiment probed the model's hidden states directly
(`../19_hidden_state_probe_claude.py`). Confidence predicts whether a step is
**wrong** at 0.702 but whether it is **load-bearing** at 0.517 — chance. A
linear probe on the same forward pass recovers load-bearingness at **0.777**.

The information is in the model. It is not in the confidence signal. No
prompting mechanism — bare numbers, demonstrations, chain of thought, or a
salient flag — can recover what the signal does not carry.
