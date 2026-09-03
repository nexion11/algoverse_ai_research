# Step Confidence, Error Influence, and Repair Value in Multi-Hop QA

Second round of experiments on the MuSiQue / Qwen3-1.7B pilot.

All work here is additive. No existing script or output was modified. Every new
script is suffixed `_claude.py` and writes to its own output directory.

**Subject model:** `Qwen/Qwen3-1.7B`, Hugging Face Transformers, Apple MPS,
float16, greedy (`do_sample=False`), batch size 1 throughout.
**Data:** the existing 60-question MuSiQue selection (30 three-hop, 30 four-hop,
seed 42) and the Test 06 sequential traces.
**Compute:** ~2 hours on MPS.

---

## Headline result

> **The model internally represents which reasoning steps carry the answer.
> It never expresses that in confidence.**

On the same 150 hops, from the same single forward pass:

| target | best confidence signal | **hidden-state probe** |
|---|---|---|
| is this step **wrong**? | 0.702 (entropy) | **0.807** |
| is this step **load-bearing**? | 0.550 — chance | **0.777** |

Confidence detects errors. It is at chance on whether an error matters. A linear
probe on the residual stream recovers that at 0.777 (permutation p = 0.005).

Nine experiments established that no form of confidence -- verbalized,
token-level, normalized, aggregated, supplied directly, or taught by
demonstration -- carries information about which steps matter. The natural
reading was that the information does not exist. It does. It is simply not in
the channel anyone was reading.

**Which errors are worth repairing is decodable from the model's internals, but
not from anything the model reports about its own confidence.**

---

## Contents

| Script | What it does | Output |
|---|---|---|
| `12_confidence_aggregation_claude.py` | 200-cell grid of per-step signal × normalization × aggregation vs the final result | `outputs/qwen17b_aggregation_claude/` |
| `13_step_influence_claude.py` | Frozen-substitution influence measurement on every non-terminal hop | `outputs/qwen17b_influence_claude/` |
| `13b_analyze_influence_claude.py` | Analysis of the above | same |
| `14_verbal_elicitation_sweep_claude.py` | Five verbalized-confidence elicitation prompts compared | `outputs/qwen17b_elicitation_claude/` |
| `15_policy_metrics_claude.py` | The repository's repair-policy metric suite, with CIs and clustering correction | `outputs/qwen17b_policy_claude/` |
| `16_bake_confidence_claude.py` | Supplying per-step confidence and its aggregate to the final-answer step | `outputs/qwen17b_bake_claude/` |
| `11b_sequential_confidence_intervention_claude.py` | Confidence-intervention harness — **written, never executed** | — |
| `confidence-experiments/icl/17_icl_confidence_claude.py` | First ICL attempt — **superseded, invalid demonstrations** | `confidence-experiments/icl/outputs/icl_claude/` |
| `confidence-experiments/icl/18_icl_proper_claude.py` | In-context learning, corrected | `confidence-experiments/icl/outputs/icl_proper_claude/` |
| `confidence-experiments/19_hidden_state_probe_claude.py` | Hidden-state probe — the headline result | `confidence-experiments/outputs/probe_claude/` |
| `confidence-experiments/icl/20_icl_variants_claude.py` | Chain-of-thought and salient-flag ICL variants | `confidence-experiments/icl/outputs/icl_variants_claude/` |
| `confidence-experiments/icl/21_cot_icl_proper_claude.py` | CoT rebuild — **written, never run** | — |

Reproduce:

```bash
cd basic-tests-updated
python 12_confidence_aggregation_claude.py --run-dir outputs/qwen17b_aggregation_claude
python 13_step_influence_claude.py       --run-dir outputs/qwen17b_influence_claude
python 13b_analyze_influence_claude.py   --run-dir outputs/qwen17b_influence_claude
python 14_verbal_elicitation_sweep_claude.py --run-dir outputs/qwen17b_elicitation_claude
python 15_policy_metrics_claude.py       --run-dir outputs/qwen17b_policy_claude
python 16_bake_confidence_claude.py      --run-dir outputs/qwen17b_bake_claude
```

---

# TODO 1 — Raw verbalized confidence, and tuning the elicitation prompt

**Status: complete.**

### What was run

Raw verbalized confidence was carried as a first-class signal through every
experiment below. Separately, five elicitation prompts were compared on the
**same fixed Test 06 answers** (210 hops × 5 prompts = 1,050 calls, 0% parse
failures). Answers are never regenerated, so eliciting confidence cannot change
what is being judged and the variants are directly comparable.

| Variant | Description |
|---|---|
| `baseline` | byte-identical to `06_sequential_confidence.py`; control |
| `anchored` | full-range instruction, explicit calibration anchors, "avoid round numbers" |
| `bins` | seven labeled bins mapped to numbers |
| `evidence_first` | short evidence statement, then the number |
| `twosided` | asks P(answer is **incorrect**), then inverts |

### Results

| variant | distinct values | mode | mode share | flat traces | **AUROC** | AUPRC |
|---|---|---|---|---|---|---|
| **baseline** | 9 | 40 | 0.533 | 18 | **0.626** | 0.669 |
| evidence_first | 5 | 100 | 0.629 | 20 | 0.609 | 0.672 |
| anchored | 8 | 75 | 0.829 | 34 | 0.584 | 0.648 |
| bins | 3 | 67 | 0.924 | 47 | 0.549 | 0.619 |
| twosided | 3 | 50 | 0.752 | 31 | **0.371** | 0.553 |

### Findings

1. **Raw verbalized confidence is the weakest signal available.** 0.626 AUROC
   for hop error detection, 0.692 at best for predicting final correctness,
   0.617 for detecting wrongness in the influence experiment. White-box signals
   sit at 0.70–0.72 on the same data.

2. **Raw beats every normalization**, confirming the existing finding and
   extending it to new targets. Within-trace z-scoring is *worse than chance*
   averaged across the aggregation grid (0.473).

3. **Prompt tuning does not help. All four variants scored below baseline.**
   Anchoring made the distribution *more* degenerate, not less — mode share
   0.53 → 0.83, flat traces 18 → 34. It relocated the attractor from 40 to 75
   rather than spreading it. Constraining the output to labeled bins was worse
   still: 3 values used, 92% on one label, 47 of 60 traces completely flat.

4. **`twosided` explains why tuning cannot work.** Asking for the probability
   the answer is *incorrect* and inverting gives AUROC **0.371** — meaningfully
   below chance — and Spearman against token-F1 flips sign, +0.241 → −0.272. A
   model holding a genuine internal correctness estimate would produce a
   consistent signal under that reversal. This one does not. The value is a
   response to the surface framing of the prompt, not a readout of a belief
   about correctness.

   **There is no latent signal being poorly expressed, so no phrasing recovers
   one.** This is a claim about Qwen3-1.7B on this task, not about verbalized
   confidence generally.

5. **Control validation:** `baseline` reproduced the original Test 06
   distribution exactly — 9 distinct values, mode 40, mode share 0.5333
   (112/210), 18 flat traces, AUROC 0.6264.

---

# TODO 2 — Which per-step confidence score and which aggregation help the final result

**Status: complete, in both readings of "helps".**

"Helps the final result" is ambiguous between *predicts* the outcome and
*improves* the outcome. Both were run. The predictive half is here; the causal
half is TODO 3.

### What was run

**A. Aggregation grid.** 5 signals × 3 normalizations (raw / within-trace z /
within-trace min-max) × 15 aggregations (mean, min, max, median, first, last,
range, std, position-weighted both directions, token-weighted, product,
noisy-or, harmonic, min-over-terminal-ancestors) = 200 usable cells, scored on
56 questions against final correctness (AUROC/AUPRC) and against
`baseline_final_gold_logprob` — the quantity `repair_gain` is a difference of.

25 cells were dropped as constant by construction: within-trace min-max forces a
0 and a 1 into every trace, so `minmax|min`, `minmax|max` and `minmax|range`
carry no information.

**B. The repository's policy metric suite**, implemented to the existing
definitions: expected top-1 under uniform tie-breaking, candidate-adjusted
random `K/N`, pairwise accuracy with 0.5 ties, repair regret, normalized regret,
selected vs oracle gain, rescue rate, and the earliest/latest-wrong structural
baselines — with two additions: bootstrap confidence intervals, and a
correction for non-independent questions.

### Results

**Aggregation choice does not matter. Signal choice does.**

| Signal | Best aggregation | AUROC | Best AUPRC |
|---|---|---|---|
| margin | mean | **0.866** | 0.620 |
| entropy | median | 0.821 | **0.679** |
| mean_logprob | median | 0.818 | 0.675 |
| min_logprob | median | 0.787 | 0.660 |
| verbal | last | 0.692 | 0.414 |

Eight different margin aggregations fill the top ten, all between 0.808 and
0.866. Product, noisy-or, harmonic mean and position weighting buy nothing over
a plain average. This is not an artifact of short traces — the aggregations
genuinely differ from one another (median pairwise Spearman 0.68, only 12% of
pairs above 0.9) and still perform alike.

**Mean AUROC by normalization, across all 200 cells:** raw **0.683**,
min-max 0.590, z-score **0.473**.

**Which signal wins is unsettled.** Margin leads on AUROC; entropy and
mean_logprob lead on AUPRC, which is the more appropriate metric given only 14
of 56 finals are correct. A best-of-grid permutation test (5,000 shuffles,
recomputing all 200 cells each time) confirms the grid contains real signal
(p = 0.0024) but that p applies to the maximum cell, not to its identity.

**Against the repair metric, everything is null.**

| Signal | top-1 | −random | 95% CI | dedup top-1 | dedup CI |
|---|---|---|---|---|---|
| mean_logprob | 0.875 | +0.375 | [0.125, 0.500] | 0.80 | **[−0.10, 0.50]** |
| entropy | 0.750 | +0.250 | [−0.125, 0.500] | 0.60 | [−0.30, 0.50] |
| margin | 0.750 | +0.250 | [0.0, 0.500] | 0.60 | [−0.30, 0.50] |
| min_logprob | 0.625 | +0.125 | [−0.250, 0.500] | 0.40 | [−0.50, 0.30] |
| verbal | 0.5625 | +0.0625 | [0.0, 0.1875] | 0.50 | **[0.0, 0.0]** |
| *latest-wrong baseline* | **1.000** | — | — | **1.000** | — |

### Findings

1. **The implementation reproduces the existing published numbers exactly** —
   mean_logprob top-1 0.875, random 0.5, pairwise 0.875, regret 0.546,
   normalized regret 0.125, selected gain +0.304, oracle +0.850.

2. **The 87.5% headline does not survive two corrections.** First, bootstrap
   CIs: even at n=8 the interval is [0.125, 0.500]. Second, and more
   importantly, **four of the eight multi-candidate questions share hops 2–4**
   (`4hop3__220945_88460_30152_20999`, `310746_…`, `405277_…`, `426860_…`).
   They are MuSiQue variants of a single reasoning chain, not independent
   samples. Collapsing them gives n=5, and **every confidence signal's CI then
   includes zero.** The latest-wrong positional baseline stays at 100% in both.

   *Recommendation: de-duplicate the question selection on shared sub-question
   chains before scaling.*

3. **Verbalized confidence is exactly random for repair selection** — top-1 0.50,
   CI [0.0, 0.0], mean selected gain −0.031 against an oracle of +0.850.

4. **Trace-level aggregation does not predict question-level repair value.**
   Across ~80 cells on 36 questions, only 2 reach p<0.05 where 4.0 are expected
   by chance. The strongest (`margin|first`, ρ = 0.498, p = 0.002) does not
   survive Bonferroni at 0.05/80.

5. **`repair_gain` and final correctness can disagree.** Among the 16 repairs in
   the multi-candidate subset, exactly one produced a correct final answer — and
   its `repair_gain` was **−0.71**. Optimizing logP(gold) selected against the
   only repair that actually worked. Worth noting since the entire policy suite
   is built on the log-probability version.

---

# TODO 3 — Baking per-step confidence and the aggregation method into the model

**Status: Tier 1 complete. Tier 2 not attempted.**

Baking in has two tiers. **Tier 1** supplies confidence and its aggregate as
input state to the final-answer step and measures whether the answer improves.
**Tier 2** trains on it (LoRA, or a confidence head). Tier 1 must come first: it
isolates whether the information is *usable at all*, separately from whether the
model has been taught to use it.

### What was run

60 questions × 5 conditions. Hop answers are byte-identical in every condition;
only the metadata differs.

| Condition | Final prompt contains |
|---|---|
| `control` | the original confidence-free prompt |
| `per_step` | + a CONFIDENCE value on every step |
| `agg_only` | + a single TRACE RELIABILITY number, no per-step values |
| `per_step_agg` | + both |
| `shuffled` | + per-step confidences **randomly permuted across steps** |

`shuffled` is the control that decides the result. It preserves the multiset of
numbers, the aggregate, the prompt length and the format; only the *assignment*
of confidence to step changes.

### Results

| condition | Δ logP(gold) vs control | 95% CI | accuracy | Δ accuracy |
|---|---|---|---|---|
| control | — | — | 14/60 (23.3%) | — |
| per_step | −0.143 | [−0.449, +0.165] | 13/60 | −1 question |
| **agg_only** | **+0.233** | **[+0.050, +0.429]** | 15/60 | +1 question |
| per_step_agg | −0.084 | [−0.442, +0.264] | 13/60 | −1 question |
| shuffled | −0.154 | [−0.445, +0.134] | 13/60 | −1 question |

**The decisive comparison — `per_step` vs `shuffled`:**

```
Δ logP(gold)  = +0.012   95% CI [−0.035, +0.062]
Δ accuracy    =  0.000   (13/60 in both conditions)
```

### Findings

1. **Supplying per-step confidence does not help, and the model cannot tell
   correctly-assigned confidence from randomly-assigned confidence.** Identical
   accuracy, and a logP difference tightly centred on zero. Whatever the model
   is doing with those numbers, it is not reading which step is uncertain.

2. **The one significant improvement is a prompt effect, not a confidence
   effect.** `agg_only` genuinely raises logP(gold) with a CI excluding zero.
   But the supplied TRACE RELIABILITY value is nearly constant across
   questions — mean 44.5, **std 6.3**, IQR 40–47.5, 13 of 60 at exactly 40 —
   because it is the mean of an already-degenerate signal. And its correlation
   with the gain is **−0.127 (p = 0.334)**: the improvement does not depend on
   the number. The instruction ("if reliability is low, re-check the state
   against the documents") is doing the work. On accuracy it is one question out
   of sixty. Adding real per-step numbers on top (`per_step_agg`) cancels it.

   *Limitation: no matched constant-value control arm was run, so this reading
   rests on the value-gain correlation. Adding that arm would settle it.*

3. **Tier 2 was not attempted, and the evidence argues against it.** Training a
   model to consume a signal that shows no effect when supplied directly, and
   that the shuffled control indicates it cannot read, is unlikely to be a good
   use of GPU time. This is a recommendation, not a conclusion — there may be
   reasons to want the fine-tune regardless.

4. Needs-review finals were 4 per condition, identical across all five, so they
   do not bias the comparison.

---

# Additional experiment — Step influence (load-bearing vs inert)

Not part of the three items above. It is the strongest result obtained, and it
explains why the other three come back null.

### Motivation

Repair experiments can only rank hops on questions with ≥2 wrong non-terminal
candidates — 8 questions here, effectively ~5 once the shared chain is
collapsed. Nothing can be ranked reliably at that size.

This measures a different quantity, defined for **every** hop, correct or wrong:

> How much does the final answer actually depend on this step's answer?

That is the load-bearing question, and it is separable from whether the step was
right. It yields **150 scored hops instead of 8 ranked candidates**.

### Design — frozen substitution

For each non-terminal hop: take the original trace, substitute only that hop's
answer, **regenerate nothing**, rebuild the final prompt and rescore. Holding
regeneration depth at zero for every hop turns position from a mechanical
advantage into a testable covariate.

| Substitution | Meaning |
|---|---|
| `redact` → `[unavailable]` | leave-one-out; primary measure, needs no gold |
| `gold` | repair direction |
| `crossq` | same hop index from a different question; corruption direction |
| `sham` → itself | control; prompt is byte-identical |

### Results

**Sham control: max \|Δ logP\| = 0.0 across 30 replays, 0 answer changes.** Zero
noise floor.

**Influence is real and strongly graded:**

| | mean \|Δ logP(gold)\| | flips the final answer |
|---|---|---|
| inert tercile | 0.025 | 8% |
| load-bearing tercile | 1.134 | 28% |

A 45× spread. Overall, removing a step changes the final answer only 14.7% of
the time, so roughly 85% of reasoning steps are close to inert for the final
answer.

**Confidence detects wrongness but not load-bearingness** — same 150 hops, same
signals, same rows:

| Signal | AUROC: step is **wrong** | AUROC: step is **load-bearing** |
|---|---|---|
| mean_logprob | **0.702** [0.615, 0.782] | 0.517 [0.420, 0.618] |
| entropy | **0.702** [0.614, 0.783] | 0.516 [0.416, 0.615] |
| min_logprob | **0.698** [0.612, 0.781] | 0.512 [0.413, 0.613] |
| margin | 0.644 [0.555, 0.729] | 0.518 [0.414, 0.614] |
| verbal | 0.617 [0.530, 0.697] | 0.550 [0.462, 0.643] |

Every load-bearing interval straddles 0.5. A best-of-grid permutation test over
the confidence-vs-influence family gives **p = 0.25**.

**The 2×2**, mean log-probability by cell — confidence varies down the rows and
is flat across the columns:

| | inert | load-bearing |
|---|---|---|
| **correct step** | −0.014 | −0.031 |
| **incorrect step** | −0.074 | −0.082 |

Wrong-and-load-bearing vs wrong-and-inert: no signal separates them, all
Mann-Whitney p > 0.31.

**Position predicts repair value but not influence.** The latest wrong hop won
8/8 in the repair experiment, but influence is flat with position: ρ = 0.051,
p = 0.53. So latest-hop dominance is not because later steps are intrinsically
more load-bearing — it lives in the regeneration dynamics after a repair, not in
what the final answer depends on.

### Why this is trustworthy as a negative result

- **The positive control fires**: the same signals on the same rows detect
  wrongness at 0.702, matching the previously published 0.712.
- **The sham control is exactly zero.**
- **Influence is not uniformly small** — a 45× spread between terciles.
- **n = 150**, not 8.

---

# TODO 3 continued — In-context learning

**Status: complete, after one invalid attempt.**

### The first attempt was broken, and is retained as superseded

`17_icl_confidence_claude.py` used two demonstrations in which the final answer
equalled the LAST STEP's answer. The demo set was therefore perfectly explained
by "copy the last step" -- a rule that is confidence-independent by
construction, so the demonstrations could not have taught confidence use no
matter what the model did. The low-confidence demo also asserted in its
rationale that correcting a step "changes the final answer" while displaying the
answer that ignoring the correction produces. Demonstrations carried no
documents while the test item carried ~2400 tokens of them, and everything sat
in one user turn rather than multi-turn.

Its null is uninterpretable and should not be cited. It is kept in the
repository because the failure is instructive: an ICL null is worthless without
a positive control showing the demonstrations do anything at all.

### The corrected experiment

`18_icl_proper_claude.py` fixes all four problems. Demonstrations are sourced
from Test 07 repair results -- real cases where replacing a wrong hop with gold
produced a DIFFERENT and CORRECT final answer -- so discounting a step provably
changes the outcome rather than being asserted to. Four demonstrations: two
where the lowest-confidence step is wrong and the answer must depart from the
chain (confidence 29 and 10, both minima), two where everything is confident and
the chain should be followed. The two classes require opposite behaviour, so no
confidence-blind heuristic satisfies both. The script ASSERTS that "copy the
last step" fails on the demo set and aborts otherwise. Demos carry their real
MuSiQue supporting paragraphs and are presented multi-turn.

Confidence is per-step `margin` on a 0-100 global scale with the trace mean
shown as TRACE CONFIDENCE.

### Results (56 held-out questions)

| condition | accuracy | mean logP(gold) |
|---|---|---|
| **icl_noconf** | **15/56 (26.8%)** | -6.522 |
| icl_conf | 14/56 (25.0%) | -6.779 |
| icl_conf_shuffled | 14/56 (25.0%) | -6.775 |
| zero_shot | 12/56 (21.4%) | -6.663 |
| zero_shot_conf | 12/56 (21.4%) | -6.821 |

### Findings

1. **In-context learning works on this task.** `icl_noconf` vs `zero_shot`:
   +5.4 points accuracy and **17 of 56 answers changed**. This is the positive
   control the first attempt lacked.

2. **Confidence adds nothing on top of it.** `icl_conf` vs `icl_noconf` --
   identical demonstrations, identical targets, confidence present or stripped:
   delta accuracy -0.018, only 4 of 56 answers differ. The stripped version
   scores slightly higher.

3. **Permuting confidence changes nothing.** `icl_conf` vs
   `icl_conf_shuffled`: delta accuracy **0.0000**, CI [0.000, 0.000], with
   **1 of 56** answers differing.

4. **The behavioural gradient is the robust result**, independent of accuracy
   noise:

   | manipulation | answers changed |
   |---|---|
   | add demonstrations | **17/56 (30%)** |
   | add confidence to those demonstrations | 4/56 (7%) |
   | permute which step is flagged uncertain | **1/56 (2%)** |

   A third of answers move when worked examples are added. One moves when the
   confidence assignment is scrambled.

*Limits: 56 questions; the demonstration gain is 3 questions with a CI whose
lower bound is exactly 0, so it should not be sold hard. `needs_review` rose to
28 of 280 rows as demonstrations made outputs more verbose.*

---

# Headline experiment — Hidden-state probe

Not one of the three items. It is the strongest result in the repository and it
reframes all of them.

### The question nothing else asked

Every prior experiment asks whether the model USES a confidence signal it is
handed. All are null. Experiment 13 found the deeper problem: confidence
predicts wrongness at 0.702 but load-bearingness at 0.517. No prompting or
training can extract information a signal does not carry.

But nothing tested whether the information exists ANYWHERE in the model.
Verbalized and token-level confidence are both narrow read-outs.

### Method

One forward pass per hop over the same hop prompt Test 06 used. Last-position
hidden state at five layers. L2 logistic regression, GroupKFold **by question**
-- hops from one question share documents, so a random split would leak.
Baselines are the same confidence signals on identical rows and folds.

### Results

| target | n | positives | best confidence | **probe (layer 28)** | permutation p |
|---|---|---|---|---|---|
| is_wrong | 203 | 119 | 0.702 | **0.807** | 0.005 |
| is_load_bearing | 150 | 50 | 0.550 | **0.777** | 0.005 |

Probe AUROC by depth, `is_load_bearing`:

| layer | 1 | 7 | 14 | 21 | 28 |
|---|---|---|---|---|---|
| AUROC | 0.562 | 0.650 | 0.647 | 0.679 | **0.777** |

Near chance early, emerging with depth -- the profile of a semantic property,
not of noise-fitting.

### Validation

**Permutation null.** Shuffled labels give mean 0.485, p95 0.590 against an
observed 0.777. p = 0.005.

**Not position or structure.** hop index 0.512, position fraction 0.524,
hops-from-end 0.467, n_hops 0.470; all four combined under the same CV give
**0.392**, worse than chance. Adding them to the probe leaves it unchanged at
0.777.

**Not leakage.** Grouped CV keeps hops from one question on one side of the
split.

**Robust to specification.** Outliers clipped 0.780; C=0.05 0.774; C=10 0.788;
10-fold 0.762. Every variant lands in 0.762-0.788.

### Finding

**The model encodes which steps are load-bearing; confidence does not report
it.** This is why every prompting and demonstration experiment came back null --
the information was never in the channel being read.

It also gives repair prioritisation a concrete route that does not depend on
confidence at all: decode step importance from the residual stream of a forward
pass that is already being run.

*Limits: n=150 with 50 positives; one model, one dataset, one seed; the probe
inherits the frozen-substitution definition of load-bearing from experiment 13;
and it is trained and evaluated on the same 60 questions. Grouped CV protects
against leakage but replication on fresh questions matters more here than
anywhere else, because this is the result worth building on. Separately, for
`is_wrong` the layer-1 probe already reaches 0.723, which is high for so early a
layer and hints at surface features; that does not affect the load-bearing
result, where layer 1 sits at 0.562.*

---

# TODO 3 continued — ICL variants: room to act, and a salient flag

**Status: the flag result is valid. The chain-of-thought arm is not.**

### Motivation

The model reads the confidence numbers -- logP shifts on 45 of 58 questions when
they change -- but does not act on them. One explanation fits every null so far:
**it has no room to act.** In every experiment it receives documents plus a fixed
state and emits ~16 tokens. Re-deriving a step from the documents and
recomputing the chain does not fit in a bare short answer. Two interventions
follow.

**Variant 1, chain of thought.** Demonstrations where the assistant reasons
before answering, so there is space to act on a flagged step.

**Variant 2, salient flag.** Exactly one step marked
`[UNRELIABLE - verify against the documents]`, no numbers anywhere. Against a
control where the flag is placed on a RANDOM step.

### Results (56 held-out questions)

| condition | accuracy |
|---|---|
| icl_cot_noconf | 16/56 (28.6%) |
| icl_noconf | 15/56 (26.8%) |
| icl_flag_lowest | 14/56 (25.0%) |
| icl_flag_random | 14/56 (25.0%) |
| icl_cot_conf | 13/56 (23.2%) |
| icl_cot_conf_shuffled | 12/56 (21.4%) |

### The valid result: flag placement does not matter

`icl_flag_lowest` vs `icl_flag_random` -- identical prompt shape, exactly one
flag in both arms, only WHICH step it points at differs:

```
delta accuracy = 0.0000   CI [0.000, 0.000]
answers changed = 2 / 56
```

The flag itself demonstrably does something: adding it changes **9 of 56**
answers relative to no flag. The model reads the instruction and acts on it. It
acts identically whether the flag points at the genuinely least-reliable step or
an arbitrary one.

**This is the sharpest confidence-targeting null in the repository**, because it
cannot be dismissed as "the model ignores numeric fields". There are no numbers.
It is one salient English instruction, the model responds to it, and
confidence-based targeting adds nothing over random targeting.

It is also immune to every criticism of the demonstrations, since both arms use
byte-identical demos -- whatever they taught or failed to teach, they taught
equally to both.

### The invalid result: the CoT arm measured template copying

Its demonstration rationales were generated from two string templates, so the
prompt contained only two surface forms. The model copied the form rather than
the behaviour. From a real output:

```
"The lowest confidence is step 1 at 17, which is still high, and the
 documents agree with the state. ANSWER: 2010"
```

17 is near the bottom of the scale, not "still high". The model reproduced the
Type-B template verbatim and slotted in a number contradicting it. Mean output
length was 147 characters, almost exactly the template length.

**The CoT contrasts should not be cited** -- including the -0.054 for
`cot_conf` vs `cot_noconf`. A second weakness affects the CoT arm and
experiment 18 both: demonstration turns carried ~2k characters of supporting
paragraphs only, while the test turn carried ~9.5k including distractors, so
"re-check against the documents" was an easier task in the demos than at test.

`21_cot_icl_proper_claude.py` rebuilds it with hand-written structurally
heterogeneous rationales (pairwise similarity max 0.335, with an abort guard
above 0.60), demonstration document sets containing distractors, and two new
validity checks -- a parrot-similarity score and a step-identification check
asking whether the model names the true lowest-confidence step. **It has never
been run.** Its smoke test surfaced a further issue to fix first: the model
reads confidence VALUES as step numbers ("step 17" in a three-step trace), which
corrupts the step-identification diagnostic. Steps need explicit `STEP n:`
labels and the parser needs to reject indices exceeding the trace length.

### Finding

Giving the model room to reason did not rescue confidence, and neither did
replacing five numbers with one salient word. Combined with the probe, the
picture is coherent: **the information is not in the signal, so no prompting
mechanism can recover it.**

---

# Overall conclusion

> **Wrongness and influence are separable axes of a reasoning step. The model
> represents both internally, but its confidence reports only the first.**

The original question -- can confidence tell you which errors are most valuable
to repair -- has a clean answer: **no**, and the reason is now specific rather
than mysterious. Confidence detects errors at ~0.70 and is at chance on whether
an error matters. That single fact explains why repair prioritisation kept
losing to hop position, why no aggregation of confidence predicts repair value,
why supplying confidence to the model changes nothing, and why demonstrating its
use teaches nothing.

But the load-bearing/inert distinction the project set out to find **does exist**
-- a 45x influence spread between terciles -- **and the model does represent
it**, at 0.777 from a linear probe on the residual stream.

So the negative result is about the confidence channel specifically, not about
the model's knowledge. The productive direction is to stop asking the model how
confident it is and start decoding what it already represents.

# Limitations

- One model (`Qwen3-1.7B`), one dataset (MuSiQue), one seed (42). Nothing here
  should be generalized to other models or scales.
- Influence is measured **frozen**. A step could matter in a live rollout by
  derailing downstream reasoning while showing low frozen influence. The
  propagated counterpart is the existing Test 07.
- 56–60 questions; 14 correct finals. Confidence intervals are wide throughout
  and are reported rather than elided.
- `agg_only` lacks a matched constant-value control arm.
- Verbalized confidence findings are specific to this model's elicitation
  behaviour and should not be read as a claim about verbalized confidence in
  general.

# Suggested next steps

1. **Replicate the probe on fresh questions.** It is trained and evaluated on
   the same 60. Grouped CV protects against leakage, but this is the result
   worth building on and it needs out-of-sample confirmation before anything is
   built on top of it.
2. **De-duplicate the question selection** on shared sub-question chains. The
   current 60 contain a 4-way near-duplicate cluster (`88460_30152_20999`).
3. **Scale influence measurement, not repair ranking.** It yields ~2.5 scored
   hops per question instead of a candidate pair on one question in seven.
4. **Probe-guided repair.** Use the probe to select which step to re-check and
   measure end-to-end accuracy, against a random-step control. This is the first
   selection rule in the project with a signal behind it.
5. **Larger models and a non-Qwen family**, at which point vLLM on rented CUDA
   is worthwhile. Note that vLLM returns only top-k logprobs, so full-vocabulary
   entropy needs verification before migrating.
6. **Fine-tuning is not the obvious next move.** Three experiments show the
   model cannot use a supplied confidence signal, and one shows the signal
   lacks the information regardless. Training a model to read an uninformative
   field has a ceiling that has already been measured. If anything is to be
   trained, train on the probe target.
