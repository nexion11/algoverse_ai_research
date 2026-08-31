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

> **Confidence detects that a reasoning step is wrong. It carries no usable
> information about whether that step matters. And supplying it to the model
> does not change what the model does.**

On the same 150 hops, with the same signals:

| | AUROC |
|---|---|
| Is this step **wrong**? | **0.702** [0.615, 0.782] |
| Is this step **load-bearing**? | 0.517 [0.420, 0.618] |

The load-bearing/inert distinction is real and large — it just is not something
confidence can see.

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

# Overall conclusion

> **Wrongness and influence are separable axes of a reasoning step, and this
> model's confidence is sensitive to only one of them.**

This reframes rather than abandons the original question. Repair prioritization
requires knowing which errors matter. Confidence does not encode that, which is
why the earlier pilot kept losing to hop position, why no aggregation of it
predicts repair value, and why supplying it to the model changes nothing.

The load-bearing/inert distinction the project set out to find **does exist** and
is large. The negative result is about confidence as a way of detecting it.

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

1. **De-duplicate the question selection** on shared sub-question chains before
   scaling. The current 60 contain at least one 4-way near-duplicate cluster.
2. **Scale influence measurement**, not repair ranking. It gives ~2.5 scored
   hops per question instead of a candidate pair on 1 question in 7.
3. **Test whether any signal predicts influence** — hidden-state probes are the
   obvious candidate, since verbalized and token-level confidence both fail.
4. **Add the constant-value control** to the bake-in experiment to settle whether
   the `agg_only` gain is purely instructional.
5. **Larger models and a non-Qwen family**, at which point vLLM on rented CUDA
   becomes worthwhile. Note that vLLM returns only top-k logprobs, so
   full-vocabulary entropy — one of the two strongest signals here — needs
   verification before migrating.
