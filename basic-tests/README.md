# Step-Level Confidence vs. Repair Value in Multi-Hop QA

## Overview

This repository contains a pilot study investigating the following question:

> **Does step-level model confidence identify the reasoning error that is most valuable to repair in multi-hop question answering?**

Most work on step-level confidence evaluates whether confidence can identify whether a reasoning step is **correct or incorrect**.

However, correctness and usefulness of repair are not necessarily the same thing.

A model may make several mistakes during a multi-step reasoning process, but correcting one mistake may substantially improve the final answer while correcting another may have little effect or may even cause the downstream trajectory to become worse.

This pilot therefore separates two questions:

1. **Error detection:** Does confidence identify incorrect reasoning steps?
2. **Repair prioritization:** Among incorrect steps, does confidence identify the error whose repair produces the greatest downstream improvement?

The current experiments are intended as **basic feasibility tests**, not final results.

---

# Research Question

The main research question is:

> **Among incorrect reasoning steps, how well does step-level confidence identify the error whose repair most improves downstream performance?**

An additional question emerging from the pilot is:

> **Does confidence provide information about repair value beyond simple structural signals such as hop position?**

---

# Pilot Setup

### Dataset

We use **MuSiQue**, a document-grounded multi-hop QA benchmark containing explicit question decompositions and intermediate answers.

For this pilot, we sampled:

- **60 questions**
- **30 three-hop questions**
- **30 four-hop questions**
- random seed: `42`

The same selected questions are saved in:

```text
outputs/qwen17b_seed42/selection.json
```

### Model

The current pilot tests only:

```text
Qwen/Qwen3-1.7B
```

The run was performed locally using:

```text
Device: Apple MPS
dtype: float16
PyTorch: 2.8.0
Transformers: 4.57.6
```

**Important:** These results currently come from only one model. They should not be interpreted as a general result about LLMs.

Different model sizes and model families may have substantially different confidence, error, and repair behavior.

---

# Confidence Signals

For each generated intermediate answer, we record four white-box confidence signals:

- **Mean token log-probability**
- **Minimum token log-probability**
- **Mean token entropy**
- **Top-1 vs. top-2 logit margin**

Lower log-probability, higher entropy, and smaller margins generally correspond to greater uncertainty.

These are derived directly from the model's token distributions rather than asking the model to verbally report a confidence score.

---

# Experimental Pipeline

## Test 1 — Confidence vs. Hop Correctness

Script:

```text
01_confidence_correctness.py
```

### Goal

Determine whether step-level confidence contains information about whether an intermediate reasoning step is incorrect.

### Procedure

For each MuSiQue question:

1. Use the benchmark-provided decomposition.
2. Resolve decomposition references using benchmark gold intermediate answers so each hop has a fixed semantic target.
3. Ask Qwen3-1.7B to answer each intermediate sub-question.
4. Record the model's answer and confidence signals.
5. Compare each answer against the MuSiQue intermediate gold answer.
6. Measure how well uncertainty distinguishes incorrect from correct hops using AUROC and AUPRC.

A fixed structured reasoning state is also constructed for each question and used by the later repair experiments.

---

## Test 1 Results

Across the 60 questions:

```text
Total intermediate hops:     210
Automatically scored hops:   201
Manual-review hops:             9
Hop accuracy:                53.7%
Baseline final accuracy:     46.3%
```

### Confidence → Error Detection

| Confidence signal | AUROC | AUPRC |
|---|---:|---:|
| Mean log-probability | **0.730** | 0.679 |
| Minimum log-probability | **0.736** | 0.697 |
| Entropy | **0.732** | 0.687 |
| Margin | **0.672** | 0.635 |

Random AUROC would be approximately `0.50`.

### Initial interpretation

The pilot suggests that confidence contains **meaningful but imperfect information about whether a reasoning hop is wrong**.

This establishes an important prerequisite for the main experiment: there is a confidence/error signal to investigate further.

However, detecting that a step is wrong does not tell us whether that step is the most useful one to fix.

---

# Test 2 — Intervention-Defined Repair Value

Script:

```text
02_repair_value.py
```

### Goal

Measure how valuable it is to repair each incorrect reasoning hop.

Instead of assuming that all errors are equally important, we directly intervene on each error.

### Primary intervention: propagated repair

For every clearly incorrect **non-terminal** hop:

1. Start from the same original saved reasoning state.
2. Replace that hop's incorrect answer with its MuSiQue gold answer.
3. Identify downstream decomposition nodes that depend on the repaired hop.
4. Regenerate those dependent downstream nodes.
5. Regenerate the final answer.
6. Measure the change in probability assigned to the correct final answer.

The terminal decomposition hop is excluded because directly replacing an answer-bearing terminal state with the gold answer could make the intervention artificially easy.

### Repair gain

The primary continuous measurement is:

```text
repair gain =
    average log P(gold final answer | propagated repaired state)
  - average log P(gold final answer | original state)
```

Therefore:

```text
positive repair gain  → repair made the gold answer more likely

near-zero repair gain → repair had little effect

negative repair gain  → repair caused the downstream trajectory
                        to make the gold answer less likely
```

### Secondary frozen-repair control

We additionally measure a more localized control:

```text
replace incorrect hop with gold
        ↓
keep all other intermediate answers fixed
        ↓
rescore the gold final answer
```

This distinguishes the direct effect of changing one state from the additional effects introduced by downstream regeneration.

---

# Test 2 Results

The repair analysis focused on questions where the original final answer was incorrect.

```text
Baseline-failed questions rescored:       29
Total repair candidates:                  30
Questions with a repair candidate:        21
Questions with 2+ repair candidates:       8
```

Candidate distribution:

```text
13 questions → 1 candidate
 7 questions → 2 candidates
 1 question  → 3 candidates
```

### Propagated repair effects

```text
Mean repair gain:                  -0.483
Median repair gain:                +0.079
Mean absolute repair gain:          1.308
```

For questions containing multiple repair candidates:

```text
Mean within-question
best-vs-worst repair spread:        2.132 logprob units
```

### Frozen repair control

```text
Mean frozen repair gain:           -0.098
Median frozen repair gain:         +0.107
Mean absolute frozen gain:          0.692
```

The pooled Spearman correlation between frozen and propagated repair gains was approximately:

```text
ρ = 0.848
```

### Final-answer rescue

Only:

```text
1 / 30 propagated repairs
```

turned an automatically scored failed final answer into a correct final answer.

Because binary rescue is sparse in this small pilot, the continuous gold-answer log-probability change is more informative as the primary repair metric.

---

# What Test 2 Suggests

The most important observation is that **different incorrect hops do not have equal repair value**.

Within the same question, correcting one error can meaningfully improve the final-answer probability while correcting another can have little effect or even make the regenerated trajectory worse.

This supports the distinction between:

```text
"this reasoning step is wrong"
```

and

```text
"this reasoning step is the one most useful to repair"
```

That distinction motivates Test 3.

---

# Test 3 — Does Confidence Select the Best Repair?

Script:

```text
03_policy_analysis.py
```

### Goal

For questions containing multiple clearly incorrect non-terminal hops, test whether the **lowest-confidence error** is also the error with the **largest measured repair gain**.

We compare confidence against:

- candidate-adjusted random selection
- earliest incorrect hop
- latest incorrect hop
- oracle highest-repair-gain selection

Metrics include:

- top-1 repair hit rate
- pairwise ranking accuracy
- repair regret
- normalized regret
- Spearman correlation
- final-answer rescue rate
- bootstrap confidence intervals

---

# Test 3 Results

Only **8 questions** contained at least two eligible repair candidates:

```text
7 questions → 2 candidates
1 question  → 3 candidates
```

Because `n = 8`, these numbers should be treated as **pilot/debugging evidence rather than statistical conclusions**.

### Mean log-probability confidence

```text
Confidence-selected best repair:     75.0%
Random expected hit rate:            47.9%
Earliest-error baseline:              0.0%
Latest-error baseline:               87.5%
Pairwise ranking accuracy:           80.0%
Mean normalized repair regret:       0.25
```

The bootstrap 95% interval for the improvement of confidence over random was wide and included zero:

```text
confidence - random:
approximately [-0.104, 0.542]
```

The pooled uncertainty-vs-repair-gain correlation was:

```text
Spearman ρ ≈ 0.204
p ≈ 0.280
```

This is not statistically significant.

---

## Other Confidence Signals

| Signal | Best-repair hit | Pairwise accuracy |
|---|---:|---:|
| Mean log-probability | **75.0%** | **80%** |
| Minimum log-probability | **75.0%** | **80%** |
| Entropy | **75.0%** | **80%** |
| Margin | **87.5%** | **90%** |

Margin performed particularly well descriptively, but this result comes from only eight questions and should **not** yet be interpreted as evidence that margin is definitively superior.

---

# An Important Finding: Hop Position

One of the most interesting observations from the pilot was the strength of a trivial structural baseline.

```text
Earliest incorrect hop → best repair:    0.0%
Latest incorrect hop → best repair:     87.5%
```

The latest-error baseline performed at least as well as the confidence signals in this very small sample.

There are multiple possible explanations.

One possibility is that later errors genuinely have greater direct influence on the final answer.

Another is that propagated repair introduces a structural effect: repairing an early hop requires regenerating more downstream reasoning, creating more opportunities for the model to introduce additional errors.

Therefore, a central next question is:

> **Does confidence predict repair value beyond information already provided by hop position?**

Future experiments should treat hop position as a required baseline rather than interpreting confidence performance alone.

---

# Current Interpretation

These experiments do **not** establish a final answer to the research question.

Instead, they establish that the research question is experimentally viable.

The pilot currently suggests:

### 1. Confidence can detect errors

Step-level white-box confidence has a moderate relationship with whether intermediate answers are correct.

The strongest Test 1 signals reached approximately:

```text
AUROC ≈ 0.73
```

### 2. Different errors have different repair values

Incorrect hops within the same question can differ substantially in their downstream repair effect.

Therefore, error prioritization is a meaningful problem rather than all errors being interchangeable.

### 3. Error detection and repair utility are not identical

Correcting an incorrect reasoning state does not guarantee downstream improvement.

Some gold repairs produced negative propagated repair gain.

This means:

```text
error awareness ≠ necessarily repair/consequence awareness
```

### 4. Confidence may contain information about repair value

In the small ranking subset, confidence selected the highest-value repair more frequently than candidate-adjusted random selection.

However, the sample is far too small to establish this reliably.

### 5. Position is a major competing explanation

The latest-error baseline performed extremely strongly in the pilot.

Future experiments must determine whether confidence adds predictive information **beyond hop position**.

---

# What We Cannot Conclude Yet

The current pilot uses only:

```text
Qwen/Qwen3-1.7B
```

Therefore, we cannot currently claim that these findings generalize to:

- other Qwen model sizes
- other model families
- larger reasoning models
- LLMs in general

Model scale could substantially affect:

- intermediate error frequency
- confidence calibration
- propagation stability
- number of eligible multi-error trajectories
- confidence-to-repair alignment
- positional effects

A stronger model may also produce fewer incorrect intermediate hops, meaning more benchmark questions may be required to obtain enough multi-error cases for reliable repair-ranking analysis.

---

# Current Status

This repository should therefore be interpreted as:

> **A basic feasibility/pilot study validating the experimental pipeline, not a completed study or final research result.**

The current evidence supports continuing the investigation, but a substantially larger and multi-model evaluation is required before drawing conclusions.

---

# Planned Next Steps

The next phase should include:

1. **Scale the number of MuSiQue questions**
   - The current ranking analysis has only 8 eligible multi-error questions.
   - A much larger sample is needed for stable estimates and tighter confidence intervals.

2. **Run multiple model scales**
   - Qwen3-1.7B
   - Qwen3-4B
   - Qwen3-8B

3. **Add another model family**
   - This will test whether observed confidence/repair relationships are Qwen-specific.

4. **Compare confidence directly against position**
   - Test whether confidence contains predictive information after controlling for hop index / reasoning depth.

5. **Analyze frozen and propagated repair separately**
   - This can help determine whether strong latest-hop performance is caused primarily by downstream regeneration or by direct final-answer dependence.

6. **Run additional controls**
   - no-op / semantically equivalent substitutions
   - reasoning-depth breakdowns
   - controlled context-length experiments

7. **Potential repair-aware learning experiment**
   - The intervention procedure automatically creates labels of the form:

```text
(reasoning trace, error) → measured repair gain
```

   - These could later be used to train or fine-tune a model to predict which error is most valuable to repair.
   - Such a model could be compared against raw confidence, random selection, positional baselines, and an oracle repair policy.

---

# Repository Structure

```text
.
├── README.md
├── requirements.txt
├── common.py
│
├── 00_speed_test.py
├── 01_confidence_correctness.py
├── 02_repair_value.py
├── 03_policy_analysis.py
├── 04_noop_control.py
├── 05_depth_context_breakdown.py
│
└── outputs/
    └── qwen17b_seed42/
        ├── selection.json
        │
        ├── 01_summary.json
        ├── 01_hops.csv
        ├── 01_questions.csv
        ├── 01_review_queue.csv
        ├── 01_trace.jsonl
        │
        ├── 02_baseline_rescored.csv
        ├── 02_repairs.csv
        ├── 02_summary.json
        ├── 02_propagated_repairs.jsonl
        ├── 02_review_queue.csv
        │
        ├── 03_policy_summary.json
        ├── 03_question_policy.csv
        ├── 03_question_policy_mean_logprob.csv
        ├── 03_question_policy_min_logprob.csv
        ├── 03_question_policy_entropy.csv
        └── 03_question_policy_margin.csv
```

The main reported pilot results come from:

```text
01_confidence_correctness.py
02_repair_value.py
03_policy_analysis.py
```

`common.py` contains the shared model loading, prompt construction, scoring, confidence measurement, and utility functions used by these experiments.

`00_speed_test.py` is an infrastructure benchmark rather than a research experiment.

`04_noop_control.py` and `05_depth_context_breakdown.py` are additional control/exploratory analyses and are not part of the main results summarized above.

---

# Reproducing the Pilot

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check Apple MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Run Test 1:

```bash
python 01_confidence_correctness.py \
  --model Qwen/Qwen3-1.7B \
  --n-questions 60 \
  --batch-size 1 \
  --score-batch-size 1 \
  --run-dir outputs/qwen17b_seed42
```

Run Test 2:

```bash
python 02_repair_value.py \
  --run-dir outputs/qwen17b_seed42 \
  --generate-batch-size 1 \
  --score-batch-size 1
```

Run Test 3:

```bash
python 03_policy_analysis.py \
  --run-dir outputs/qwen17b_seed42
```

---

# Summary

The pilot currently provides three useful observations:

```text
Confidence → incorrect hop:
moderate signal (~0.73 AUROC)

Incorrect hop → repair value:
substantial variation exists

Confidence → highest-value repair:
possible signal, but only 8 eligible questions
and hop position is a strong competing baseline
```

The current results therefore **do not answer the research question yet**.

They instead show that:

> **The problem is measurable, incorrect steps differ in repair value, and the relationship between confidence and repair utility is worth testing at larger scale and across multiple models.**