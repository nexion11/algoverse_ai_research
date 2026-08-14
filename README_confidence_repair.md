# Confidence, Error Localization, and Repair Value in Multi-Hop QA

## Project status

This repository contains **pilot / methodology-validation experiments**, not final paper results.

The current goal is to study whether an LLM's step-level confidence can identify:

1. **which reasoning step is wrong**, and
2. **which wrong step is most valuable to repair**.

The current experiments use only **Qwen3-1.7B** on **60 MuSiQue questions** (30 three-hop and 30 four-hop questions, seed 42). Results may change substantially with larger models, other model families, more questions, or other datasets.

The central research distinction is:

> **Error detection is not necessarily the same as repair prioritization.**

A step can be wrong but have little downstream effect, while another wrong step may be much more important to repair.

---

# Research question

The current research question is:

> **Among incorrect reasoning steps, how well does step-level confidence identify the error whose repair most improves downstream performance?**

The follow-up additionally asks:

> **Does confidence still localize errors when reasoning is generated sequentially, and does raw, normalized, or verbalized confidence best identify the wrong or most valuable-to-repair hop?**

A major structural baseline that emerged from the pilot is **hop position**, especially the latest wrong hop.

---

# Repository files and which ones were actually run

## Shared code

### `common.py`

Shared utilities used by the experiment scripts:

- Qwen loading
- MuSiQue loading
- prompt construction
- reference resolution
- generation
- token-level confidence extraction
- gold-answer scoring
- answer grading
- AUROC/AUPRC helpers
- JSON/CSV helpers

---

## Infrastructure only

### `00_speed_test.py`

Used only to benchmark inference speed and memory on Apple MPS.

This is **not a research result**.

---

# Part I — Original isolated-hop baseline

These were the original basic tests.

## `01_confidence_correctness.py`

### Purpose

Tests whether white-box confidence predicts whether a MuSiQue reasoning hop is correct.

### Important design choice

Each hop is evaluated independently using a **gold-resolved subquestion**.

Example:

```text
Hop 1 model answer: WRONG

Benchmark Hop 2 template:
"Where was #1 born?"
```

For this isolated-hop baseline, `#1` is replaced with the **benchmark gold Hop 1 answer**, not the model's wrong Hop 1 answer.

This gives every hop a fixed benchmark meaning and allows clean hop-level correctness grading.

### Confidence signals

For each generated hop answer, we record:

- mean token log-probability
- minimum token log-probability
- mean token entropy
- top-1 vs. top-2 logit margin

For error detection, these are converted to uncertainty scores:

```text
mean-logprob uncertainty = -mean_logprob
min-logprob uncertainty  = -min_logprob
entropy uncertainty      = entropy
margin uncertainty       = -margin
```

Higher uncertainty should correspond to a greater chance that the hop is wrong.

### Error labels

For AUROC/AUPRC:

```text
correct hop   = 0
incorrect hop = 1
```

`needs_review` hops are excluded from the automatic metric calculation.

### How AUROC is calculated

For each confidence signal:

```python
AUROC = roc_auc_score(error_label, uncertainty_score)
```

Intuitively, AUROC asks:

> If we randomly choose one incorrect hop and one correct hop, how often does the uncertainty score rank the incorrect hop as more uncertain?

- `0.50` ≈ random
- `1.00` = perfect ranking

### How AUPRC is calculated

AUPRC summarizes the precision-recall curve where the positive class is **incorrect hop**.

It asks how well uncertainty retrieves actual errors across different decision thresholds.

---

## Test 1 results

Model:

```text
Qwen/Qwen3-1.7B
```

Data:

```text
Questions:             60
Total hops:           210
Automatically scored: 201
Manual-review hops:     9
Hop accuracy:          53.7%
Baseline final accuracy: 46.3%
```

### Confidence → error detection

| Signal | AUROC | AUPRC |
|---|---:|---:|
| Mean log-probability | **0.730** | 0.679 |
| Minimum log-probability | **0.736** | 0.697 |
| Entropy | **0.732** | 0.687 |
| Margin | **0.672** | 0.635 |

### Interpretation

Raw white-box confidence contains a **moderate but imperfect error-detection signal**.

This establishes that confidence is useful enough to investigate further, but it does not yet tell us whether confidence identifies the error that matters most downstream.

---

# Part II — Original repair-value pilot

## `02_repair_value.py`

### Purpose

Measures how useful it is to repair each clearly wrong non-terminal hop.

### Main intervention

For each eligible wrong hop:

```text
original trace
      ↓
replace one wrong hop with its MuSiQue gold answer
      ↓
regenerate dependency descendants
      ↓
regenerate final answer
      ↓
measure change in probability of the gold final answer
```

### Repair gain

The primary continuous metric is:

```text
repair_gain =
    mean log P(gold final | repaired propagated trace)
  - mean log P(gold final | original trace)
```

Interpretation:

```text
repair_gain > 0  → repair made the correct final answer more likely
repair_gain ≈ 0  → little effect
repair_gain < 0  → repair made the correct final answer less likely
```

A frozen repair control is also recorded by replacing the hop but holding later intermediate answers fixed.

---

## Original Test 2 results

```text
Baseline-failed questions rescored: 29
Repair candidates:                  30
Questions with a repair candidate: 21
Questions with >=2 candidates:      8

Mean propagated repair gain:       -0.483
Median propagated repair gain:     +0.079
Mean absolute propagated gain:      1.308

Mean within-question best-vs-worst
repair spread (2+ candidates):      2.132

Final-answer rescue rate:           1 / 30 = 3.3%
```

### Interpretation

Different wrong hops clearly have different measured repair effects.

That supports the central motivation:

> **being wrong is not the same as being the most useful error to repair.**

---

## `03_policy_analysis.py`

### Purpose

Among questions with multiple wrong repair candidates, asks whether confidence chooses the candidate with the highest measured repair gain.

### Policy metrics

For each question:

- **confidence top-1 hit**: did the most uncertain candidate have maximum repair gain?
- **random expected hit**: expected probability of choosing a max-gain candidate uniformly at random
- **earliest wrong baseline**
- **latest wrong baseline**
- **pairwise ranking accuracy**
- **repair regret**
- **normalized regret**
- **rescue rate**
- bootstrap confidence intervals

### Original Test 3 result

Only **8 questions** had multiple eligible candidates, so this was a debugging/pilot result.

For mean log-probability:

```text
Confidence top-1:       75.0%
Random expected:        47.9%
Earliest wrong:          0.0%
Latest wrong:           87.5%
Pairwise accuracy:      80.0%
```

This exposed an important possible **hop-position effect**.

---

# Part III — sequential confidence experiment

The next experiment was added after discussion with Xiang.

The main change is that hops are still queried **individually**, but references are now resolved using **prior model answers**, not gold answers.

This creates a real sequential model trajectory.

---

# `06_sequential_confidence.py`

## Purpose

Tests confidence under sequential reasoning and adds:

- raw white-box confidence
- within-trace normalization
- verbalized confidence
- first-error localization
- any-error localization
- sequential propagation of earlier mistakes

## Sequential design

For a three-hop question:

```text
Call 1:
H1 → model answer + white-box confidence

Call 2:
H2 uses MODEL H1 when resolving #1
H2 → model answer + white-box confidence

Call 3:
H3 uses current MODEL trajectory
H3 → model answer + white-box confidence

Then:
Final answer generated from the model-created reasoning state
```

Each hop is still a separate model call.

---

## Verbalized confidence

After generating the hop answer, the script makes a separate confidence query asking Qwen to estimate the probability that its answer is correct.

The model returns an integer:

```text
0–100
```

This is intentionally a **separate call** so adding verbalized confidence does not change the original hop answer generation prompt.

---

# Within-trace normalization

For each question, confidence scores are also normalized across that question's hops.

The experiment records:

- centered uncertainty
- z-scored uncertainty
- min-max normalized uncertainty

Important:

> Monotonic within-trace normalization cannot change the rank of hops inside the same question.

For example, if Hop 2 is the most uncertain raw score, it remains the most uncertain after ordinary z-scoring.

Therefore, normalization is mainly useful for testing whether **pooled cross-question error detection** improves after removing question-level confidence offsets.

---

# Sequential correctness and downstream corruption

Sequential reasoning introduces an important issue.

Example:

```text
Gold H1 = A
Model H1 = B   ← wrong

Gold H2 asks about A
Sequential model H2 now asks about B
```

If H2's answer differs from the benchmark H2 answer, that does **not necessarily mean H2 made a new independent reasoning error**. It may simply be answering a different question because H1 changed the trajectory.

Therefore the experiment stores whether:

```text
referenced_parents_benchmark_correct == True
```

A hop with benchmark-correct referenced parents is cleaner for local error analysis.

---

# `06b_analyze_sequential_confidence.py`

This is an **analysis-only** script.

It does not load Qwen or regenerate any model output.

It was added to:

1. recompute error-detection metrics on hops whose referenced parents were benchmark-correct
2. handle verbalized-confidence ties fairly
3. report verbal confidence score distribution

---

# Sequential confidence results

## All sequential hops

```text
Questions:                  60
Total hops:                210
Automatically scored:      203
Needs review:                7
Verbal parse failures:       0

Benchmark hop accuracy:     41.4%
Sequential final accuracy:  25.0%
```

The accuracy drop relative to the isolated-hop baseline is expected because upstream model errors are now allowed to propagate.

### Raw pooled error detection on all sequential hops

| Signal | AUROC | AUPRC |
|---|---:|---:|
| Mean log-probability | **0.712** | 0.753 |
| Minimum log-probability | **0.716** | 0.771 |
| Entropy | **0.717** | 0.765 |
| Margin | **0.659** | 0.722 |
| Verbalized confidence | **0.626** | 0.669 |

---

# Clean local-error subset

For local benchmark error analysis, we also restricted to hops whose referenced parents were benchmark-correct.

```text
Valid-parent hops:           142
Scored valid-parent hops:    140
Valid-parent hop accuracy:   54.3%
```

### Raw confidence on valid-parent hops

| Signal | AUROC | AUPRC |
|---|---:|---:|
| Mean log-probability | **0.735** | 0.676 |
| Minimum log-probability | **0.731** | 0.691 |
| Entropy | **0.739** | 0.690 |
| Margin | **0.674** | 0.638 |
| Verbalized confidence | **0.613** | 0.529 |

### Main observation

On clean local steps, raw token-level confidence remains around the original **~0.73 AUROC** range.

This suggests that much of the apparent degradation in the full sequential trajectory comes from downstream states whose benchmark meaning has already changed because of upstream errors.

---

# Did within-trace normalization help?

No, not in this Qwen3-1.7B pilot.

Example for mean log-probability on valid-parent hops:

```text
Raw AUROC:       0.735
Centered AUROC:  0.571
Z-score AUROC:   0.596
Min-max AUROC:   0.626
```

For all sequential hops:

```text
Raw mean-logprob AUROC:      0.712
Z-normalized AUROC:          0.584
```

### Interpretation

The hypothesis that within-trace normalization might improve error detection was **not supported in this pilot**.

Raw confidence was substantially better.

This should still be tested on more questions and other models before drawing a general conclusion.

---

# Within-trace error localization

For each question containing at least one clear error, the script asks:

1. Is the most uncertain hop the **first wrong hop**?
2. Is the most uncertain hop **any wrong hop**?
3. Across wrong-vs-correct pairs inside a trace, is the wrong hop more uncertain?

## Mean log-probability

```text
Questions with clear error:              53
Questions with wrong + correct hops:     42

First-error top-1:                       35.8%
Random expected first-error top-1:       30.5%

Any-error top-1:                         79.2%
Random expected any-error top-1:         66.2%

Pairwise wrong-vs-correct accuracy:      64.1%
Wrong/correct pairs:                     117
```

### Interpretation

Mean log-probability appears more useful for identifying **some erroneous state** than for locating the **original first divergence**.

This is important because later hops can be downstream consequences of an earlier error.

---

# Verbalized confidence result

Qwen's verbal confidence was extremely coarse.

Across 210 hops, only **9 unique confidence values** were produced.

Most common values:

```text
40 → 112 hops
50 →  50 hops
45 →  29 hops
```

Because many hops tie, top-1 evaluation must be tie-aware.

The corrected analysis uses **expected accuracy under uniform tie-breaking** instead of arbitrarily picking the first tied hop.

### Tie-aware verbal confidence localization

```text
First-error top-1:                   31.0%
Random expected:                     30.5%

Any-error top-1:                     69.3%
Random expected:                     66.2%

Pairwise wrong-vs-correct accuracy:  56.4%
```

### Interpretation

In this pilot, verbalized confidence is much weaker than raw token-level confidence and is approximately random for first-error localization.

---

# Part IV — Sequential repair experiment

# `07_sequential_repair_v2.py`

**This corrected v2 file is the one used for the reported sequential repair results.**

The earlier `07_sequential_repair.py` is superseded for this analysis.

## Candidate eligibility

We begin with wrong, non-terminal hops on trajectories where the final answer was wrong.

However, we exclude downstream-corrupted hops by default.

A repair candidate must have:

```text
label == incorrect
non-terminal hop
baseline final answer == incorrect
referenced parents benchmark-correct
```

Why?

If an upstream error changes the downstream question, the MuSiQue benchmark gold answer may no longer be a valid answer to the model-resolved downstream question.

Injecting that benchmark gold answer would not be a clean local repair.

---

## Sequential repair intervention

For each eligible candidate:

```text
original sequential trace
       ↓
replace one wrong hop with the MuSiQue gold answer
       ↓
regenerate all dependency descendants
       ↓
regenerate final answer
       ↓
score gold final answer
```

Repair gain:

```text
repair_gain =
    mean log P(gold final | repaired sequential trace)
  - mean log P(gold final | original sequential trace)
```

---

# Sequential repair results

Before the parent-validity filter:

```text
Wrong non-terminal candidates: 58
```

Excluded because their referenced parents were benchmark-wrong:

```text
Skipped downstream-corrupted: 14
```

Final repair set:

```text
Valid repair candidates:            44
Questions with repair candidate:    36
Questions with >=2 candidates:       8

Candidate distribution:
28 questions → 1 candidate
 8 questions → 2 candidates
```

### Repair effect

```text
Mean repair gain:                  +1.638
Median repair gain:                +0.288
Mean absolute repair gain:          2.488

Mean within-question gain spread
for 2+ candidate questions:         1.769
```

### Final-answer rescue

```text
Repair rescue rate: 7 / 44 = 15.9%
```

Two repaired final answers were placed in the manual-review queue.

### Interpretation

Repairing a benchmark-valid wrong step often changes downstream behavior substantially.

The non-zero within-question gain spread again confirms that different wrong errors can have different repair value.

---

# Part V — Which confidence signal identifies the highest-value repair?

# `08_sequential_policy_analysis_v2.py`

**This corrected v2 file is the one used for the reported sequential policy results.**

The earlier `08_sequential_policy_analysis.py` is superseded for this analysis.

This script is analysis-only and does not load Qwen.

---

# Repair-policy metric definitions

Only questions with **2 or more eligible repair candidates** can test repair prioritization.

Current sample:

```text
Eligible questions: 8
Candidates per eligible question: 2
```

This is a very small pilot sample.

---

## Confidence top-1 repair hit

For each question:

```text
confidence-selected candidate =
candidate with maximum uncertainty

oracle candidate =
candidate with maximum measured repair_gain
```

Then:

```text
confidence_top1_hit = 1
```

if the confidence-selected candidate is also an oracle best-repair candidate.

Otherwise:

```text
confidence_top1_hit = 0
```

For ties in confidence, the corrected script uses the **expected result under uniform tie-breaking**.

---

## Candidate-adjusted random baseline

For a question with `N` repair candidates and `K` tied best-gain candidates:

```text
random_expected_top1 = K / N
```

All 8 current questions have 2 candidates and one best candidate, so random expectation is:

```text
1 / 2 = 50%
```

---

## Pairwise ranking accuracy

For candidate pairs inside the same question:

> Does the confidence signal order the two candidates in the same direction as measured repair gain?

A confidence tie counts as `0.5`.

---

## Repair regret

For each question:

```text
regret =
best repair gain
-
gain of confidence-selected repair
```

Lower is better.

If confidence chooses the oracle repair:

```text
regret = 0
```

---

## Normalized regret

```text
normalized_regret =
regret
/
(best gain - worst gain)
```

This makes regret more comparable across questions whose repair-gain scales differ.

---

## Mean confidence-selected gain

Average repair gain obtained by following the confidence policy.

Compared with:

```text
mean oracle gain
```

which is the average gain obtained by always choosing the best candidate.

---

# Sequential repair-policy results

## Summary

| Signal | Best-repair top-1 | Random | Pairwise |
|---|---:|---:|---:|
| Mean log-probability | **87.5%** | 50.0% | **87.5%** |
| Minimum log-probability | 62.5% | 50.0% | 62.5% |
| Entropy | 75.0% | 50.0% | 75.0% |
| Margin | 75.0% | 50.0% | 75.0% |
| Verbalized confidence | 56.25% | 50.0% | 56.25% |

### Mean log-probability details

```text
Eligible questions:                 8
Confidence top-1 hit:              87.5%
Random expected:                   50.0%
Confidence - random:               +37.5 percentage points

Earliest wrong baseline:            0.0%
Latest wrong baseline:            100.0%

Pairwise ranking accuracy:         87.5%

Mean confidence regret:             0.546
Mean normalized regret:             0.125

Mean confidence-selected gain:     +0.304
Mean oracle gain:                  +0.850

Confidence-selected rescue rate:    0.0%
Oracle one-repair rescue rate:     12.5%
```

The 95% bootstrap interval for mean-logprob confidence-minus-random in this very small sample was approximately:

```text
[0.125, 0.500]
```

This should still be interpreted cautiously because only 8 questions are eligible.

---

# The strongest structural result: latest-hop baseline

Across all 8 multi-error questions:

```text
Earliest wrong hop was best repair: 0 / 8
Latest wrong hop was best repair:   8 / 8
```

So:

```text
latest wrong baseline = 100%
```

This is stronger than any confidence signal.

### Interpretation

The current pilot does **not** establish that confidence independently understands which error matters most.

A possible explanation is structural:

```text
repair early hop
→ regenerate many descendants
→ more opportunities for new trajectory errors

repair late hop
→ regenerate fewer descendants
→ less trajectory disturbance
```

Therefore a key next experiment is to determine whether confidence contains repair-value information **beyond hop position**.

---

# Raw vs. normalized confidence for repair ranking

Raw and z-normalized confidence produce the same within-question top-1 rankings in this experiment.

Examples:

```text
Mean logprob:
raw top-1 = 87.5%
z top-1   = 87.5%

Entropy:
raw top-1 = 75.0%
z top-1   = 75.0%
```

This is expected because z-scoring within one question is monotonic and preserves ordering.

Normalization is therefore more relevant to pooled cross-question detection/calibration than to within-question argmax repair selection.

---

# Verbalized confidence for repair selection

Tie-aware verbalized confidence:

```text
Best-repair hit:               56.25%
Random expected:               50.0%
Pairwise ranking:              56.25%

Mean selected repair gain:    -0.031
Mean oracle repair gain:      +0.850
```

### Interpretation

Verbalized confidence is currently a weak repair-prioritization signal for Qwen3-1.7B.

Raw white-box token confidence is substantially stronger.

---

# Main findings so far

## 1. Raw token confidence does contain error information

In the clean valid-parent subset:

```text
Mean logprob AUROC: 0.735
Entropy AUROC:      0.739
```

This is close to the original isolated-hop baseline of approximately `0.73`.

---

## 2. Sequential error propagation changes the meaning of downstream "errors"

Later benchmark mismatches cannot always be treated as independent mistakes if an upstream wrong answer changed the downstream question.

This motivated the valid-parent filter.

---

## 3. Within-trace normalization did not help error detection

For this Qwen3-1.7B pilot, normalized confidence performed worse than raw confidence in pooled error detection.

---

## 4. Verbalized confidence was weak and highly discretized

Qwen produced only 9 distinct confidence values over 210 hops.

Tie-aware verbal confidence was approximately random for first-error localization.

---

## 5. Confidence is better at finding an error than the first source of the error

Mean logprob:

```text
Any-error localization:   79.2%
Random expected:          66.2%

First-error localization: 35.8%
Random expected:          30.5%
```

This suggests confidence may identify corrupted states better than it identifies the original point of divergence.

---

## 6. Different errors have different repair value

Across sequential valid repair candidates:

```text
Mean absolute repair gain: 2.488
Mean within-question spread: 1.769
```

So repair choice matters.

---

## 7. Mean logprob showed promising repair ranking in the tiny sample

```text
Mean-logprob best-repair hit: 87.5%
Random expected:              50.0%
```

However:

```text
Latest wrong hop baseline:   100.0%
```

So position is currently an even stronger predictor.

---

# What we can and cannot conclude

## Supported by this pilot

We can say:

- white-box confidence contains moderate information about local reasoning correctness
- sequential propagation creates downstream states that require careful labeling
- raw confidence outperformed within-trace normalized confidence in this model
- verbalized confidence was weaker than token-level confidence
- wrong hops differ in measured repair value
- confidence may contain some repair-ranking signal
- hop position is an extremely strong competing baseline

## Not supported yet

We **cannot** currently claim:

- that these findings generalize to LLMs
- that confidence reliably identifies the highest-value repair
- that mean logprob is universally the best confidence signal
- that normalization is generally harmful
- that verbalized confidence is generally useless
- that latest-hop dominance is a universal property
- that the current repair-ranking results are statistically stable

The largest reason is sample size:

```text
Only 8 questions currently have >=2 valid repair candidates.
```

Also, only one subject model has been tested.

---

# Model coverage

Current subject model:

```text
Qwen/Qwen3-1.7B
```

No current claim should be generalized beyond this model.

Likely future models:

```text
Qwen3-4B
Qwen3-8B
at least one non-Qwen model family
```

Stronger models may produce fewer multi-error trajectories, so more benchmark questions may be needed.

---

# Next experiments

## 1. Frozen-repair diagnostic on the same sequential candidates

The current sequential repair intervention regenerates descendants.

That may structurally favor later errors.

Next compare:

```text
PROPAGATED:
repair Hi
→ regenerate descendants
→ score final

FROZEN:
repair Hi
→ keep later intermediate states fixed
→ score final
```

Then compare latest-hop dominance under both conditions.

If latest-hop dominance largely disappears under frozen repair, the effect is likely caused by regeneration depth / trajectory instability.

If it remains strong under frozen repair, later errors may have greater direct influence on the final state.

---

## 2. Scale the sample

We need substantially more than 8 multi-error questions.

The immediate goal should be enough eligible questions to obtain stable confidence intervals and position-controlled analyses.

---

## 3. Test additional model sizes and families

Repeat the same fixed methodology on:

- Qwen3-4B
- Qwen3-8B
- another model family

---

## 4. Test whether confidence adds information beyond position

A larger study should compare models such as:

```text
repair value ~ hop position

repair value ~ confidence

repair value ~ hop position + confidence
```

The important question is whether confidence adds predictive value once position is controlled.

---

## 5. Confidence perturbation / independence tests

For verbalized confidence, future tests can keep the reasoning answer fixed but alter only a confidence value supplied in the reasoning state.

Example:

```text
same H2 answer
confidence 60 → confidence 66
```

Then test whether downstream confidence changes despite identical reasoning content.

This probes whether verbalized confidence is stable or merely anchors later self-reports.

---

## 6. Tree / branching reasoning

After the chain experiments are stable, extend to branching reasoning where one branch can be perturbed while unrelated branches should remain unaffected.

This gives a cleaner test of confidence independence.

---

# Exact run order for the reported results

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Original isolated-hop baseline

```bash
python 01_confidence_correctness.py \
  --model Qwen/Qwen3-1.7B \
  --n-questions 60 \
  --batch-size 1 \
  --score-batch-size 1 \
  --run-dir outputs/qwen17b_seed42
```

Then:

```bash
python 02_repair_value.py \
  --run-dir outputs/qwen17b_seed42 \
  --generate-batch-size 1 \
  --score-batch-size 1
```

Then:

```bash
python 03_policy_analysis.py \
  --run-dir outputs/qwen17b_seed42
```

---

## Sequential follow-up

Use the exact same 60-question selection:

```bash
python 06_sequential_confidence.py \
  --model Qwen/Qwen3-1.7B \
  --selection-file outputs/qwen17b_seed42/selection.json \
  --run-dir outputs/qwen17b_sequential_seed42
```

Then analysis-only correction / valid-parent breakdown:

```bash
python 06b_analyze_sequential_confidence.py \
  --run-dir outputs/qwen17b_sequential_seed42
```

Then the **corrected v2 sequential repair script**:

```bash
python 07_sequential_repair_v2.py \
  --run-dir outputs/qwen17b_sequential_seed42
```

Then the **corrected v2 policy analysis**:

```bash
python 08_sequential_policy_analysis_v2.py \
  --run-dir outputs/qwen17b_sequential_seed42
```

---

# Important file-status note

The reported sequential repair/policy results use:

```text
07_sequential_repair_v2.py
08_sequential_policy_analysis_v2.py
```

not the earlier versions without `_v2`.

The v2 repair script excludes downstream-corrupted candidates whose referenced parents were already benchmark-wrong.

The v2 policy script handles tied confidence values using expected uniform tie-breaking.

---

# Important output files

## Original pilot

```text
outputs/qwen17b_seed42/
├── selection.json
├── 01_hops.csv
├── 01_questions.csv
├── 01_review_queue.csv
├── 01_summary.json
├── 01_trace.jsonl
├── 02_baseline_rescored.csv
├── 02_repairs.csv
├── 02_summary.json
├── 02_propagated_repairs.jsonl
├── 03_policy_summary.json
└── 03_question_policy*.csv
```

## Sequential follow-up

```text
outputs/qwen17b_sequential_seed42/
├── 06_sequential_hops.csv
├── 06_sequential_questions.csv
├── 06_sequential_trace.jsonl
├── 06_review_queue.csv
├── 06_summary.json
├── 06b_analysis_summary.json
├── 07_repairs.csv
├── 07_propagated_repairs.jsonl
├── 07_review_queue.csv
├── 07_summary.json
└── 08_policy_summary.json
```

---

# One-paragraph summary for collaborators

We first established an isolated-hop baseline on 60 MuSiQue questions with Qwen3-1.7B, where token-level confidence detected incorrect hops at roughly 0.73 AUROC. We then switched to sequential hop generation, where each hop is still queried separately but MuSiQue references are resolved using prior model answers. On benchmark-valid local steps, raw token confidence remained around 0.73 AUROC, while within-trace normalization reduced performance and verbalized confidence was substantially weaker and heavily tied. Confidence was better at selecting any erroneous state than locating the first divergence. We then gold-repaired valid wrong non-terminal hops and regenerated their dependency descendants. Across 44 repair candidates, mean repair gain was +1.638 and 7/44 repairs rescued the failed final answer. Only 8 questions had multiple valid repair candidates; on those, mean-logprob uncertainty selected the higher-value repair in 7/8 cases, but the latest-wrong-hop baseline selected it in 8/8. Therefore the current pilot suggests a possible confidence-to-repair signal, but hop position is a stronger competing explanation and must be controlled before drawing conclusions.

---

# Bottom line

The current experiments validate the research pipeline but do **not** answer the final research question yet.

The strongest current takeaway is:

> **Qwen3-1.7B's token-level confidence contains useful error information, but identifying the most valuable repair is confounded by reasoning-chain structure—especially hop position.**

The next priority is to determine whether confidence provides repair-value information **beyond position**, then scale the experiment across more questions and models.
