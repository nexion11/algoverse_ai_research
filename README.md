# Confidence, Error Influence, and Repair Value in Multi-Hop QA

Research pilot on whether a language model's step-level confidence can tell you
**which reasoning errors are worth fixing**.

All experiments use **Qwen3-1.7B** on **MuSiQue** (60 questions: 30 three-hop,
30 four-hop, seed 42), Hugging Face Transformers on Apple MPS, greedy decoding.
These are pilot / methodology results, not final paper results.

---

## The question

A multi-hop question is answered one step at a time. Some steps come out wrong.
But not every wrong step matters — some errors are **load-bearing** and change
the final answer, others are **inert** and change nothing.

> Can step-level confidence tell you which errors are the load-bearing ones?

---

## The two headline results

### 1. No — and not for lack of trying

Confidence detects that a step is **wrong** at AUROC ≈ 0.70. It is at **chance**
on whether that step **matters**.

| | AUROC |
|---|---|
| is this step wrong? | **0.702** |
| is this step load-bearing? | **0.517** |

Every mechanism for making the model use confidence came back null:

| what was tried | result |
|---|---|
| supply per-step confidence in the prompt | Δ accuracy **0.000** vs shuffled |
| teach it by demonstration (in-context learning) | Δ accuracy **0.000** vs shuffled |
| flag one step as `UNRELIABLE` | flagging the right step = flagging a **random** step |
| tune the elicitation prompt (5 variants) | all **worse** than the original |
| aggregate confidence across steps (200 combinations) | no aggregation predicts repair value |

### 2. But the information is in the model — just not in its confidence

A linear probe on the hidden states of the **same forward pass** recovers it:

| target | best confidence signal | **hidden-state probe** |
|---|---|---|
| is this step wrong? | 0.702 | **0.807** |
| is this step load-bearing? | 0.550 — chance | **0.777** |

Permutation p = 0.005. Position and structure ruled out (all trivial features at
chance). Grouped cross-validation by question. Stable across regularization and
fold count (0.762–0.788).

> **The model represents which steps carry the answer. It never reports that in
> its confidence. So no amount of prompting can recover it — but a probe can.**

---

## Why the load-bearing / inert distinction is real

Removing a step and rescoring shows steps differ enormously in how much they
matter:

| | mean \|Δ logP(gold)\| | flips the final answer |
|---|---|---|
| inert third | 0.025 | 8% |
| load-bearing third | 1.134 | 28% |

A **45× spread**. Overall, removing a step changes the final answer only 14.7%
of the time — roughly 85% of reasoning steps are close to inert.

So the distinction the project set out to find does exist. The negative result is
about the confidence channel, not about the model's knowledge.

---

## Write-ups

| document | covers |
|---|---|
| **[README_confidence_influence.md](README_confidence_influence.md)** | The current round. Step influence, the hidden-state probe, in-context learning, elicitation tuning, aggregation, and the repair-policy metrics. Full method, results, validation and limitations. |
| [README_confidence_repair.md](README_confidence_repair.md) | The original pilot. Isolated-hop and sequential confidence baselines, gold-repair experiments, repair-value and repair-prioritisation analyses. |

---

## Layout

```
basic-tests/                 original isolated-hop pilot
basic-tests-updated/         sequential pilot + this round's experiments 11b-16
  common.py                  shared measurement layer: model loading, prompts,
                             token-level confidence, teacher-forced scoring, grading
confidence-experiments/      experiments 17-21; imports common.py rather than
                             forking it, so all numbers stay comparable
```

Every experiment writes CSV/JSONL alongside its summary, so each number traces
back to raw rows.

---

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r basic-tests-updated/requirements.txt

cd basic-tests-updated
python 12_confidence_aggregation_claude.py   --run-dir outputs/qwen17b_aggregation_claude
python 13_step_influence_claude.py           --run-dir outputs/qwen17b_influence_claude
python 13b_analyze_influence_claude.py       --run-dir outputs/qwen17b_influence_claude
python 14_verbal_elicitation_sweep_claude.py --run-dir outputs/qwen17b_elicitation_claude
python 15_policy_metrics_claude.py           --run-dir outputs/qwen17b_policy_claude
python 16_bake_confidence_claude.py          --run-dir outputs/qwen17b_bake_claude

cd ../confidence-experiments
python 18_icl_proper_claude.py        --run-dir outputs/icl_proper_claude
python 19_hidden_state_probe_claude.py --run-dir outputs/probe_claude
python 20_icl_variants_claude.py      --run-dir outputs/icl_variants_claude
```

Analysis-only scripts (12, 13b, 15) need no GPU. The rest run in 25–50 minutes
each on an M-series Mac.

---

## Superseded and unrun scripts

Kept deliberately, and labelled, because the failures are instructive:

- `17_icl_confidence_claude.py` — **superseded**. Its demonstrations were all
  explained by "copy the last step", so its null was uninterpretable. An ICL null
  is worthless without a positive control showing the demonstrations do anything.
  Replaced by `18`, which asserts that heuristic fails on its demo set.
- The chain-of-thought arm of `20` — **invalid**. Its rationales came from two
  string templates and the model copied the wording rather than the behaviour.
  The salient-flag arms of `20` are valid and are the sharpest null in the
  repository.
- `21_cot_icl_proper_claude.py` — **written, never run.** Rebuilds the CoT arm
  properly; one known issue remains (the model reads confidence values as step
  numbers).
- `11b_sequential_confidence_intervention_claude.py` — **written, never run.**

---

## Limitations

One model, one dataset, one seed. 56–60 questions with only 14 correct final
answers, so intervals are wide throughout and are reported rather than elided.
Influence is measured with descendants frozen, so it captures how much the final
read-out depends on a step, not the full effect that step would have in a live
rollout. The probe is trained and evaluated on the same 60 questions — grouped CV
prevents leakage, but out-of-sample replication is the top next step.

Nothing here should be generalised to other models or scales.

---

## Next steps

1. **Replicate the probe on fresh questions.** It is the result worth building
   on and it needs out-of-sample confirmation.
2. **De-duplicate the question selection** on shared sub-question chains. The
   current 60 contain a 4-way near-duplicate cluster (`88460_30152_20999`) that
   inflated an earlier repair-prioritisation result.
3. **Scale influence measurement, not repair ranking** — ~2.5 scored hops per
   question instead of a candidate pair on one question in seven.
4. **Probe-guided repair.** Use the probe to choose which step to re-check,
   against a random-step control. The first selection rule in the project with a
   signal behind it.
5. **Larger models and a non-Qwen family**, at which point vLLM on rented CUDA is
   worthwhile — but verify full-vocabulary entropy first, since vLLM returns only
   top-k logprobs.
