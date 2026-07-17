# Entropy-gain selection after the random-resampling result: defensibility, recovery paths, and experiment plan

*Prepared 2026-07-17. This report updates `docs/entropy-gain-selection-paper-strategy.md` after the random-resampled baseline became available. It combines the team-reported headline results, a code/configuration audit, offline analysis of the stored per-epoch selection artifacts, and a targeted prior-art review current to the preparation date.*

---

## Executive conclusion

The new result is important, but the most obvious pivot is not novel enough:

> **“Select a fresh random 10% subset every epoch; random rotation beats informed selection.”**

That algorithm and much of that conclusion already exist. The current random baseline is essentially **Repeated Sampling of Random Subsets (RS2)**, introduced and evaluated against many pruning and distillation methods at ICLR 2024. Recent LLM fine-tuning work also reports that random selection is difficult to beat and attributes the result to diversity or coverage. A paper whose main contribution is only “RS2 also works for reasoning distillation” would be a domain replication, not a strong ICLR novelty claim.

The internal logs do, however, expose a more specific and potentially publishable phenomenon:

> **The online entropy score ceases to measure what the selector assumes it measures. Early in training it identifies student errors; late in training its error-prediction AUC falls to chance, and the highest-entropy subset becomes more solved than the pool. At the same time, hard top-k selection creates large exposure inequality and persistent coverage debt.**

This changes the central scientific question from:

> “Is entropy gain a better selector than random?”

to:

> **“When, if ever, does a model-dependent selection score predict downstream training utility in reasoning distillation, and how should a selector react when that relationship changes?”**

### Recommended strategy

1. **Do not sell random rotation as a new method.** Rename the baseline RS2 and cite it prominently.
2. **Run a short matched-control sprint before choosing a paper.** The current random-vs-full comparison is confounded; random-vs-entropy is cleaner but still conflates subset membership with within-epoch ordering.
3. **Use the sprint to choose between two viable branches:**
   - a broad, compute-matched **reasoning-distillation selection benchmark / phase-diagram paper**; or
   - a higher-risk **reliability-gated selection method** that mixes score-based exploitation with uniform coverage only while the score demonstrably predicts utility.
4. **Abandon the original entropy-gain headline unless a corrected measurement or coverage-aware variant clearly beats RS2 across seeds.** Similar accuracy to student entropy is not enough; the proxy must provide a statistically supported marginal benefit.
5. **Treat a negative-only result realistically.** With multiple datasets, recent strong baselines, seeds, compute accounting, and the signal-collapse diagnosis, it can become a strong empirical paper. On one dataset with the current heuristics, it is more naturally a workshop or Findings-style paper.

---

## 1. Evidence inventory and confidence labels

This report separates facts by evidence level.

- **Observed:** computed directly from code, committed summaries, or stored per-epoch parquets in the workspace.
- **Team-reported:** provided in the result message, but not every corresponding evaluation summary is currently present in the repository.
- **Inferred:** a mechanism consistent with observed evidence but not yet isolated experimentally.
- **Proposed:** a future experiment, claim, or method.

### 1.1 Team-reported headline results

| Student | Random 10% resampled/epoch | Entropy gain | Student entropy | Full-corpus distillation |
|---|---:|---:|---:|---:|
| Qwen | **0.57** | 0.55 | 0.55 | 0.52 |
| Phi | **0.67** | 0.64 | — | 0.63 |
| Llama | **0.52** | 0.51 | — | 0.48 |

Immediate pattern:

- Random exceeds entropy gain for all three students.
- Random exceeds the reported full-corpus result for all three students.
- On Qwen, the proxy-referenced entropy gain and plain student entropy have the same rounded peak.

This pattern is serious enough to change experimental priorities. It is not yet sufficient for a final statistical or causal claim.

### 1.2 What is present in the workspace

- Qwen student-entropy evaluation summaries through epoch 100.
- Per-epoch Qwen entropy-gain and student-entropy scoring parquets through epoch 99.
- Partial per-epoch Llama entropy-gain artifacts.
- Older Qwen entropy-gain corrected/direct runs.
- Full-corpus evaluation summaries for multiple models and trace variants.
- The new random experiment scripts, but not all reported random evaluation artifacts.

Consequently, the detailed mechanism analysis below is strongest for Qwen. It must be reproduced for Phi and Llama before it becomes a cross-model claim.

---

## 2. What the current numbers do and do not establish

### 2.1 What they support now

The results support the following provisional statements:

1. **RS2-style random resampling is a load-bearing baseline.** Any entropy-based selector must beat it, not merely a fixed random subset or full-corpus run.
2. **The current entropy-gain implementation does not show a downstream advantage over student entropy on Qwen.** The proxy term has no demonstrated marginal value.
3. **Hard uncertainty selection is not robustly superior to uniform coverage in the current setting.** The direction is consistent across three model families.
4. **The original “entropy gain beats full distillation” interpretation is no longer credible without a matched random control.** The earlier strategy document correctly identified this risk.

### 2.2 What they do not establish

They do not yet establish that:

- random resampling universally beats data selection;
- random resampling causally beats full-corpus training because of regularization;
- entropy gain is worse than random with statistical significance for every model;
- entropy selection fails because it selects longer or noisier traces;
- rotation is a new method or insight;
- 10% per epoch means only 10% of teacher traces are needed over the run;
- the same ordering holds on open-ended mathematical reasoning, other datasets, larger students, or non-answer-leaked traces;
- current published reasoning selectors such as LARK or dynamic compatibility methods lose to RS2.

### 2.3 Approximate uncertainty from the test set alone

The evaluation set contains 2,406 questions. Ignoring training-seed variability and pairing, the approximate conservative unpaired 95% intervals for the reported random-minus-entropy-gain differences are:

| Student | Difference | Approximate 95% interval |
|---|---:|---:|
| Qwen | +2.0 points | [-0.8, +4.8] points |
| Phi | +3.0 points | [+0.3, +5.7] points |
| Llama | +1.0 point | [-1.8, +3.8] points |

These are only test-sampling intervals. They omit training variance, random subset variance, checkpoint-selection bias, and hyperparameter variance.

The fact that random wins on three of three students is suggestive but not independently decisive. A one-sided sign test on three model-level comparisons gives `p = 0.125`. The models also share the same data and evaluation items, so treating them as fully independent replicates would overstate the evidence.

### 2.4 The checkpoint-selection problem

The current evaluator scores every saved checkpoint on the test set. Reporting the best test checkpoint turns the test set into a validation set and inflates all peaks, potentially by different amounts across learning curves.

The corrected protocol is:

1. select the checkpoint using a validation split;
2. evaluate the selected checkpoint once on the test set;
3. retain all-checkpoint test curves only as clearly labeled diagnostic plots, not as the source of headline numbers.

---

## 3. Matchedness and implementation audit

### 3.1 Random versus entropy gain: mostly matched, but not perfectly isolated

The new random and entropy-gain scripts share `src/experiments/distillation_by_metrics/mmlu/shared.py`. They use:

- the same corrected-answer trace file;
- the same 1024 reasoning examples per epoch;
- the same 256 single-token auxiliary examples per epoch;
- the same model-specific LoRA configuration;
- the same epoch/save schedule;
- the same evaluation code and generation caps;
- the same `ResamplingTrainer` infrastructure.

This makes random versus entropy gain the most interpretable comparison currently available.

However, it still changes more than the selection membership:

- `BaseDatasetSampler` sorts selected examples by increasing score before training. Entropy therefore creates a deterministic easy-to-hard order inside the selected subset.
- Random assigns random scores and is then sorted by those random scores, producing a random order.

Thus the current comparison is:

> random membership + random order

versus

> score-based membership + score-sorted curriculum order.

The paper must either shuffle every selected subset identically or cross the factors explicitly:

| Membership | Order |
|---|---|
| random | shuffled |
| random | sorted by an unrelated fixed/random key |
| entropy | shuffled |
| entropy | easy-to-hard |

At minimum, entropy-selected data should be shuffled before drawing a conclusion about membership.

### 3.2 Random versus full corpus: not currently matched

The random/entropy and full-corpus pipelines differ in several material ways.

| Axis | Resampling experiments | Full-corpus experiments |
|---|---|---|
| Training objective | 1024 reasoning + 256 single-token responses | reasoning responses only |
| Packing | disabled in current shared config | enabled |
| Per-device batch | 2 | 1 packed block |
| Epoch definition | 1,280 selected rows | entire corpus / packed corpus |
| LR horizon | expressed in resampling epochs | expressed in full-corpus epochs |
| Evaluation caps | loop over 4096 and 2048 | default 8192 unless overridden |
| Duplicate objective views | top 256 also appear among top 1024 | absent |

Therefore the statement “10% random rotation beats full-corpus distillation” is not yet causally attributable to rotation or data reduction.

Possible alternative causes include:

- the single-token auxiliary objective improves answer formatting;
- packing changes token weighting or optimization;
- different total optimizer updates or supervised-token exposure;
- different LR decay in update/token units;
- evaluation-cap mismatch;
- full-corpus overtraining;
- test-selected checkpoints.

The full-corpus comparison must be rerun through the same training engine and objective mix, or the paper must describe it as an unmatched reference rather than a controlled baseline.

### 3.3 The random implementation pays an unnecessary scoring cost

`RandomEstimator` ignores model outputs, but `ComplexityEstimationRunner` still performs `model.generate(...)` for every pool item before assigning a Python random value. Thus the current random run artificially pays a full one-token forward pass over the pool every epoch.

This has two implications:

- for a controlled training comparison, random and entropy currently pay more similar online-loop overhead than a practical random baseline would;
- for an end-to-end efficiency claim, an optimized RS2 implementation should skip model scoring entirely and will be cheaper than the current code suggests.

Report both:

1. **measured current wall-clock**, including the intentionally matched but unnecessary pass; and
2. **practical RS2 wall-clock**, using direct random sampling with no model forward pass.

### 3.4 Seed plumbing is not ready for a multi-seed claim

`src/core/utils/seed.py` hard-codes seed 42 and seeds Python and PyTorch, but not NumPy. `BaseTrainingArgs` exposes `seed` and `data_seed`, yet the pre-trainer seed function does not accept them.

Before running seeds:

- make the experiment seed an explicit config value;
- seed Python, NumPy, PyTorch CPU, and all CUDA devices;
- propagate the seed to subset sampling, trainer initialization, data order, and any split generation;
- log all seeds in the output metadata;
- keep one shared seed set across methods for paired comparisons.

Without this fix, “three seeds” may still share important sources of randomness.

### 3.5 Answer-leaked traces remain a headline risk

The current matched metric experiments use corrected-answer traces generated with access to the gold answer. This keeps the random-vs-entropy comparison internally matched, but it weakens external validity and creates a hidden dependency in the headline result.

Recommended hierarchy:

1. headline: answer-hidden teacher generation, correctness filtered or rejection sampled;
2. ablation: corrected-answer/post-hoc-rationalized traces;
3. analysis: interaction between trace correctness and selection method.

---

## 4. Offline mechanism analysis from the existing Qwen trajectories

### 4.1 Selection-signal health collapses over training

For each epoch, the stored complexity parquet contains:

- current student entropy;
- proxy entropy;
- current direct-answer correctness;
- question identifier and metadata.

Using score as a predictor of the event “the current student answers incorrectly,” the following AUCs are obtained.

#### Plain student entropy

| Scoring epoch | Pool direct-answer accuracy | Accuracy among selected top 1024 | AUC: score predicts incorrect |
|---:|---:|---:|---:|
| 0 | 0.364 | 0.161 | **0.710** |
| 9 | 0.359 | 0.133 | **0.762** |
| 19 | 0.511 | 0.263 | **0.705** |
| 39 | 0.729 | 0.549 | **0.718** |
| 59 | 0.858 | 0.728 | 0.667 |
| 79 | 0.922 | 0.924 | 0.554 |
| 99 | 0.931 | **0.961** | **0.511** |

#### Entropy gain

| Scoring epoch | Pool direct-answer accuracy | Accuracy among selected top 1024 | AUC: score predicts incorrect |
|---:|---:|---:|---:|
| 0 | 0.363 | 0.192 | 0.530 |
| 9 | 0.439 | 0.259 | **0.689** |
| 19 | 0.552 | 0.421 | **0.679** |
| 39 | 0.710 | 0.547 | **0.656** |
| 59 | 0.824 | 0.719 | 0.563 |
| 79 | 0.885 | 0.862 | 0.507 |
| 99 | 0.897 | **0.904** | **0.495** |

Interpretation:

- early and middle training: student entropy can identify examples on which the student is currently wrong;
- late training: both scores approach random discrimination;
- by epoch 99, the selected subset is slightly more often correct than the full pool;
- entropy gain starts worse than student entropy at epoch 0, becomes useful during the middle regime, then also collapses.

This is stronger evidence than merely observing that random wins. It shows that the selector's assumed surrogate becomes invalid during the run.

### 4.2 Why low failure counts do not rescue the measurement

The online direct-answer parser has near-zero late failure rates in these Qwen runs because the training mix includes 256 single-token examples each epoch. Nevertheless, score AUC collapses.

Therefore the problem is not only unparseable `<think>` output. The entropy value can remain numerically measurable while ceasing to reflect answer uncertainty or training utility.

Likely contributors:

1. entropy is computed over the entire vocabulary, not the normalized answer-option distribution;
2. high tail mass or formatting alternatives can raise entropy while the argmax answer remains correct;
3. the direct-answer prompt is not the CoT training/evaluation regime;
4. the model is optimized on a long teacher trace, while the selector examines one answer token;
5. fine-tuning changes calibration and output sharpness over time;
6. error uncertainty and marginal training utility are different targets even when entropy is calibrated.

### 4.3 Coverage debt is large and persistent

The selected top-1024 question sets were reconstructed from every stored Qwen scoring parquet.

| Epochs completed | Expected RS2 unique coverage | Entropy-gain unique coverage | Student-entropy unique coverage |
|---:|---:|---:|---:|
| 1 | 10.6% | 10.6% | 10.6% |
| 5 | 43.0% | 30.4% | 30.3% |
| 10 | 67.5% | 39.5% | 40.9% |
| 20 | 89.5% | 55.3% | 53.9% |
| 40 | 98.9% | 71.3% | 68.6% |
| 60 | 99.9% | 78.8% | 78.2% |
| 80 | approximately 100% | 83.9% | 83.1% |
| 100 | approximately 100% | **84.7%** | **83.7%** |

For random sampling without replacement inside each epoch, the expected unique count after `E` independent epochs is:

```text
E[unique(E)] = N * (1 - (1 - k/N)^E)
```

with `N = 9626` and `k = 1024`.

Consequences:

- RS2 is not a “train on only 10% of the dataset” method over the whole run. It is a near-full-coverage stream after roughly 40 rounds.
- entropy selection rotates, but much more slowly and unevenly;
- the long-run teacher-trace saving from entropy gain is not 3x. At 100 rounds, only about 15% of questions remain unseen;
- because teacher traces were pre-generated in the present experiments, even that saving is hypothetical unless trace generation is moved on demand.

### 4.4 Exposure inequality is severe

Across 100 Qwen selection rounds:

| Diagnostic | Entropy gain | Student entropy |
|---|---:|---:|
| Never selected | 15.3% | 16.3% |
| Median selections among ever-selected items | 9 | 9 |
| 90th-percentile selections | 29 | 30 |
| Maximum selections for one item | 69 | 80 |
| Exposure Gini over all questions | 0.548 | 0.561 |

The 256 single-token examples are chosen from the same ranked pool and are contained within the top 1024, so the most highly ranked questions receive an additional objective view in the same epoch.

This produces two coupled biases:

1. repeated question exposure;
2. repeated exposure under both long-reasoning and single-token targets.

Random uses the same nested-objective design, but its long-run exposure is far closer to uniform.

### 4.5 Entropy gain and student entropy do not select the same data

Despite similar rounded Qwen accuracy, the mean Jaccard similarity between their epoch-matched top-1024 sets is only **0.110**. At epoch 0 it is 0.559; by epoch 99 it is approximately 0.105.

Within the entropy-gain trajectory, the mean Spearman correlation between student entropy and proxy entropy is about 0.475.

Therefore:

- the proxy term materially changes the selected questions;
- the equality in final accuracy is not explained by two nearly identical rankings;
- many materially different hard-example curricula may converge to the same performance ceiling because the dominant issue is coverage/utility rather than the exact difficulty ranking;
- “proxy adds nothing” should be stated as a downstream result, not as “proxy leaves the ranking unchanged.”

### 4.6 Category skew and trace length are secondary, not primary, explanations

For Qwen selection views:

- category-distribution total variation from the full pool is approximately 0.062 for entropy gain and 0.074 for student entropy;
- biology is undersampled and business is oversampled, but the skew is not extreme;
- mean teacher-trace token count is approximately 162 for the pool, 164 for entropy gain, and 161 for student entropy;
- selected trace lengths therefore do not support a simple “entropy chooses much longer traces” explanation.

The strongest currently observed mechanisms are:

1. score validity collapses;
2. exposure is highly unequal;
3. cumulative coverage is much lower than RS2;
4. the score optimizes current predictive difficulty rather than demonstrated downstream utility.

Noise concentration and category imbalance remain plausible secondary effects but must not be presented as established causes without direct tests.

---

## 5. Mechanistic interpretation

### 5.1 RS2 approximates ordinary uniform training with a different round structure

A fresh uniform subset each round yields approximately unbiased stochastic gradients from the full empirical distribution. Over many rounds, almost every item is seen and exposure approaches uniformity.

At a high level, RS2 combines:

- representative coverage;
- stochastic data-level regularization;
- lower work per nominal epoch;
- a long sequence of fresh batches;
- no score-estimation error;
- no selector-induced distribution shift.

It is not mysterious that this is hard to beat. A learned selector must estimate a sufficiently accurate utility advantage to compensate for the bias and variance it introduces.

### 5.2 Hard top-k selection is a fragile decision rule

Top-k transforms small score errors near the boundary into discrete inclusion/exclusion decisions. When score reliability decays:

- the selected distribution remains highly non-uniform;
- no mechanism pulls it back toward the empirical distribution;
- examples can be repeated because of persistent score idiosyncrasies;
- useful but moderate-score examples may remain unseen;
- the method continues exploiting even when its exploitation signal has become random.

This suggests that the relevant design axis is not only **which score**, but also **how strongly the score is allowed to distort uniform coverage**.

### 5.3 Predictive difficulty is not training utility

The current selector assumes:

```text
high current uncertainty -> large benefit from training on the teacher trace
```

That implication can fail because an uncertain item may be:

- noisy or ambiguous;
- already correct but poorly calibrated;
- difficult in a way the provided trace does not teach;
- out of distribution for the student's representational capacity;
- redundant with repeatedly selected items;
- high entropy because of output formatting rather than semantic uncertainty;
- locally learnable but harmful to validation/generalization;
- useful only early or only after prerequisite skills are learned.

A selector should target a utility such as short-horizon validation improvement, loss reduction, gradient alignment, or skill coverage—not current error alone.

### 5.4 Entropy gain does not currently justify the epistemic-minus-aleatoric story

The intended interpretation is:

```text
student entropy = reducible uncertainty + irreducible difficulty
proxy entropy   = irreducible difficulty
difference      = reducible uncertainty
```

The implementation weakens every equality:

- entropies come from different model families and tokenizers;
- raw full-vocabulary nats are subtracted directly;
- the proxy has its own epistemic errors;
- calibration differs across models and changes over training;
- the measured event is the next token, not the final reasoning outcome;
- clipping negative differences destroys magnitude information;
- “reducible uncertainty” is not shown to equal “benefit from this teacher trace.”

This story should not appear as theory unless the measurement is repaired and empirically linked to downstream utility.

### 5.5 Why the proxy can change the subset without changing accuracy

Several explanations are consistent with the low cross-method Jaccard and equal rounded accuracy:

1. both selectors choose different examples from a broad region of similarly suboptimal hard cases;
2. both lose enough coverage that the specific hard cases matter less than the common coverage deficit;
3. both scores become unreliable late, so their distinct curricula end in a similar plateau;
4. the auxiliary objective or optimization regime dominates the subtle selection difference;
5. evaluation noise masks a small true difference;
6. the selected traces have different immediate effects that wash out over long training.

Only seeds plus utility measurements can distinguish these explanations.

---

## 6. Prior-art collision: what is already claimed

### 6.1 The exact random baseline already has a name

[Repeated Random Sampling for Minimizing the Time-to-Accuracy of Learning](https://proceedings.iclr.cc/paper_files/paper/2024/hash/68b8d2bc77268facfc75a78782da9559-Abstract-Conference.html) was published at ICLR 2024. It introduces **Repeated Sampling of Random Subsets (RSRS/RS2)**: sample a fresh random subset for every training round rather than train repeatedly on a fixed random subset. It evaluates RS2 against dozens of pruning and dataset-distillation methods, includes a 10%-per-epoch example, analyzes convergence/generalization, and argues that data-selection methods must beat RS2 after accounting for selection cost.

Implication:

> The current random baseline should be called RS2. “Fresh random 10% per epoch” cannot be presented as a new method.

### 6.2 Random selection is already a strong LLM fine-tuning result

[Rethinking Data Selection at Scale: Random Selection is Almost All You Need](https://aclanthology.org/2025.findings-emnlp.146/) (Findings of EMNLP 2025) evaluates self-scoring methods on million-scale SFT pools and reports that nearly all struggle to significantly outperform random selection. It emphasizes diversity over narrow quality scoring.

Implication:

> “Random beats common LLM data selectors because it preserves diversity” is already an explicit recent claim.

### 6.3 Coverage over difficulty is already an explicit generative-fine-tuning thesis

[Rethinking Data Selection: The Importance of Coverage over Difficulty in Generative Fine-Tuning](https://openreview.net/forum?id=qImiy98UhN) (ICLR 2026 DATA-FM workshop) reports that difficulty-based scores fall behind random selection on generative tasks because they fail to cover the input distribution, then proposes clustering-based coverage selection.

Implication:

> “Difficulty loses to random because random has better coverage” is not sufficiently novel by itself, even in LLM fine-tuning.

### 6.4 Random/difficulty data dropout is already a peer-reviewed efficiency method

[Progressive Data Dropout: An Embarrassingly Simple Approach to Train Faster](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ed2dad593d87ca474a636cba610a29d3-Abstract-Conference.html) (NeurIPS 2025) studies changing data subsets across epochs, including difficulty-based and schedule-matched random variants, and reports that the random schedule-matched variant is strongest in its experiments.

Implication:

> The regularization/efficiency story for dynamic random data dropout is also occupied outside language modeling.

### 6.5 Dynamic pruning already treats bias toward informative points as a core problem

[InfoBatch](https://openreview.net/forum?id=C61sk5LsK6) (ICLR 2024 oral) dynamically prunes less-informative samples but uses randomization and gradient rescaling to approximate the original gradient, explicitly addressing selection-induced bias.

Implication:

> A stochastic or reweighted selector must compare against established unbiased dynamic-pruning ideas, not only deterministic top-k.

### 6.6 “Learnable, worth learning, not yet learned” is already formalized

[RHO-LOSS](https://proceedings.mlr.press/v162/mindermann22a.html) (ICML 2022) selects points by reducible holdout loss rather than raw hardness, motivated by the fact that high-loss examples may be noisy or unlearnable.

Implication:

> The distinction between difficulty and reducible training value is established. Entropy gain must be compared empirically and conceptually to RHO-LOSS-style utility.

### 6.7 Recent reasoning-distillation selection work raises the baseline bar

Several 2025–26 works claim that non-random selection can help reasoning distillation:

- [Skill-Aware Data Selection and Fine-Tuning for Data-Efficient Reasoning Distillation](https://arxiv.org/abs/2601.10109) prioritizes weak student skills and reports gains over random across mathematical reasoning benchmarks.
- [LARK: Learnability-Grounded Trajectory Selection for Efficient Reasoning Distillation](https://arxiv.org/abs/2605.30651) targets loss-decrease rate and explicitly regularizes toward distributional coverage with a chi-squared penalty.
- [Tailoring the Curriculum: Student-Centered Reasoning Distillation via Dynamic Data-Model Compatibility](https://arxiv.org/abs/2605.29229) argues that compatibility changes during training and reports benefits from dynamic selection.
- [Unified Data Selection for LLM Reasoning](https://openreview.net/forum?id=heVn5cNfje) proposes a sequence-level high-entropy score and reports wins over random in some reasoning-training regimes.

Implication:

> Showing that the current one-token entropy heuristic loses to RS2 does not establish that “selection fails.” A defensible broad conclusion requires reproducing at least one strong learnability/compatibility method and one sequence-level reasoning selector.

### 6.8 What appears less occupied

The following combination is more distinctive:

1. online, per-epoch scoring during reasoning distillation;
2. direct measurement of score validity over training;
3. separation of current-error prediction from downstream utility prediction;
4. exposure and coverage diagnostics tied to downstream outcomes;
5. a reliability-adaptive policy that shrinks toward uniform sampling when score utility decays;
6. a phase diagram over training stage, selection ratio, student, teacher reliability, and scoring family;
7. compute accounting that includes teacher generation, selector scoring, and student training.

This is the strongest available novelty space.

---

## 7. Defensibility matrix for possible paper theses

Scores below are strategic judgments, not formal probabilities.

| Thesis | Novelty | Supported now | Additional cost | Likely ceiling | Recommendation |
|---|---:|---:|---:|---|---|
| Entropy gain beats baselines | low–medium | very low | high | weak unless result reverses | stop as headline |
| Random rotation beats informed selection | low | medium | low | replication/workshop | do not use as novelty |
| Coverage beats difficulty in reasoning distillation | low–medium | medium | medium | incremental; close prior work | only as supporting result |
| Entropy signal collapses during online reasoning distillation | medium | strong on Qwen | medium | Findings/strong empirical section | promising |
| Current error is not downstream training utility | medium–high | suggestive | medium–high | strong if broadly demonstrated | recommended core question |
| Compute-matched benchmark of dynamic reasoning selection | medium–high | infrastructure partly ready | high | ICLR dataset/benchmark shape | recommended safer path |
| Reliability-gated score/uniform mixture | high | not tested | medium–high | ICLR method shape if it wins | recommended high-upside path |
| Teacher-reliability × training-stage phase diagram | high | offline hints only | high | original analysis paper | promising secondary path |
| Trace-on-demand economic selection | medium–high | not demonstrated | high/API | practical paper if savings large | only at small budgets |

---

## 8. Path A — compute-matched benchmark and phase diagram

### 8.1 Proposed thesis

> **Data-selection methods for reasoning distillation are commonly evaluated against weak or mismatched random baselines. Under compute-matched online evaluation, selector performance depends on budget, training stage, coverage, teacher reliability, and whether the score predicts learning utility rather than current difficulty.**

This path treats RS2 as prior art and contributes a reasoning-distillation-specific evaluation standard plus new empirical findings.

### 8.2 Core research questions

1. Does a selector beat RS2 at the same backpropagation sample views?
2. Does it beat RS2 after including score-computation cost?
3. Does it reduce unique teacher traces when traces are generated on demand?
4. Does score quality change across training?
5. Does a score that predicts current errors also predict validation improvement?
6. At what selection ratios does informed selection help or hurt?
7. Does the answer change with student family, dataset, and teacher reliability?
8. Are published fixed-random baselines materially weaker than RS2?

### 8.3 Required methods

Minimum benchmark set:

1. full-corpus uniform stream, compute matched;
2. fixed random subset;
3. RS2 independent random subset each round;
4. cyclic random without replacement;
5. student entropy top-k;
6. entropy gain top-k;
7. RHO-LOSS-style reducible loss;
8. one recent reasoning selector: preferably LARK or DMC;
9. one stochastic/coverage-aware selector;
10. optional sequence-level entropy baseline such as HES if feasible.

### 8.4 Required regimes

- Selection ratios: `1%, 2%, 5%, 10%, 20%`.
- Training stages: early, middle, late; report score diagnostics continuously.
- Students: Qwen, Phi, Llama already available.
- Datasets: MMLU-Pro STEM plus at least one open-ended reasoning dataset such as GSM8K or MATH.
- Trace source: answer-hidden correct traces for headline results.
- Seeds: at least three for headline cells.

The full Cartesian product is too expensive. Use staged racing:

1. Qwen single-seed sweep across ratios and selectors;
2. retain only qualitatively distinct/winning methods;
3. three seeds at the most informative two ratios;
4. cross-model confirmation at the selected ratio;
5. second-dataset confirmation for the final three methods.

### 8.5 Main figures

1. accuracy versus backpropagation sample views;
2. accuracy versus training tokens;
3. accuracy versus end-to-end wall-clock including scoring;
4. accuracy versus cumulative unique teacher traces;
5. score-to-error AUC over training;
6. score-to-short-horizon-utility correlation over training;
7. coverage and exposure Gini over training;
8. phase diagram: best method by selection ratio and training stage;
9. paired per-question win/loss matrix at the selected checkpoint.

### 8.6 Novelty and risk

Strengths:

- directly addresses a methodological gap in a fast-moving reasoning-selection literature;
- can invalidate conclusions that use fixed random but omit RS2;
- uses existing online infrastructure and stored diagnostics;
- remains valuable even if no new selector wins.

Risks:

- benchmark papers require breadth and impeccable matching;
- reproducing LARK/DMC may be nontrivial because their selection unit or candidate pool differs;
- if all effects vanish after matching, the paper becomes primarily a cautionary evaluation report;
- one MMLU multiple-choice dataset is insufficient.

### 8.7 Plausible titles

- **When Does Data Selection Help Reasoning Distillation?**
- **Beyond Fixed Random: A Compute-Matched Study of Data Selection for Reasoning Distillation**
- **Difficulty, Coverage, and Utility in Dynamic Reasoning Distillation**
- **Are We Evaluating Reasoning Data Selection Correctly?**

---

## 9. Path B — reliability-gated selection

### 9.1 Proposed thesis

> **A data selector should exploit a model-dependent score only while that score demonstrably predicts training utility. When reliability decays, the sampling distribution should shrink toward uniform coverage.**

This turns the observed score collapse into an algorithmic contribution.

### 9.2 Minimal method

Define a score-based distribution `q_t(i)` and uniform distribution `u(i)=1/N`. Train from:

```text
p_t(i) = lambda_t * q_t(i) + (1 - lambda_t) * u(i)
```

where `lambda_t` is determined by a reliability estimate rather than fixed forever.

Possible reliability estimates, in increasing order of faithfulness and cost:

1. score-to-current-error AUC on a small labeled probe set;
2. score calibration against option-level correctness;
3. score correlation with one-epoch per-bin loss reduction;
4. gradient alignment between selected training batches and a validation batch;
5. short-horizon validation improvement from training on score quantiles.

The method should satisfy:

- `lambda_t -> 0` when score reliability is at chance or negative;
- a minimum probability floor for every example;
- bounded divergence from the empirical distribution;
- optional without-replacement coverage inside a cycle;
- no hard top-k discontinuity unless reliability is very high.

### 9.3 Relationship to prior work

This is close to:

- LARK's learnability objective and chi-squared coverage regularization;
- InfoBatch's concern with unbiased gradients;
- RHO-LOSS's reducible-utility target.

The novelty must therefore be specifically **online reliability adaptation**:

- the selector measures whether its own score is still useful;
- it changes exploitation strength during the same run;
- it can fall back to RS2 automatically;
- it exposes a diagnostic certificate/curve explaining its decisions.

Simply mixing entropy and random with a fixed coefficient is a useful baseline, not a sufficient method contribution.

### 9.4 A possible theoretical angle

Let `u_i(t)` denote the unknown marginal utility of training on item `i` at time `t`, and let `s_i(t)` be an observed score. A score-based policy is beneficial only when its induced sampling weights have positive covariance with utility after accounting for importance and coverage.

A theory section could formalize:

1. when a noisy utility proxy improves expected one-step progress over uniform sampling;
2. how proxy reliability determines the optimal shrinkage toward uniform;
3. a regret bound for a mixture that estimates reliability online;
4. why hard top-k can be worse than uniform when reliability crosses zero;
5. why coverage constraints cap worst-case harm under score drift.

This is high risk but much more novel than another entropy formula.

### 9.5 Required empirical result

For this path to survive, the adaptive mixture must:

- match entropy selection early when entropy is informative;
- transition toward uniform sampling before the AUC collapse;
- beat or match RS2 across seeds at fixed compute;
- outperform a fixed 50/50 mixture;
- outperform a fixed annealing schedule;
- remain competitive with LARK/DMC or clearly use less scoring compute.

If it only ties RS2 while adding complexity, it is an analysis tool, not a method contribution.

---

## 10. Path C — repair entropy measurement, then retest once

This path gives the original method one controlled chance. It should be time-boxed.

### 10.1 Repair 1: option-alphabet entropy

Current entropy is computed over the entire vocabulary. Replace it with:

1. collect probability mass for every valid option tokenization, including case and leading-space variants;
2. aggregate variants by semantic option;
3. renormalize across the `K` valid options;
4. compute `K`-way entropy and divide by `log K`;
5. separately log probability mass assigned to valid options.

This yields:

- comparable support across model tokenizers;
- bounded entropy in `[0,1]`;
- an explicit format-compliance diagnostic;
- less contamination from irrelevant vocabulary tail mass.

### 10.2 Repair 2: align scoring with the training target

Test three scoring contexts offline before training:

1. direct-answer option entropy;
2. option entropy after a forced `<think>` prefix or fixed short reasoning scaffold;
3. normalized student loss on the actual teacher trace, optionally minus a proxy/reference loss.

The third is more expensive but much closer to what training optimizes.

### 10.3 Repair 3: normalize cross-model comparisons

Do not subtract raw full-vocabulary nats from different tokenizers.

Candidate replacements:

- option-normalized entropies;
- per-model ECDF quantiles, using `q_student - q_proxy`;
- isotonic residual `q_student - E[q_student | q_proxy]`;
- calibrated expected utility `P(student wrong) * P(teacher correct)`;
- RHO-style normalized trace-loss difference.

### 10.4 Repair 4: replace hard top-k with controlled stochasticity

Test independently:

1. score-proportional sampling with temperature;
2. uniform probability floor;
3. capped per-item exposure;
4. cyclic coverage with score-based prioritization inside each cycle;
5. category/skill stratification only if skew is empirically material.

Do not combine normalization, stratification, tail removal, soft sampling, and annealing in one first experiment. If the combination wins, no component will be attributable.

### 10.5 Kill criterion

After offline validation, run one corrected selector and one coverage-aware version on Qwen with three seeds.

Stop pursuing entropy as the paper's method if:

- corrected entropy still loses to RS2;
- its score-to-utility correlation decays to chance;
- the proxy term still has no downstream marginal benefit;
- gains appear only for answer-leaked traces;
- improvement is smaller than seed variance.

---

## 11. Path D — teacher-reliability and training-stage phase diagram

### 11.1 Motivation

The earlier notebook audit found that metric rankings change when correctness labels come from weak versus strong teachers. The online audit now shows that metric reliability also changes with student training stage.

This suggests a two-axis hypothesis:

> **The best selection policy depends jointly on teacher reliability and the student's current stage.**

### 11.2 Experimental design

Teacher axis:

- gold/correctness-filtered trace;
- strong teacher direct trace;
- weaker teacher direct trace;
- synthetically corrupted teacher correctness at controlled rates;
- answer-leaked corrected trace as an upper-quality but artificial condition.

Student-stage axis:

- base model;
- early checkpoint;
- middle checkpoint;
- late checkpoint.

Selectors:

- student entropy;
- proxy entropy;
- entropy gain;
- calibrated expected utility;
- RHO-style loss difference;
- RS2;
- coverage-aware mixture.

Outcomes:

- teacher-correct/student-wrong rate;
- one-epoch student loss decrease;
- validation improvement;
- final downstream accuracy;
- trace-generation cost.

### 11.3 Potential contribution

A phase diagram can reconcile apparently contradictory results:

- strong reliable teacher: student uncertainty may be enough early;
- weak teacher: proxy confidence/noise avoidance matters more;
- late student: uncertainty may no longer identify useful data;
- low budget: exploitation may help;
- moderate budget: coverage may dominate.

This path is analytically original but experimentally broad. It is best combined with Path A, not pursued as an isolated formula paper.

---

## 12. Path E — trace-on-demand economic selection

### 12.1 Why this framing remains potentially valuable

Most reasoning-selection methods rank already generated trajectories. A question-level selector that operates before teacher generation could save the most expensive resource: teacher output tokens.

The correct pipeline would be:

1. maintain a pool of questions without traces;
2. score questions cheaply with the student/proxy;
3. select a subset;
4. generate teacher traces only for selected questions;
5. cache traces permanently;
6. train and repeat.

### 12.2 Why the current result does not demonstrate this benefit

- traces were already generated;
- Qwen entropy gain covers 84.7% of questions by 100 rounds;
- long-run savings would therefore be modest;
- selection itself scans the entire pool every epoch;
- corrected-answer generation uses gold information.

### 12.3 Conditions under which it can become a paper

- operate at genuinely small cumulative unique budgets, not only small per-epoch budgets;
- compare against random trace acquisition without replacement;
- report teacher tokens, monetary/API cost, latency, and student training tokens;
- demonstrate a Pareto improvement in accuracy versus unique teacher tokens;
- use answer-hidden, pinned teacher models;
- show transfer across at least two students or datasets.

This path is attractive only if selection helps at `1–5%` ratios or if a reusable selected set transfers across students. At 85% cumulative coverage it is not a strong economic story.

---

## 13. Recommended two-week diagnostic sprint

The goal is not to finish the paper. It is to determine which paper, if any, is supported.

### 13.1 Days 1–2: protocol repair

1. Add real seed plumbing and log all seeds.
2. Add a validation split for checkpoint selection.
3. Standardize evaluation generation cap.
4. Standardize objective mix and packing across full and resampling runs.
5. Shuffle selected subsets independently of selection score.
6. Implement direct RS2 without a model-scoring pass.
7. Add cyclic-without-replacement random sampling.
8. Log sample views, supervised tokens, unique examples, scoring tokens/FLOPs, and wall-clock.

### 13.2 Days 2–4: offline diagnostics on all available trajectories

For Qwen, Phi, and Llama where artifacts exist:

- score-to-error AUC by epoch;
- score-to-correctness calibration;
- consecutive-set Jaccard;
- cumulative unique coverage;
- exposure histogram and Gini;
- category/skill distribution;
- trace length and teacher correctness by score quantile;
- student/proxy score correlation;
- top-k overlap across methods;
- relationship between offline HV and downstream accuracy.

Add option-alphabet entropy and residualized/quantile gain offline. Do not train a new score whose offline diagnostic is already at chance.

### 13.3 Days 4–10: Qwen matched pilot

Run the following at the existing 10% budget, initially with one seed and frequent validation checkpoints:

1. full-corpus uniform stream stopped at matched sample views;
2. fixed random 10%;
3. RS2 independent random 10%;
4. cyclic random 10%;
5. shuffled student-entropy top-k;
6. shuffled entropy-gain top-k;
7. one simple coverage mixture, e.g. 50% cyclic + 50% corrected score.

If corrected option entropy passes offline diagnostics, use it in the mixture. Otherwise use student entropy only to isolate coverage effects.

### 13.4 Days 10–14: replication and gate

- Replicate the three most informative methods with two additional seeds.
- Run the final three methods on one additional student for directional confirmation.
- Perform paired per-question tests on validation-selected checkpoints.
- Produce preliminary compute/coverage/signal-health figures.

### 13.5 Sprint decision gate

| Outcome | Decision |
|---|---|
| RS2 wins after full matching; score AUC/utility collapses | pursue benchmark/diagnostic Path A |
| Corrected score + coverage mixture beats RS2 across seeds | pursue reliability/method Path B |
| Full-corpus matched stream ties or beats RS2 | drop rotation claim; focus on selection/evaluation confounds |
| Option entropy fixes AUC but not downstream accuracy | current error is not utility; pursue utility-target benchmark |
| No result replicates across seeds | stop headline claims; fix protocol before more GPU spend |
| Effects exist only with corrected-answer traces | do not use as headline; treat as trace-generation artifact |

---

## 14. Full experiment program for an ICLR-shaped submission

### Phase 0 — validity

- seed plumbing;
- validation checkpoint selection;
- matched objectives, packing, batches, steps, tokens, and caps;
- answer-hidden traces;
- exact RS2 implementation;
- test-set freeze.

**Gate:** no paper claim before this phase is complete.

### Phase 1 — Qwen selector comparison

- ratios `1, 2, 5, 10, 20%`;
- fixed random, RS2, cyclic random, entropy, entropy gain, RHO-style, coverage mixture;
- one seed for racing, three for finalists;
- continuous signal-health diagnostics.

**Gate:** identify whether any selection regime beats RS2 and where.

### Phase 2 — recent external baselines

- reproduce/adapt LARK or DMC;
- include one sequence-level reasoning selector;
- document differences in selection unit: question versus trajectory;
- compare scoring overhead and prerequisite trace generation.

**Gate:** a broad “selection fails” claim is forbidden unless recent strong methods are included.

### Phase 3 — generalization

- second dataset with open-ended answers;
- second and third student confirmation;
- OOD evaluation;
- per-skill/category breakdown;
- teacher-reliability ablation if it remains central.

**Gate:** one-dataset effects remain workshop-level evidence.

### Phase 4 — final method or benchmark freeze

If a method wins:

- freeze method and hyperparameters before final test runs;
- compare against fixed-mixture and fixed-schedule ablations;
- add theoretical or utility-estimation analysis.

If no method wins:

- freeze the benchmark protocol;
- expand diagnostic breadth;
- emphasize invalid baselines, signal drift, and phase boundaries;
- release selection logs and standardized configs.

---

## 15. Metrics the paper must report

### 15.1 Outcome metrics

- validation-selected test accuracy;
- mean and standard deviation across seeds;
- paired McNemar or paired bootstrap intervals;
- OOD accuracy;
- per-category/skill accuracy;
- calibration if entropy is a claimed mechanism.

### 15.2 Training-cost metrics

- optimizer steps;
- backpropagated examples;
- supervised tokens;
- total forward tokens;
- selector forward/backward cost;
- wall-clock and GPU-hours;
- time-to-accuracy;
- maximum memory.

### 15.3 Teacher-cost metrics

- unique questions for which traces were generated;
- teacher input/output tokens;
- teacher generation retries/rejection rate;
- monetary cost if an API is used;
- cache reuse across epochs and students.

### 15.4 Selection-health metrics

- cumulative unique coverage;
- coverage by category/skill/cluster;
- exposure Gini and maximum repeat count;
- consecutive and long-horizon Jaccard;
- score distribution and threshold;
- score-to-error AUC;
- score-to-utility correlation;
- selected-set teacher correctness;
- valid-answer probability mass;
- score calibration drift.

### 15.5 Pareto reporting

The main result should be a frontier, not a single peak:

- accuracy versus student training tokens;
- accuracy versus end-to-end GPU-hours;
- accuracy versus unique teacher tokens;
- accuracy versus wall-clock;
- accuracy versus cumulative coverage.

This prevents a method from appearing efficient on one resource while silently spending another.

---

## 16. Analysis that would make the negative result scientifically useful

### 16.1 Utility-by-score-quantile intervention

At several checkpoints:

1. divide the pool into score deciles;
2. clone the same checkpoint;
3. train each clone for a small fixed number of updates on one decile;
4. evaluate validation loss/accuracy change;
5. correlate decile score with actual improvement.

This directly tests whether the score ranks training utility. It is more informative than teacher-correct/student-wrong rate.

### 16.2 Cross-over intervention

At the point where entropy AUC begins collapsing:

- continue one branch with entropy top-k;
- switch one branch to RS2;
- switch one branch to cyclic coverage;
- switch one branch to a score/uniform mixture.

If switching improves the trajectory, the result supports a stage-dependent policy rather than only a retrospective correlation.

### 16.3 Exposure equalization

Compare:

- unconstrained top-k;
- top-k with maximum exposure cap;
- score priority among never-seen items;
- cyclic pool with score-based ordering;
- pure cyclic random.

This isolates whether entropy's harm comes from the score itself or from repeated exposure.

### 16.4 Score permutation control

Preserve the exact score distribution and selected-set size but randomly permute scores across questions each epoch. This separates effects of:

- score distribution/threshold mechanics;
- selection ordering;
- semantic association between score and item.

### 16.5 Frozen-score versus online-score control

Compare:

- fixed epoch-0 entropy subset;
- epoch-0 score with stochastic sampling every epoch;
- online entropy top-k;
- online entropy soft sampling;
- RS2.

This determines whether recursion helps, hurts, or merely changes which examples are repeatedly emphasized.

---

## 17. Safe and unsafe claim language

### 17.1 Unsafe with current evidence

- “Random sampling beats all data-selection methods.”
- “Entropy gain is harmful.”
- “Using 10% of the data outperforms full-data training.”
- “We reduce teacher data by 90%.”
- “Entropy gain measures epistemic uncertainty.”
- “The proxy contributes nothing.”
- “Rotation regularization causes the gain.”
- “Current models cannot be trained properly.”
- “Selection does not work for reasoning distillation.”

### 17.2 Safer after the current audit

- “In our current MMLU-Pro setup, RS2-style random resampling exceeds deterministic entropy-based top-k selection for all three tested student families.”
- “For Qwen, the error-discrimination of the online entropy score falls from useful early in training to approximately chance late in training.”
- “The entropy-based policies produce substantially lower cumulative coverage and higher exposure inequality than RS2.”
- “The proxy term materially changes selected subsets but has no demonstrated downstream advantage in the current Qwen result.”
- “The existing full-corpus reference is not a fully matched causal control.”

### 17.3 Strong claims that could become safe after the proposed program

- “Fixed-random baselines systematically overstate gains from reasoning-data selection; RS2 is the appropriate baseline.”
- “Predictive difficulty does not reliably estimate downstream training utility in online reasoning distillation.”
- “Score reliability changes over training, creating a phase boundary between selection and uniform coverage.”
- “Reliability-adaptive sampling matches exploitation when useful and falls back to uniform coverage when the score becomes uninformative.”
- “Selection helps only in specified budget/teacher/stage regimes.”

---

## 18. Candidate paper narratives

### Narrative 1 — recommended safer narrative

**Title:** *When Does Data Selection Help Reasoning Distillation?*

Story:

1. Recent reasoning distillation relies on heuristic or learned selectors.
2. Many evaluations use fixed random subsets and omit RS2 or end-to-end costs.
3. Under a matched protocol, results vary sharply by budget and training stage.
4. Current-error scores lose validity during training.
5. Coverage and exposure diagnostics explain failures.
6. The paper provides an evaluation standard and phase diagram.

Required contribution:

- breadth, recent baselines, and standardized artifacts.

### Narrative 2 — recommended high-upside narrative

**Title:** *Know When Not to Select: Reliability-Gated Data Sampling for Reasoning Distillation*

Story:

1. Model-dependent scores are nonstationary.
2. Hard selection assumes permanent reliability.
3. Measure score utility online.
4. Shrink selection toward uniform coverage as reliability decays.
5. Beat fixed selection and RS2 across regimes.

Required contribution:

- a real win over RS2 and fixed mixtures, plus differentiation from LARK.

### Narrative 3 — narrower negative/diagnostic narrative

**Title:** *Predictive Uncertainty Is Not Training Utility in Reasoning Distillation*

Story:

1. Entropy predicts errors early.
2. It becomes uninformative late.
3. Top-k continues exploiting the failed signal.
4. Coverage debt and repetition accumulate.
5. Correct measurement and utility interventions identify the failure.

Likely venue ceiling:

- strong Findings/workshop paper unless replicated broadly or paired with a corrective method/theory.

### Narrative 4 — not recommended as a standalone novelty claim

**Title:** *Data Rotation Beats Uncertainty Selection*

Problem:

- too close to RS2, Progressive Data Dropout, random-at-scale SFT, and coverage-over-difficulty work;
- reads as an application of established random-resampling baselines;
- requires a stronger reasoning-specific mechanism to justify publication.

---

## 19. Recommended decision tree

```text
Start
  |
  |-- Is random vs full still positive after identical objective, packing,
  |   update/token budget, validation checkpoint selection, and eval cap?
  |       |
  |       |-- No -> Drop the rotation-vs-full claim.
  |       |         Continue only with selector-vs-RS2 diagnostics.
  |       |
  |       `-- Yes -> Report as reasoning-domain replication of RS2/PDD,
  |                 not as the main novelty.
  |
  |-- Does entropy/EG beat RS2 across >=3 seeds in any budget/stage regime?
  |       |
  |       |-- Yes -> Map the phase boundary and test corrected measurement.
  |       |
  |       `-- No -> Stop the original method headline.
  |
  |-- Does corrected measurement maintain utility correlation late?
  |       |
  |       |-- Yes -> Test coverage-aware stochastic selection.
  |       |
  |       `-- No -> Entropy is diagnostic only; move to utility scores.
  |
  |-- Does a reliability-gated or utility-based selector beat RS2?
  |       |
  |       |-- Yes -> Method paper.
  |       |
  |       `-- No -> Broad benchmark/negative paper only if replicated
  |                 across datasets, models, recent methods, and seeds.
  |
  `-- Does the broad negative fail to replicate?
          |
          `-- Yes -> Do not force a paper; document the result and redirect.
```

---

## 20. Immediate recommendations in priority order

1. **Rename the random baseline RS2 in code, plots, and writing.** Cite ICLR 2024.
2. **Freeze the current test set.** Add validation-based checkpoint selection before new headline runs.
3. **Fix seed plumbing.** Current hard-coded seed 42 is incompatible with a credible seed study.
4. **Make random, entropy, and full-corpus training identical on objective, packing, update/token budget, and evaluation cap.**
5. **Shuffle all selected subsets.** Remove the membership/order confound.
6. **Reproduce the Qwen score-AUC and coverage analysis for Phi and Llama.** This is the most distinctive current evidence.
7. **Implement option-alphabet entropy and reject it offline if it does not preserve score validity.**
8. **Run fixed random, RS2, and cyclic random.** These separate subset size, rotation, and guaranteed coverage.
9. **Measure short-horizon utility by score quantile.** Stop using current error/HV as an unvalidated surrogate.
10. **Run one coverage mixture and one utility-based external baseline.** Avoid a large new formula zoo.
11. **Only after the Qwen gate, spend on a second dataset and recent reasoning selectors.**
12. **Choose the paper after the evidence, not before it.** The original method, benchmark pivot, and adaptive-method pivot have mutually incompatible claims.

---

## Bottom line

The random result does not kill the project, but it kills two easy stories:

1. **“Entropy gain beats the relevant baselines.”** It currently does not.
2. **“Fresh random subsets are a novel solution.”** RS2 and related work already occupy that claim.

The scientifically interesting opening is narrower and better:

> **Online data-selection scores are nonstationary. In reasoning distillation, a score that initially identifies student weaknesses can lose all relationship to errors or training utility, while deterministic top-k continues to distort coverage. The correct question is not which static formula wins, but when a selector deserves to override uniform sampling.**

The recommended near-term objective is therefore not to prove that random beats everything. It is to establish whether score validity and actual training utility can be measured online, whether a selector can safely adapt to their decay, and which budget/stage/teacher regimes justify selection at all.

If that program produces a reliability-adaptive win, the project can become a method paper. If it produces a broad, compute-matched negative result, it can become a benchmark/diagnostic paper. If neither replicates beyond the current setup, the correct decision is to document the negative result and stop investing in the entropy-gain headline.
