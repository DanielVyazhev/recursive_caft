**Adaptive proxy selection for reasoning distillation: evidence, mathematical foundations, and a three-week plan**

Prepared 19 September 2026; revised 21 September 2026 to reflect the agreed entropy-only method and one-configuration training budget. This report uses the current checkout, the adjacent `../reasoning-fine-tune` implementation, the [previous paper](https://arxiv.org/html/2506.21220v4), and a focused search of relevant primary literature. I extracted 880 checkpoint/cap evaluation records, recomputed diagnostics from 2,100 scoring snapshots, and ran additional CPU analyses on the saved random trajectories. I did not train new language models. Findings below distinguish observations, mathematical statements under explicit assumptions, and proposals.

**The current experiment uses an empirical diminishing proxy correction, computed only from the available scalar entropies.** At each resampling round, use the existing normalized student and proxy entropies directly without centering, subtract the proxy with weight `(1 - training_progress)^2`, and sample using positive scores plus uniform exploration. The selector uses no correctness labels, cross-entropy, probability vectors, or additional proxy calls. Existing student entropy estimation continues each round.

The mathematical rationale below uses a working aleatoric–epistemic decomposition with a fixed imperfect proxy. A relative-error budget supplies the additional principle needed to motivate decay: remove as much aleatoric uncertainty as possible without letting erroneous proxy subtraction dominate the student's remaining epistemic signal. The schedule remains empirical; increasing AUROC alone does not establish the learning assumption. The remaining budget is one configuration with multiple seeds. The [concise report](combined-score-minimal.md) includes the same complete derivation and implementation specification.

> A fixed proxy retains epistemic uncertainty. As the student's remaining epistemic uncertainty decreases, controlling relative over-subtraction calls for a smaller proxy weight.

This rationale replaces the regression and intercept derivations. Historical rank-based and correctness-based diagnostics remain separate from the proposed method and do not identify its latent uncertainty components.

**The current results support a more specific conclusion than “random always wins.”** The following table fixes the subset round at 100 and the inference cap at 2,048 thinking tokens. Accuracies are percentages. Random is mean ± sample standard deviation across the three saved seeds; the other rows are individual runs. The proportional entropy-gain row is the original, non-shuffled variant.

| Selection policy, round 100, cap 2,048 | Qwen | Phi | Llama |
|---|---:|---:|---:|
| Random, three runs | 56.22 ± 0.66 | 65.92 ± 0.40 | 51.19 ± 1.59 |
| Hard top-k student entropy | 53.03 | 64.42 | 50.08 |
| Hard top-k entropy gain | 53.28 | 64.63 | 50.37 |
| Proportional entropy gain | 56.94 | 66.58 | 53.53 |
| Proportional student entropy, shuffled, seed 42 | 56.19 | 66.46 | 52.20 |

Hard selection underperforms the random mean for all three students here. Proportional selection already looks much more competitive. This makes it premature to discard the score family, but also premature to attribute the proportional results to the proxy: proportional student entropy is competitive too. Moreover, the original hard/proportional runs finish at round 100 while the random and shuffled proportional runs finish at 200. Consequently, their cosine learning-rate schedules differ even at the same round. These are descriptive comparisons, not matched causal estimates of a selector's effect.

Inference budget is especially consequential for Phi. At round 150:

| Phi policy | Cap 2,048 | Cap 4,096 |
|---|---:|---:|
| Random, mean of three runs | 65.88 | 59.89 |
| Proportional entropy gain, shuffled | 64.34 | 63.51 |
| Proportional student entropy, shuffled, seed 42 | 66.58 | 62.01 |

Thus, the apparent entropy-gain advantage at cap 4,096 reverses at cap 2,048. At round 200 the random mean is 66.13% at cap 2,048 but 55.57% at cap 4,096. Longer allowed reasoning clearly does not guarantee better answers in this setup. A selection method might change reasoning length, stopping behavior, or susceptibility to unhelpful continuation. Those are interesting mechanisms, but require different claims from “better knowledge transfer.” Report both caps, and choose the primary cap before further comparisons.

The corresponding full-corpus corrected-trace baselines peak at 51.25%, 63.76%, and 47.84% at cap 2,048. These are **test-set maxima across checkpoints**, included only to locate the existing results. They should not become the final checkpoint-selection protocol. Complete curves and descriptive peaks are in [the audit outputs](../artifacts/selection_report_audit/evaluation_curves.csv).

**The increasing ROC AUC observation is real under random sampling, but it reverses under hard selection.** The current notebook evaluates direct-answer correctness on the training pool, using negative student entropy as the correctness score. This is equivalent to evaluating positive entropy against an error indicator. Recomputed values at scoring snapshot 99 are:

| Training policy | Qwen AUROC | Phi AUROC | Llama AUROC |
|---|---:|---:|---:|
| Random | 0.927 | 0.923 | 0.924 |
| Hard student entropy | 0.510 | 0.483 | 0.473 |
| Hard entropy gain | 0.622 | 0.658 | 0.663 |
| Proportional entropy gain | 0.873 | 0.828 | 0.846 |

Initial student AUROCs are approximately 0.71–0.74. Under random, they rise further to approximately 0.94–0.97 by snapshot 199. Under proportional student entropy they eventually deteriorate too, although much later; by then the training pool contains very few remaining errors, so uncertainty around AUROC also becomes important.

This is more informative than a universal claim that “the student becomes a better uncertainty estimator.” The reliability trajectory depends on the selection policy. One plausible explanation is that selection removes some types of mistakes while repeatedly neglecting others. Another is that repeated training changes confidence or answer-format behavior on the preferred subset. The evidence does not yet separate these explanations.

At snapshot 99, the hard entropy-gain selector's selected examples are already directly answered correctly at rates of 90.4%, 98.8%, and 94.6% for Qwen, Phi, and Llama. In Phi, this exceeds the full training-pool correctness of 93.5%. The clipped entropy-gap score's correctness AUROC is only 0.329 there. Whatever it initially measured, its late ranking is not concentrating current direct-answer errors.

![Reconstructed coverage and student error-discrimination dynamics](../artifacts/selection_report_audit/coverage_and_auc.png)

These are training-pool diagnostics, not held-out reasoning calibration. For example, the random Qwen run reaches 96.9% direct-answer training accuracy at snapshot 199 while its round-200 held-out reasoning accuracy is about 57.6%. Memorization and generalization must remain separate in the interpretation.

**Random beats hard selection partly because hard selection repeatedly revisits a restricted population.** After 100 draws of 1,024 traces, reconstructed unique coverage is:

| Policy | Qwen | Phi | Llama |
|---|---:|---:|---:|
| Random | 99.99% | 100.00% | 100.00% |
| Hard entropy gain | 84.67% | 81.99% | 83.53% |
| Hard student entropy | 83.72% | 80.72% | 84.44% |
| Proportional entropy gain | 99.70% | 98.54% | 98.72% |

At the last hard entropy-gain draw, 81.7%, 87.8%, and 87.5% of the selected examples also appeared in the preceding draw. Random's analogous retention is around 10%. Exposure concentration is substantial even among examples that eventually get selected: the quantity `(sum exposures)^2 / sum(exposures^2)` is roughly 4,100–4,500 for hard entropy gain, versus about 8,900 for random. This quantity summarizes exposure inequality; it is not an effective sample size for a statistical confidence interval.

Some proportional-sampling measurements failed and their random keys were redrawn in the actual sampler. Those draws cannot be recovered from the saved files. The proportional coverage figures are therefore reconstructions using available keys, rather than exact training manifests. Hard selections and the saved-key random selections are reproducible from their snapshots, subject to the historical sampler matching the current implementation.

Coverage explains a plausible weakness of hard selection. It does **not** explain why random should beat an otherwise identical full-corpus training stream: full-corpus training has excellent coverage too. That is a separate question, requiring the optimization and supervision controls below.

**The Qwen sanity check confirms a substantial gap with the same unpacked trainer.** Updated after inspecting commit `025e09f3f9319da0d1a56585dff81e96f2d4356c`, its experiment source, and the locally available saved responses. The [sanity configuration](../src/experiments/distillation_by_metrics/mmlu/random/qwen_3b_head_truncated8192_sanity_check.py) uses the same resampling trainer, draws 9,600 of 9,626 questions once, shuffles each epoch, and trains for 20 epochs. Its estimation statistics confirm that all 9,600 were selected without failed measurements. There is no packing. This substantially weakens packing or a different trainer implementation as explanations of the accuracy gap.

Recomputed directly from `responses.parquet`, at checkpoint 3,000:

| Qwen run | Accuracy, cap 2,048 | Accuracy, cap 4,096 |
|---|---:|---:|
| Full-data sanity check | 50.87% | 52.16% |
| Random resampling, seed 42 | 58.23% | 58.52% |
| Difference | +7.36 points | +6.36 points |

At cap 2,048, random corrects 409 items the sanity model misses and loses 232 that the sanity model answers correctly: a net 177 additional correct answers. Mean response lengths are approximately 4,012 and 4,014 **characters**, respectively. Thinking-budget exhaustion is 34.54% versus 35.54%; invalid answer counts are one versus zero. The gap is therefore not explained by simply producing shorter outputs, avoiding the cap, or fixing answer parsing. Those aggregate diagnostics do not rule out differences in reasoning quality or other behavioral changes. At cap 4,096, the random run actually exhausts the budget more often, 12.76% versus 8.48%, while retaining its accuracy advantage. Summary measurements are in [the sanity comparison](../artifacts/selection_report_audit/qwen_sanity_comparison.csv).

The check still changes supervision. It uses only `MMLUReasoningResponseDataset`; the [standard random configuration](../src/experiments/distillation_by_metrics/mmlu/random/qwen_3b_head_truncated8192.py) calls `get_merged_adapter_with_data_mix`, which adds 256 direct-answer examples to 1,024 traces each round. Both source files match the linked commit. With effective batch size 64, the resulting step counts at checkpoint 3,000 are:

| Training exposure through step 3,000 | Full-data sanity | Random resampling |
|---|---:|---:|
| Reasoning updates | 3,000 | 2,400 |
| Direct-answer updates | 0 | 600 |
| Reasoning-example presentations | 192,000 | 153,600 |
| Direct-answer presentations | 0 | 38,400 |

These counts describe configured examples and updates, not matched supervised-token budgets. The sanity run ends at 3,000 steps while random ends at 4,000; their cosine schedules also differ. The real observed result is a better training procedure, with the independent contributions of resampling and auxiliary supervision still unresolved.

**The leading explanation is stronger task supervision from the direct-answer updates.** This is a mechanistic hypothesis, not an established causal result. In a long teacher-forced trace, the final answer is a small part of the supervised sequence and is predicted after the supplied teacher reasoning. A separate direct-answer example requires predicting the answer from the question itself. It supplies a different learning problem, even though the final answer label is the same.

Because the adapter concatenates trace examples and then answer examples, direct answers get their own optimizer steps. Their influence is not proportional to their tiny fraction of the total tokens. Approximately one in five updates is devoted to them. A useful schematic is `80% trace-imitation updates + 20% direct-answer updates`; Adam, gradient clipping, and sequential parameter changes prevent treating this as an exact fixed weighted objective.

A plausible account is that trace imitation teaches useful reasoning patterns but also spends capacity fitting the teacher's wording and intermediate choices. Direct-answer updates repeatedly reinforce solving the underlying task and can counteract unhelpful changes from imitation. The near-identical output lengths at cap 2,048 fit a competence explanation better than a simple verbosity explanation, but do not prove it. The earlier direct-answer transition diagnostic below supplies additional, explicitly non-causal evidence.

**A genuine resampling mechanism remains plausible: changing when examples recur.** A full shuffled pass prevents an example from recurring within that pass. Independently redrawing a 1,024-question subset permits repeats across short round boundaries. Under ideal uniform sampling from 9,626 questions, adjacent subsets overlap in expectation by about 109 questions, or 10.6% of a subset. This changes the temporal correlations and spacing of gradients even when minibatch size is unchanged.

Such changes can affect optimization and implicit regularization. However, selecting a uniform subset and then a uniform minibatch from it gives the same marginal minibatch distribution as selecting that minibatch directly from the whole pool, at a fixed model state. Therefore “smaller subset means more noise in each minibatch” is not a sufficient explanation here. Differences arise through the joint sequence of batches, model evolution, and optimizer state. Nor does random sampling systematically remove bad traces: it preserves their proportion in expectation and eventually exposes nearly the entire pool.

The established literature supports studying sampling dynamics, but does not prove that this particular block-resampling procedure should outperform full-data reshuffling with Adam. [Repeated Random Sampling](https://arxiv.org/abs/2305.18424) establishes a strong empirical baseline; [Smith et al.](https://research.google/pubs/on-the-origin-of-implicit-regularization-in-stochastic-gradient-descent/) analyze implicit regularization from SGD under specific assumptions; [Gürbüzbalaban et al.](https://arxiv.org/abs/1510.08560) show that full reshuffling can itself improve convergence in a different theoretical setting. None identifies the mechanism in these runs.

Within the existing resource budget, describe the sanity check as ruling out packing as the sole explanation and demonstrating the gain in the same training implementation. Do not claim that it isolates subset selection. An additional matched trace-only random run, or a full-data run with the same direct-answer update pattern, would be needed to isolate that effect; this is an unresolved control, not an expansion of the agreed one-configuration experiment plan. The diminishing-correction experiment can still be compared to the existing random baseline, since both use the same direct-answer mixture.

The direct-answer mixture deserves particular attention. The merged adapter concatenates the trace block and then the direct-answer block. Its `shuffle=True` option shuffles within children before concatenation. At the nominal batch size of 64, a round therefore contains approximately 16 trace updates followed by four direct-answer updates. This is 20% of updates, despite being a tiny fraction of target tokens. The direct-answer subset is nested inside the trace subset for the deterministic and saved-key random samplers. It is not an independent random sample.

I checked the saved random trajectories for a simple diagnostic: among currently wrong, valid direct answers, how often is the answer correct at the next scoring snapshot? Pooling rounds 10–98 gives:

| Student | Unselected this round | Trace only | Trace plus direct answer |
|---|---:|---:|---:|
| Qwen | 13.25% | 14.34% | 45.11% |
| Phi | 14.15% | 14.99% | 48.89% |
| Llama | 13.36% | 14.33% | 49.65% |

The large association with direct-answer training makes the auxiliary objective a priority control. These pooled transition rates are **not** estimates of held-out reasoning improvement: examples recur, models share parameters, rounds differ, and the measurement prompt matches the direct-answer objective. They also do not establish that reasoning training contributes little to generalization. They establish that the measurement used to motivate selection is strongly affected by an auxiliary objective missing from the full baseline.

A future control, beyond the remaining one-configuration budget, would use a full-data stream with the same trace/direct-answer update ratio, loss normalization, inference cap, optimizer schedule indexed by updates, and equivalent packing behavior. Comparing that stream with fresh random subsets at matched cumulative target-token budgets, and adding a trace-only random control, would help isolate the auxiliary objective. These extra runs are not required by the current plan; the corresponding causal claims remain unresolved. Ordinary shuffled minibatches already provide random sampling; shorter “epochs” have no inherent information-theoretic advantage.

For a uniform sample of size `k` from a pool of size `N`, the sample-mean gradient is unbiased for the full **example-averaged** gradient at a fixed parameter value. Token-normalized variable-length losses require corresponding token-budget accounting, rather than blindly applying the example-average identity. With matched objectives and update schedules, fresh random subsets and full-data shuffling are closely related SGD procedures. Their differences concern replacement, correlations, exposure counts, and optimization—not a guaranteed benefit from discarding data.

**Historical labeled diagnostic: fitted opportunity-ranking penalties decline during training.** This analysis is retained as context, not as the proposed method or a calibration requirement. I used the random trajectories to avoid fitting the hypothesis exclusively on a hard-selection trajectory. For each snapshot define:

```text
Z = 1 if student is wrong AND Qwen-72B is correct; otherwise 0
```

This is an observable **proxy-solvable student error**. It is a surrogate for a teaching opportunity, not a measurement of actual teachability or external-teacher trace utility. The score still uses the experiment's average Llama/Qwen proxy entropy; Qwen-72B correctness provides one explicitly defined target. Different proxy or teacher correctness targets could yield different fitted coefficients; no such target is used in the agreed selector.

This historical analysis converted student and proxy entropy to within-pool percentile ranks, `u_t` and `v`, then used the following score. It is not the current normalized-entropy method:

```text
historical_score = student_entropy_rank - lambda * proxy_entropy_rank
```

At six snapshots, I chose `lambda` from `{0, 0.1, ..., 2}` by AUROC on a fixed 70% partition of training-pool question IDs, then measured AUROC on the remaining IDs. This is a held-out **score-fitting** diagnostic; all these questions remain part of the language model's training pool. No claim of held-out student generalization follows.

| Student | Snapshot 0 | 10 | 50 | 99 | 149 | 199 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen: fitted proxy weight | 1.0 | 0.8 | 0.5 | 0.2 | 0.1 | 0.1 |
| Phi: fitted proxy weight | 0.8 | 0.5 | 0.4 | 0.2 | 0.1 | 0.0 |
| Llama: fitted proxy weight | 1.7 | 0.8 | 0.5 | 0.3 | 0.1 | 0.1 |

![Fitted proxy penalty and held-out-row opportunity discrimination](../artifacts/selection_report_audit/adaptive_proxy_diagnostic.png)

The pattern supports your intuition more directly than the original correctness-AUROC observation. At snapshot 50, the held-out-row AUROCs for identifying `Z_t` are:

| Student | Student entropy alone | Fixed rank gap, lambda = 1 | Fitted rank gap |
|---|---:|---:|---:|
| Qwen | 0.744 | 0.732 | 0.783 |
| Phi | 0.772 | 0.735 | 0.799 |
| Llama | 0.745 | 0.759 | 0.794 |

This is an exploratory result from one trajectory per student and one fixed row split. It has no downstream training validation yet. Late in training the target becomes rare: by snapshot 199, `Z_t` covers only approximately 0.6–0.8% of the pool, making coefficient estimates and small AUROC differences fragile. If retained in a publication, this exploratory figure needs uncertainty estimates and explicit separation from the label-free method.

The target matters enormously. When I fit the same nonnegative proxy penalty to **student error alone**, the chosen coefficient is almost always zero, including at initialization. When the coefficient can be negative, early error prediction favors *adding* proxy entropy. That is sensible: questions that confuse the proxy are often difficult for the student too. Subtracting proxy difficulty is useful for a different question: identifying errors on which a more capable model has something to offer.

There is also a potentially decisive limitation: the external trace generator is not Qwen-72B or Llama-70B, and many traces use gold-answer-conditioned correction. The generator may provide useful reasoning exactly where both proxies fail. Therefore, optimizing `Z_t` could improve proxy-opportunity prediction without improving distillation. The new experiment tests downstream selection performance directly; it does not optimize this surrogate.

**Full rationale: imperfect aleatoric estimation and a relative-error budget.** The decomposition motivates subtraction; an additional acquisition principle motivates decay. Neither the decomposition nor improving AUROC alone proves that proxy weight should decrease.

**1. Define the desired acquisition signal.** We would like to prioritize uncertainty the student can resolve through the available generated traces. Student epistemic uncertainty is a surrogate for that opportunity. This assumes the traces can address the student's knowledge gaps; uncertainty requiring different data or greater model capacity may not be resolved. Even genuinely reducible uncertainty does not guarantee that a particular trace is useful, and batch redundancy and variable trace costs remain relevant.

**2. State the working uncertainty model.** For each question x, suppose:

```text
student_entropy_t(x) = A(x) + E_student,t(x)
proxy_entropy(x)     = A(x) + E_proxy(x)
```

A is the shared aleatoric component. E_student,t and E_proxy are the student and proxy epistemic components. All are nonnegative and have finite second moments. Nothing is centered; no regression, intercept, zero-mean, independence, or zero-covariance assumption is used.

The proxy and reference question pool remain fixed, so A and E_proxy are fixed. The student component can change during training. The proxy is explicitly imperfect: its residual epistemic component need not be zero. The early-training hypothesis is that this component is small relative to the student's epistemic uncertainty; larger model size alone does not establish that hypothesis.

For separately trained models' normalized token entropies, this is a phenomenological model, not an identified Bayesian identity. It assumes that the scores reflect a shared irreducible component on comparable scales. Different vocabularies, normalization factors, and miscalibration can violate that assumption. The two saved entropy columns cannot recover the latent components. We use the model as a conditional motivation, not as a measured decomposition of uncertainty.

**3. Explain the attraction and failure of full subtraction.** Under the model:

```text
student_entropy_t - proxy_entropy = E_student,t - E_proxy
```

Full subtraction cancels A, but also subtracts the proxy's residual epistemic uncertainty. The result is a relative epistemic gap, not the student's epistemic uncertainty itself. If E_student,t is positive but no larger than E_proxy, clipping the gap at zero removes guided sampling probability even though the student may still have something to learn.

For example, let A = 0.10 and E_proxy = 0.20:

| Quantity | Early | Late |
|---|---:|---:|
| Student epistemic component | 0.60 | 0.10 |
| Student normalized entropy | 0.70 | 0.20 |
| Proxy normalized entropy | 0.30 | 0.30 |
| Full-subtraction score | 0.40 | -0.10 |

This is an illustrative configuration of the working model, not a measurement from the runs. Uniform exploration preserves eligibility when guided gain is zero; it does not repair the score's interpretation.

**4. Expose the trade-off in partial subtraction.** For 0 <= lambda_t <= 1:

```text
score_t = student_entropy_t - lambda_t*proxy_entropy
        = E_student,t + (1 - lambda_t)*A - lambda_t*E_proxy
```

The two distortions are retained aleatoric uncertainty, `(1 - lambda_t)*A`, and erroneous epistemic subtraction, `lambda_t*E_proxy`. Reducing lambda decreases the latter but increases the former. Neither endpoint is universally preferable. The proxy residual need not literally be part of the student's epistemic component; subtracting it can suppress the desired signal numerically.

**5. State the extra acquisition principle.** Our proposed design criterion is:

> Remove as much aleatoric uncertainty as possible while keeping proxy-induced over-subtraction small relative to the student's remaining epistemic signal.

This is an explicit preference for preserving remaining learning opportunities, not a theorem about entropy or a statement that total estimation error is minimized. Define magnitudes across the fixed candidate pool:

```text
A_size      = sqrt(mean(A(x)^2))
proxy_error = sqrt(mean(E_proxy(x)^2))
student_t   = sqrt(mean(E_student,t(x)^2))
```

These are root-mean-square magnitudes of the original nonnegative quantities. The mean is across questions, not across tokens or seeds. No mean is subtracted. Let kappa > 0 be a fixed tolerated fraction, such as a fraction below one. The criterion becomes:

```text
minimize over lambda:
    (1 - lambda)*A_size

subject to:
    lambda*proxy_error <= kappa*student_t
    0 <= lambda <= 1
```

The objective measures the magnitude of retained aleatoric uncertainty. The constraint bounds the magnitude of erroneous proxy subtraction relative to the desired signal. By homogeneity of RMS, these are exactly the component magnitudes for a single coefficient shared by the pool. No assumption about correlation between components is needed. This budget controls an aggregate magnitude, not every question or the score's total error.

**6. Solve the constrained problem.** Assume A_size > 0 and proxy_error > 0. The objective decreases strictly with lambda. The constraint gives an upper bound, so the largest feasible coefficient is the unique solution:

```text
lambda_budget,t = min(1, kappa*student_t/proxy_error)
```

Boundary cases:

- If proxy_error = 0, the proxy is perfect within the working model and full subtraction is permitted; it minimizes retained aleatoric uncertainty when A_size > 0.
- If student_t is large enough that kappa*student_t >= proxy_error, full subtraction satisfies the budget.
- If student_t = 0 and proxy_error > 0, the budget forces lambda = 0.
- If A_size = 0, the objective is flat. Zero correction is a feasible optimal choice, but the criterion no longer specifies a unique coefficient.

This is an optimum for the stated constrained criterion. It is not an optimum for downstream accuracy, unconstrained mean squared error, or the implemented sampling weights.

**7. Add the learning assumption and derive decay.** Assume student_t is nonincreasing over training on the fixed reference pool, while proxy_error remains fixed and positive. Then lambda_budget,t is nonincreasing. It may initially remain at one and decrease only after the constraint becomes active. If student_t tends to zero, the coefficient tends to zero.

The chain is:

```text
student learns and its epistemic RMS decreases
    -> fixed proxy error grows relative to the remaining signal
    -> the same relative-error tolerance permits less subtraction
    -> proxy weight decreases once the constraint is active
```

The proxy does not deteriorate intrinsically. Its residual error becomes less tolerable relative to the student's remaining signal. Aggregate epistemic decline is an assumption; learning need not improve every question. Improving student AUROC does not establish this assumption because AUROC measures discrimination for a specified outcome, not epistemic magnitude. A changing valid pool can also change measured diagnostics without demonstrating learning on a fixed population.

**8. Connect the unobservable ideal to the empirical schedule.** The latent RMS quantities are unavailable from the two entropy columns, so lambda_budget,t cannot be computed by the current method. We approximate its hypothesized decline with:

```text
progress = t / (T - 1)                # t = 0,...,T-1; T >= 2
lambda_t = (1 - progress)^2
score_t  = saved_normalized_student_entropy_t
           - lambda_t*saved_normalized_proxy_entropy
```

The decomposition and constrained criterion motivate the direction of decay conditionally. They do not derive the quadratic curve, initial weight 1, or zero endpoint. The ideal rule can have an initial plateau; the empirical curve does not. Finishing training does not establish zero student epistemic uncertainty. Do not claim that the empirical schedule is guaranteed to satisfy the latent budget.

Kappa belongs to the explanatory criterion; it is not an additional fitted parameter or required sweep in the practical method. The single scheduled coefficient remains the agreed configuration across seeds. Existing student estimates update each round, while proxy entropies remain precomputed. Selection needs no labels, additional proxy inference, probability vectors, ranks, or centering.

**9. State the empirical claim and its limits.** The method accepts more aleatoric contamination to reduce relative over-subtraction. Whether that trade-off improves acquisition is the experimental question. Positive-part clipping and uniform exploration are additional practical choices, not derived optima. A random-baseline comparison tests the full policy, not the decomposition or the decay mechanism in isolation.

The paper can say:

> We model proxy entropy as an imperfect estimate of shared aleatoric uncertainty, contaminated by residual proxy epistemic uncertainty. Subtracting proxy entropy can therefore suppress the student's desired acquisition signal. We motivate a diminishing correction by limiting the magnitude of this over-subtraction relative to the student's remaining epistemic uncertainty. Under decreasing student epistemic uncertainty and a fixed imperfect proxy, this criterion yields a nonincreasing proxy weight. Since the latent components are unavailable, we evaluate an empirical decay schedule.

This is a conditional mathematical rationale for an empirical method. It does not establish that the proxy initially estimates a common uncertainty target better than the student, identify true aleatoric uncertainty, or prove superiority over random selection. The primary uncertainty literature provides context for these interpretive limits: [BALD](https://arxiv.org/abs/1112.5745), [Wimmer et al.](https://arxiv.org/abs/2209.03302), and [Bickford Smith et al.](https://arxiv.org/abs/2412.20892).


**Sampling rule.** Let N count every candidate question, including those with missing measurements:

```text
gain[i] = max(score[i], 0)            # zero for missing measurements

if sum(gain) > 0:
    weight[i] = 0.5/N + 0.5*gain[i]/sum(gain)
else:
    weight[i] = 1/N

key[i] = weight[i] / (-log(random_uniform[i]))
# fresh independent random_uniform[i] strictly between 0 and 1
# select the K largest keys, without replacement
```

Every question remains eligible. Generate reproducible random keys independently of answer parsing. Do not center the score before clipping. Positive gain means the student entropy exceeds the weighted proxy entropy, not that the gap exceeds its pool mean. The uncertainty decomposition motivates subtraction, and the relative-error budget conditionally motivates decay. Neither establishes this zero threshold or these sampling weights as optimal.

The 50% uniform mass and positive-part transformation are empirical sampling choices. They do not follow from the relative-error-budget argument or require exactly half the selected batch to be uniform. Weights describe first-draw probabilities, not marginal inclusion probabilities. At zero proxy weight, this becomes sampling proportional to normalized student entropy plus uniform exploration. Guided mass remains 50% whenever gains are nonzero, even if their magnitude becomes small.

**Historical rank-correlation diagnostic.** These nonnegative regression coefficients were computed on percentile ranks in one saved random trajectory per student. They describe entropy association, not epistemic RMS, the latent relative-error budget, or coefficients for the current normalized-entropy method:

| Student | Snapshot 0 | 50 | 99 | 199 |
|---|---:|---:|---:|---:|
| Qwen | 0.462 | 0.436 | 0.350 | 0.243 |
| Phi | 0.589 | 0.443 | 0.333 | 0.000 |
| Llama | 0.421 | 0.398 | 0.352 | 0.211 |

Reproduce this diagnostic with `python src/analysis/residual_score_diagnostics.py`. Phi's late raw rank covariance is slightly negative and the reported coefficient is clipped to zero. Neither that sign nor the overall declining association identifies aleatoric or epistemic components. The earlier labeled opportunity-ranking diagnostic optimizes a different target again; both analyses remain retrospective context, not inputs to selection.

**Random's efficiency needs an exposure calculation, not an epoch label.** For independent uniform subsets of size `k` per round,

```text
expected_unique_questions = N * (1 - (1 - k/N)^T)
expected_exposures_per_question = T*k/N
```

With `N=9,626` and `k=1,024`, random covers approximately 89.5% of examples after 20 rounds, 99.6% after 50, and almost all after 100. The saved runs closely resemble this coverage pattern. One hundred rounds present 102,400 traces—10.64 full-corpus equivalents—plus 25,600 direct-answer examples. Two hundred rounds present 21.28 corpus equivalents. This is not a 90% teacher-generation saving.

If generation is cached, cost depends on cumulative unique selected prompts and their trace lengths. Under long random runs, almost every trace must eventually be generated. If generation is refreshed on every selection, count every generation request instead. These are different protocols. In this repository traces are already available offline, so on-demand generation savings are counterfactual unless the selected-ID union and generation costs are explicitly reconstructed.

Training-token savings may still exist, especially at an early accuracy threshold. Measure them directly. Also count student scoring prefill over the entire pool: one generated token does not mean one token of computation. The current random baseline even pays for student entropy estimation because random numbers are emitted by that estimator. A production random policy should draw IDs directly and have no model-scoring overhead. [Repeated Random Sampling](https://arxiv.org/abs/2305.18424) already makes random resampling and time-to-accuracy central; do not present resampling itself as novel. [Rethinking Data Selection at Scale](https://aclanthology.org/2025.findings-emnlp.146/) also documents strong random baselines for instruction tuning.

For the short paper, report at least: cumulative supervised target tokens; actual optimizer updates and their learning-rate schedule; scoring time; unique generated traces and generation tokens; held-out accuracy at a fixed inference cap. A nominal sample-view plot is useful diagnostically, but is not a compute-matched frontier:

![Accuracy against nominal trace exposure, illustrative only](../artifacts/selection_report_audit/accuracy_vs_nominal_trace_exposure.png)

**A few concrete audit issues should be resolved before new headline runs.** These are observed properties of this checkout or its artifacts, not assumed causes of the performance ordering.

- Historical hard entropy-gain dumps use raw entropy units: their mean proxy value is about 0.2324. New random/proportional dumps use normalized values, about 0.01956. Current code normalizes by `log|V|`. Both student and proxy appear to use the historical scale within the older runs, so this is not evidence of a universal raw-versus-normalized subtraction bug. Student-only AUROC is invariant to scaling. However, freeze normalization and record the code revision; current source is not an exact execution manifest for every historical result.
- The random estimator generates its random key only after output verification. Failed measurements can therefore exclude rows from the supposedly uniform draw. One saved Qwen random snapshot has 2,804 missing random keys out of 9,626. This does not invalidate the entire baseline, but it means it is not always unconditional uniform sampling. Separate random selection from answer parsing in new runs. Seed the estimation worker or derive each draw from `(run seed, round, question ID)`; the current worker does not seed Python's random generator.
- In the inspected snapshots, reconstructed trace subsets always contained 1,024 examples. The known possibility that positive-gap filtering shrinks the dataset does not explain these particular runs. Keep actual size logging, but do not repeat the earlier report's schedule-shrink hypothesis as an established mechanism here.
- `corrected_answer` supplies the gold answer during correction and marks corrected answer correctness as true. This is legitimate gold-conditioned synthetic supervision when disclosed. It is not independent evidence that the resulting reasoning is correct, nor is it evidence of test-answer leakage. It particularly weakens the claim that proxy uncertainty measures whether the external teacher can supply a useful trace.
- Train/test question IDs are disjoint, but 114 test rows repeat a training question string; 47 repeat the question and ordered options exactly. I computed sensitivity results excluding both categories. For example, Phi's shuffled proportional-gap result at round 150/cap 4,096 changes from 63.51% to 63.03% after excluding exact question/options overlap; random seed 42 changes from 59.89% to 59.56%. The ordering survives this example. Use grouped deduplication for any new calibration split and report the sensitivity, rather than pretending disjoint IDs imply disjoint content.
- One checked summary disagrees with its saved responses: Llama proportional student entropy, shuffled seed 43, checkpoint 1000, cap 4,096 reports 52.20% in the summary but 51.41% in `responses.parquet`. Reconcile provenance before using that point. The report's central tables do not depend on this conflicting point.
- Existing static-versus-resampled comparisons also change the auxiliary objective: for example, `entropy_gain_once` uses a trace-only adapter. They do not cleanly isolate reselection. Hold the direct-answer stream fixed in the static control.
- The `complexity_roc_auc_dynamics` notebook's top-k analysis ranks proportional methods by gap, not by their stochastic race key. It describes a hypothetical high-gap subset, not the actual proportional draw. Its stored execution also contains an error, so the numbers in this report were recomputed independently.

**Use the remaining budget for one fixed configuration across seeds.** Implement the score and sampling rule above, then freeze them. There is no labeled calibration set, coefficient sweep, new proxy inference, extra student family requirement, or multi-arm continuation campaign in this plan.

| Time | Work | Deliverable |
|---|---|---|
| Days 1–3 | Implement and verify direct normalized-entropy scoring, coefficient scheduling, missing-value handling, and reproducible sampling. Freeze seeds, rounds, learning-rate schedule, trace/direct-answer mixture, shuffle behavior, evaluation checkpoints, and inference caps to match the random reference. | One reviewable configuration and selection manifests. |
| Days 3–14 | Train that configuration with the available seeds. Log coefficients, valid-row counts, coverage, repeated exposure, target tokens, and scoring cost. | One comparison against the existing random baseline. |
| Days 15–17 | Evaluate fixed checkpoints and report seed variability, inference-cap sensitivity, and existing duplicate sensitivity. | A central result table and a coefficient/coverage figure. |
| Days 18–21 | Write the short paper and freeze reproducibility materials. | A submission with claims limited to the evidence. |

Existing random runs are usable as the reference only to the extent that their objectives, schedules, and sampling behavior match. In particular, their documented missing-random-key issue cannot be repaired retrospectively. Disclose any mismatch; if a fair comparison would require unavailable additional training, do not describe the result as an isolated causal effect of the score.

Use a fixed checkpoint and primary inference cap for the main comparison, not each method's best test-set checkpoint. Report training-seed dispersion and paired differences on shared evaluation items. Three seeds provide limited evidence about run variability. If item bootstrap intervals are included, repeated question content calls for grouping. Failure to detect a difference is not evidence of equivalence.

The current pool covers all 14 MMLU-Pro categories, including humanities and social science; it is not STEM-only. A new holdout cannot be created by renaming categories already used for training. No new evaluation split or additional dataset is required for this limited experiment.

**A short paper can center on diminishing proxy correction, provided its empirical value is demonstrated.** The proposed mechanism is that fixed proxy epistemic error becomes increasingly consequential relative to the student's remaining epistemic uncertainty. The relative-error budget makes the assumptions and conditional decay result explicit; the actual schedule remains empirical. The contribution is the selection rule and its evaluation, not a new uncertainty-decomposition theorem. A matched-budget improvement over random would support the complete method. A claim that the proxy itself adds value requires a matched student-only ablation, which is outside the current budget unless an existing run genuinely matches.

If the method does not outperform random, report that result. The existing policy-dependent AUROC, coverage, exposure, and inference-cap patterns can support a focused empirical discussion, but neither they nor an elegant projection identity establish a new successful selection method. Claims explaining random's advantage over full distillation remain limited by objective and optimization differences.

Keep the main paper to the problem, the scheduled score and its imperfect-proxy/relative-error rationale, one results table, one dynamics figure, and a short limitations/related-work discussion. The full artifact audit and historical labeled diagnostics belong in supplementary material. Cite the prior paper as the static-selection starting point without directly comparing its older balanced-split accuracies to the new random split.

**Reproduction and supporting files.** The analysis scripts only read existing model artifacts and write analysis outputs; they do not change production samplers, train models, or alter the source datasets.

```bash
python src/analysis/selection_report_audit.py
python src/analysis/selection_report_diagnostics.py
python src/analysis/residual_score_diagnostics.py
```

The scripts require pandas, NumPy, PyArrow, scikit-learn, and Matplotlib, available in the inspected environment. Results are in `artifacts/selection_report_audit/`:

- [Evaluation curves](../artifacts/selection_report_audit/evaluation_curves.csv), [descriptive peaks](../artifacts/selection_report_audit/descriptive_peaks.csv), and [final evaluations](../artifacts/selection_report_audit/final_evaluations.csv).
- [Selection dynamics](../artifacts/selection_report_audit/selection_dynamics.csv), including AUROC, coverage, retention, and missing random keys.
- Historical [entropy-only residual coefficients](../artifacts/selection_report_audit/residual_coefficients.csv), reproduced by the [residual diagnostic script](../src/analysis/residual_score_diagnostics.py); this script reads only the two entropy columns.
- Historical labeled [coefficient diagnostics](../artifacts/selection_report_audit/coefficient_diagnostics.csv) and [opportunity diagnostics](../artifacts/selection_report_audit/opportunity_diagnostics.csv).
- [Random transition cells](../artifacts/selection_report_audit/random_transition_cells.csv), retained by model, round, score quintile, assignment arm, and initial correctness rather than just pooled totals.
- [Split audit](../artifacts/selection_report_audit/split_audit.json), [deduplication sensitivity](../artifacts/selection_report_audit/deduplication_sensitivity.csv), and [summary/response discrepancies](../artifacts/selection_report_audit/summary_response_mismatches.csv).
- [Audit script](../src/analysis/selection_report_audit.py) and [additional diagnostics script](../src/analysis/selection_report_diagnostics.py).

The earlier strategy documents in `docs/` refer to older artifact states. Their missing-baseline statements, raw-scale diagnoses, and proposed experiment inventories should not be treated as the current evidence inventory.
