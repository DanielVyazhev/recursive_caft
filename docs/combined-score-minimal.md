**One combined score for the remaining experiment**

Updated 21 September 2026. Use an empirical diminishing proxy correction with uniform exploration. This specification uses normalized entropy directly, replacing percentile ranks and the earlier covariance-fitted residual method. It includes the full derivation; the [full report](adaptive-proxy-selection-report-2026-09-19.md) contains the experimental evidence and audit. Run one fixed configuration with multiple seeds. Selection uses only the existing scalar student and proxy entropies; student estimates update each round through the existing estimator and the proxy column remains precomputed.

**The score.** Use the saved normalized student and proxy entropies directly, with no centering:

```text
s = saved_normalized_student_entropy
p = saved_normalized_proxy_entropy

progress     = t / (T - 1)            # t = 0,...,T-1; T >= 2
proxy_weight = (1 - progress)^2
score        = s - proxy_weight*p
```

The initial weight 1 and exponent 2 are fixed empirical choices, not estimated optimal coefficients. They give weights 1, 0.5625, 0.25, 0.0625, and 0 at 0%, 25%, 50%, 75%, and 100% of training. No correctness labels, cross-entropy, probability vectors, calibration split, or coefficient sweep are used. Use the existing `H / log(vocabulary_size)` values, with each model's own vocabulary and consistent log bases; do not normalize them again. Retain the existing proxy column, the mean of the two proxies' normalized entropies. Do not apply percentile ranks or divide by per-round standard deviations. Normalization does not calibrate the models or align their vocabularies. The initial coefficient 1 remains an empirical choice on this representation, not a transferred optimum from ranks.

Scoring requires no pool means, ranks, or variances. Missing or nonfinite entropy pairs receive zero guided gain and remain eligible through uniform exploration. With no valid pairs, sample uniformly. The theoretical RMS quantities below describe magnitudes across questions, not tokens or seeds; they are not computed by the selector.

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

Match the existing random reference's total rounds, learning-rate schedule, trace/direct-answer mixture, ordering, and evaluation caps. Log weights, valid-row counts, selected IDs, coverage, repeat exposure, target tokens, and scoring cost. Compare the same fixed checkpoint across seeds. Correctness-based AUROC is retrospective analysis only. A comparison with random tests the complete policy; it does not isolate the proxy correction or prove the latent mechanism.

Suggested paper description: “Motivated by controlling proxy-induced over-subtraction relative to the student's remaining epistemic uncertainty, we empirically decrease proxy correction during training and retain uniform exploration.” The proposed mechanism and schedule remain unvalidated in training; no training code has been changed by this report update.
