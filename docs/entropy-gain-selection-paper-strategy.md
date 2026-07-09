# Report: Entropy-gain data selection for offline reasoning distillation — paper strategy for ICLR

*Compiled 2026-07-07. Basis: full read of the prior paper (arXiv:2506.21220v4), all of `src/experiments`, `src/core` (samplers, `ResamplingTrainer`, evaluator), `src/analysis` notebooks with outputs, `artifacts/`, `docs/tracker.md`, and a sweep of 2025–26 related work.*

---

## 0. Where we actually stand (evidence inventory)

**What exists** (all Qwen2.5-3B, MMLU-Pro STEM, train 9,626 / test 2,406, single seed 42):

| Config | Data/epoch | Best acc | Source |
|---|---|---|---|
| Entropy-gain, corrected-answer traces | ~10.6% (1024+256) | **0.5503** @ ep50 | `artifacts/distillation_by_metrics/.../entropy_gain_corrected_answer/` |
| Entropy-gain, direct traces | ~10.6% | 0.5308 @ ep20 | `.../entropy_gain_direct_reasoning_trace/` |
| Full-corpus, corrected-answer | 100% | 0.5320 @ ep10 | `distillation_on_synthetic_traces/` |
| Full-corpus, direct traces | 100% | 0.5220 @ ep8 | same |
| Base model CoT | — | 0.4156 | `base_model/qwen_3b_cot/` |

**What does NOT exist yet:** student-entropy-only run (script exists, unrun), proxy-entropy-only, entropy-ratio, **random selection at matched budget**, any second student, any second dataset for the online loop, any second seed.

**Statistical reality:** at n=2406, binomial SE ≈ 1.0 accuracy point. The headline gap (+1.83 pts over the best full-corpus baseline) is ~1.8 SE from a single seed. At *matched* sample-view budget (ep10 ≈ one corpus-equivalent), entropy gain roughly ties the full-corpus peak (0.5237 vs 0.5320). The +1.83 win uses ~5.3 corpus-equivalent passes over the selected subsets.

**Known mechanical issues (from code review):**
1. Selection entropy is measured single-token/no-CoT while training and eval are CoT (`max_new_tokens=1`, no `<think>` prefix in `complexity_evaluation_dataset`); up to 10% unparseable estimations per epoch are backfilled from the previous epoch — the selection signal degrades as the student becomes CoT-native.
2. Notebook metrics use max-normalized entropies; the training sampler uses raw nats (`entropy_gain_sampler.py`) — the design study and the production run rank differently.
3. `entropy_ratio_sampler.py` also defines a class named `EntropyGainSampler` (copy-paste hazard).
4. `ResamplingDataset.__len__` returns the top-k cap while `drop_non_positive` shrinks actual yield → cosine LR schedule never fully decays (acknowledged in comments).
5. The 1024-reasoning and 256-single-token adapters draw from the same pool → duplicates with mixed objectives.
6. `corrected_answer` and `explained_answer` traces are generated with the gold answer in the teacher prompt and `distill_ans_correct` forced to True — the **best number rides on answer-leaked, post-hoc-rationalized traces**.
7. Unseeded `np.random.choice` in `splitter.py` train/test chunking.

---

## 1. Ideas to improve the paper

### 1.1 Reframe entropy gain as approximate epistemic uncertainty / learnability (the theory hook)
Right now "entropy gain" is a heuristic. Give it a story:
- Proxy entropy ≈ **aleatoric** uncertainty of the question (intrinsic ambiguity/difficulty — what no amount of student training removes). Student entropy ≈ aleatoric + **epistemic**. So `H_s − H_p` ≈ the student's *reducible* uncertainty — exactly what training can fix. The notebook even used the name "BALD+"; make the BALD connection explicit (Houlsby et al. 2011; total − expected-aleatoric decomposition), and the `docs/tracker.md` note "tie EIG to entropy windows, see BED-LLM" is the same thread — entropy gain as a cheap proxy for expected information gain.
- This maps 1:1 onto **RHO-LOSS** (Mindermann et al., ICML 2022, arXiv:2206.07137: select by training loss − holdout-reference loss = "learnable, worth learning, not yet learnt"). Entropy gain is its label-free, single-forward-pass, entropy-space analog. Write a short table mapping the notebook metrics onto the RHO decomposition: high `H_s` = "not yet learnt", low `H_p` = "learnable", the LI metric (`EG × proxy-confidence`) = adds "worth learning". Reviewers *will* find RHO-LOSS; preempt them and differentiate: (a) no labels needed to score, (b) no reference-model training on holdout data, (c) one 1-token forward pass per example vs a full trace loss.

### 1.2 Restructure the claims — two defensible claims instead of one fragile one
- **Claim A (efficiency):** at matched total compute, per-epoch entropy-gain selection *matches* full-corpus distillation while needing teacher traces for only the cumulative-selected subset → report **cumulative unique examples selected** across epochs. If that's ~25–40% of the corpus, there is a real "generate 60–75% fewer teacher traces" economic claim (teacher tokens are the expensive resource). This also makes the "active learning" framing honest: generate traces *on demand* for selected items only.
- **Claim B (quality):** with extended training, selection *exceeds* the full-corpus ceiling (0.550 vs 0.532) — evidence that some data actively hurts (Hypothesis 1). Support it directly: train on the *complement* (the never-selected bottom slice) and on the "confident-noise" region and show degradation.
- Keep the two claims on separate axes (a compute-vs-accuracy frontier plot with cumulative sample-views on x — the analysis notebook already does this; make it the main figure).

### 1.3 Statistical rigor (cheapest paper upgrade available)
- ≥3 seeds for every headline config; report mean±std.
- Paired tests on the shared 2,406 eval items (McNemar or paired bootstrap) — paired tests are much more powerful than comparing two accuracies.
- Consider enlarging eval: MMLU-Pro non-STEM as an OOD/held-out eval costs nothing to add and doubles as a generalization check.
- Seed the splitter; drop or justify `torch_compile` nondeterminism.

### 1.4 Honest cost accounting
Report end-to-end: teacher trace-generation tokens (per method, cumulative-unique), per-epoch selection cost (a 1-token forward pass over 9,626 questions — cheap, but say so with numbers), training FLOPs/tokens, and eval cost. The prior paper's "81% less data" framing was attack-proof because it counted tokens; do the same here.

### 1.5 Method hygiene before the long runs
- **Unify normalization**: pick one (recommend rank/quantile normalization per model — robust to outliers and to vocab-size differences between student and proxy; raw-nat differences across different tokenizers/vocabs are not comparable) and use it in both analysis and training. Full treatment in §7.
- Fix the LR-schedule/`__len__` mismatch, the duplicate class name, and dedupe the two adapters' pools (or make the overlap a deliberate, documented choice).
- Decide on the metric-objective mismatch: either measure selection entropy in the CoT regime (e.g., entropy of the answer token after a forced short `<think>` prefix, or first-token entropy after `<think>`), or keep single-token and *show* it still tracks CoT correctness as training progresses (plot per-epoch failed-estimation fraction and ROC-AUC of the online entropy vs current-student CoT correctness). Right now this is the biggest internal-validity hole.
- **Corrected-answer leakage**: the best result uses traces where the teacher was told the gold answer. Either (a) regenerate with answer-hidden prompting + rejection sampling to correctness, or (b) keep both trace types as an explicit ablation axis with the caveat stated. Don't let the headline number silently depend on it.

### 1.6 Positioning and terminology
- Avoid "active learning" as the main frame unless adopting the trace-on-demand story (labels/traces already exist offline ⇒ classic AL reviewers will object). "Dynamic/online data selection for reasoning distillation" is accurate.
- Name the method once, use it consistently (Recursive/adaptive entropy-gain selection; "recursive CAFT" is a good brand continuous with the prior paper).
- Cite and differentiate the close 2025–26 work up front: LARK (arXiv:2605.30651; learnability = loss-decrease rate + coverage regularizer), skill-aware selection for reasoning distillation (arXiv:2601.10109), DC-CoT data-centric benchmark (arXiv:2505.18759), entropy-aware on-policy distillation (EOPD, ICLR 2026, IBM), ELAD (already in `docs/review.pdf`). Differentiators: single-token entropy (1 forward token, no trace generation, no gradients), per-epoch recursion on the current student, and the proxy-referenced gain.

---

## 2. Most promising directions (2–3 months, part-time, 2×H100 — ranked by value/GPU-hour)

**P0 — The ablation grid that makes or breaks the paper** (infrastructure already exists):
On Qwen-3B / direct traces / fixed budget 1024/epoch / 20 epochs: {entropy-gain, student-entropy, proxy-entropy, entropy-ratio, **random-resampled-each-epoch**, **random-fixed-subset**} × 3 seeds ≈ 18 short runs. This single grid answers: does the proxy term add anything over the prior student-entropy method? Does *any* scoring beat random at matched budget+schedule?

**P0 — Seeds + significance for the headline configs** (folded into the grid).

**P1 — Static vs adaptive ablation** (the "recursive" in recursive_caft): select top-1024 once with the epoch-0 student entropy and train 20 epochs on that fixed set, vs re-selecting each epoch. If adaptivity doesn't beat static, the method simplifies dramatically (and the paper changes); if it does, the core contribution is isolated. Also report **selection churn** (Jaccard overlap of consecutive epochs' subsets) — cheap, insightful figure.

**P1 — Proxy ablation, nearly free:** single-token entropy parquets already exist for 7 models (3B→72B) × 3 datasets. Offline: recompute EG with each proxy and measure HV-rate; online: 3–4 training runs swapping the proxy (qwen_32b alone, mistral_24b alone, llama70b+qwen72b ensemble, and a 3B proxy). "How small can the proxy be?" is a question every reviewer and practitioner will ask, and it's the cheapest new-results section.

**P1 — Second dataset + second student:** GSM8K (7.5k train, entropy data exists) and Phi-4-mini or Llama-3B for the main method. One dataset/one model is a desk-reject risk at ICLR. MedMCQA (from the prior paper) as the third if time allows; GPQA (546 rows) as OOD eval only.

**P2 — RHO-LOSS baseline:** score = student answer-token loss − proxy answer-token loss (both computable from the same forward passes already run). One extra sampler class + the grid row. Highest-value external baseline.

**P2 — The 2D entropy-plane analysis** (see §3, "cartography") + the harmful-data demonstration: train on the never-selected/confident-noise region and show it hurts. Directly proves Hypotheses 1–2 instead of implying them.

Rough budget: the P0 grid at 20 epochs × 1024 samples is ~1/5 the cost of one 50-epoch run per cell; the whole P0+P1 program is plausibly 3–5 weeks of part-time 2×H100 wall-clock, leaving a month for the second dataset/model and writing.

---

## 3. Most original directions (pick 1–2, not all)

1. **Uncertainty-routed distillation (unifies both papers).** Use the (student-entropy × proxy-entropy) plane to route each example per epoch: low-`H_s` → skip; high-`H_s`/low-`H_p` (learnable) → full CoT distillation; high-`H_s`/high-`H_p` (ambiguous/noise) → skip or answer-only SFT. The prior paper routed *statically* into SFT-vs-distillation; this makes routing *dynamic and 2D*. It's a bigger story than subset selection alone, reuses all existing machinery, and gives the paper a memorable figure (the entropy plane with region labels — the LLM-era analog of Dataset Cartography, Swayamdipta et al. 2020).
2. **Proxy-free self-gain:** replace the proxy with the *frozen epoch-0 student* (or an EMA): select items where entropy has *not* decreased (not-yet-learnt) but is high (uncertain). Removes the extra-model requirement entirely — the strongest practical selling point if it works, and no published equivalent for reasoning distillation found.
3. **Token-budgeted selection (knapsack):** high-entropy questions tend to have long traces (`reasoning_length_vs_accuracy` artifacts: incorrect/hard items run 1.9k+ tokens vs ~400). Select to maximize Σscore under a *token* budget rather than example count. Turns "10% of examples" into the honest "X% of training tokens" and connects to the truncation-mode results.
4. **Selection-quality theory via EIG:** formalize entropy gain as a bound/approximation of expected information gain under a two-model Bayesian setup (the BED-LLM thread in the tracker). Even a 1-page proposition + synthetic validation elevates the paper's tier.
5. **Cross-student transfer of subsets:** does Qwen-selected data help Llama? If yes → selection is a property of the data (precompute once, amortize); if no → adaptivity is essential. Either answer is a publishable finding, and it's two training runs.

---

## 4. Baselines — extensive list with rationale

### Tier 1 — required; absence is a rejection risk (**bold = most important**)
| Baseline | Why |
|---|---|
| **Random resampled each epoch, matched budget & schedule** | THE control. Holds budget, curriculum re-sort, LR schedule, and fresh-subset stochasticity constant; only the scoring function differs. If EG doesn't beat this, nothing else matters. Currently missing. |
| **Random fixed 10% subset** | Separates "less data per epoch" from "adaptively chosen data". With the resampled variant, brackets the value of (a) rotation and (b) scoring. |
| **Full-corpus distillation** | The ceiling claimed to be matched/beaten; already run — add seeds. |
| **Student-entropy top-k (prior method)** | Self-baseline: isolates the proxy term's marginal value — the paper's central delta. Script exists, unrun. |
| **Proxy-entropy top-k (static difficulty)** | Isolates the student term / adaptivity; equals "hard-for-a-competent-model" curation. |
| **Static entropy-gain (select once at epoch 0)** | Isolates recursion — the titular contribution. |

### Tier 2 — strongly recommended (external methods, cheap to implement in the sampler API)
| Baseline | Why |
|---|---|
| **RHO-LOSS-style excess loss** (arXiv:2206.07137) | Nearest published relative; loss-space analog of EG. Differentiating from it convincingly requires running it. Watch its known length bias (Irreducible Curriculum, arXiv:2310.15389, fixed via reference-loss normalization). |
| High student trace-loss top-k | Classic hard-example mining; tests whether cheap 1-token entropy matches expensive full-trace loss signals. |
| InfoBatch-style dynamic loss pruning (arXiv:2303.04947) | The established per-epoch dynamic-pruning method (ICLR'24 oral); validates the dynamic setting against non-entropy dynamic selection. |
| Margin / least-confidence (1 − p_top1, p_top1 − p_top2) | Standard uncertainty-sampling alternatives from the same logits — free ablation showing entropy specifically matters (or doesn't). |
| Trace-length selection (longest / shortest) | Embarrassingly cheap difficulty proxy; the length–accuracy artifacts show length correlates with difficulty, so reviewers will ask. |
| Easy→hard curriculum over full corpus (proxy-entropy ordered) | Connects to the prior paper's curriculum baseline; the sampler already re-sorts easy→hard *within* the subset, so disentangle subset-choice from ordering. |
| Middle-band ("window") selection | Directly tests Hypothesis 2 (extremes are bad); the Window-* metrics from the notebooks, aligned with Sorscher et al. 2022 (arXiv:2206.14486; best-kept difficulty depends on data regime). |

### Tier 3 — cite and discuss; run only if a reviewer forces it
- **LESS** (arXiv:2402.04333; gradient-influence selection) — strong but needs gradient stores; justify skipping by cost.
- **S2L** (arXiv:2403.07384; small-model loss-trajectory clustering) — cited in the prior paper; conceptually adjacent (small model as proxy).
- **IFD / Superfiltering** (arXiv:2402.00530), DEITA, AlpaGasus — instruction-data quality selection; different regime (quality, not difficulty).
- GraNd/EL2N, forgetting events — vision-era pruning scores; mention in related work.
- **LIMO** (arXiv:2502.03387) / **s1** (arXiv:2501.19393) — tiny-curated-set reasoning SFT; frame as static, human/heuristic curation vs automatic dynamic selection.
- **DC-CoT** (arXiv:2505.18759) — adopt any of its standardized selection baselines that overlap; citing a benchmark shows awareness.
- **On-policy/logit KD (GKD, MiniLLM, EOPD)** — orthogonal axis (how to distill vs what data); discuss, don't compare.
- **LARK** (arXiv:2605.30651), skill-aware selection (arXiv:2601.10109), ELAD — closest 2024–26 relatives; must be in related work with explicit differentiation (cost per scoring call, recursion, proxy-referenced gain).

---

## 5. What we're not thinking of — biggest risks first

1. **The random-resampled control could explain everything.** The 50-epoch selection runs beat baselines that peaked and overfit by epoch 10. Fresh 10% subsets each epoch act as a regularizer (data rotation) independent of *which* 10% is picked. Until random-resampled-at-matched-schedule is run, "entropy gain works" is unproven — repetition effects are well documented (data-constrained scaling, arXiv:2305.16264). **Run this first, before any new machinery.**
2. **EG may be indistinguishable from plain student entropy in practice.** If `H_s` and `H_s − H_p` rank the pool nearly identically (one static proxy vector shifts ranks only where the proxy is confidently wrong), the novel term adds nothing. One afternoon of offline analysis: Spearman correlation of scores + Jaccard of top-1024 sets across epochs, from parquets that already exist. Do this *before* burning GPU on the grid.
3. **The best number depends on answer-leaked traces** (corrected_answer, §1.5). If a reviewer catches it before it's disclosed, the whole results section loses credibility.
4. **Selection-signal drift**: the single-token entropy of a model being trained to always `<think>` is measured increasingly out-of-distribution (≤10% unparseable already tolerated). Plot the failure fraction and online ROC-AUC by epoch; if it degrades, that's a limitation to disclose (or the motivation for the CoT-regime entropy fix).
5. **Effect size vs noise**: +1.83 pts at SE≈1.0, single seed, one model, one dataset. As-is this is a workshop paper; the grid + seeds + second dataset is what makes it an ICLR paper.
6. **Coverage/diversity blindness of top-k**: pure top-k can concentrate in a few categories (entropy's ROC-AUC varies 0.59–0.84 by subject). Report per-category composition of selected sets; if skewed, stratified top-k or a light diversity term (LARK uses a coverage regularizer) is an easy fix and ablation.
7. **No OOD/generalization eval**: subset-trained models might overfit MMLU-Pro STEM style. Add MMLU-Pro non-STEM + GPQA eval of final checkpoints — near-zero cost, preempts a standard review question.
8. **Novelty clock is ticking**: LARK, skill-aware, AIR, HES all appeared in the last ~12 months. The moat is the *cheapness* of the signal (one forward token; no traces, no gradients, no judge) + recursion + proxy-gain. Lead with the cost-per-scored-example table.
9. **Unverifiable teacher identity**: trace generators are OpenRouter route strings (`deepseek/deepseek-v4-flash`, `qwen/qwen3.6-plus`). Pin down and report exactly what served those routes, or regenerate with a pinned open model — reproducibility reviewers will ask.
10. **ICLR 2027 timing**: deadlines unannounced, but by the ICLR 2026 pattern expect abstract ~Sept 19 / paper ~Sept 24, 2026 (AoE) — ~10–11 weeks away. Plan backwards: freeze experiments ~2 weeks before, keep the last week for writing only.

---

## 6. The signal-processing treatment (`active_learning_analysis`) — critical evaluation

*Added after an in-depth read of `all_models_analysis.ipynb` and `using_larger_model_as_techer_proxy.ipynb`, including all output tables.*

### 6.1 What the treatment is
Both notebooks score examples with 2D functions of max-normalized student entropy `Ĥs` and proxy entropy `Ĥt`, several framed in signal-processing terms:
- `Hs_var = Ĥs(1−Ĥs)` — labeled the "Fisher Information signal term" (signal power, peaked at mid-uncertainty);
- `Window-Wiener = Hs_var/(Hs_var+Ĥt)` — Wiener filter gain S/(S+N), with proxy entropy as noise power;
- `ER = Ĥs²/Ĥt²` (ex-"SNR") and `Window-Ratio = Hs_var/Ĥt` — SNR power ratios;
- `Window-Log (IDS) = log(1+Hs_var/Ĥt)` — the Shannon–Hartley capacity form log(1+S/N);
- plus `EG`, `LI = EG·(1−Ĥt)`, `Window-Product = Hs_var·(1−Ĥt)`.
Surrogate objective: HV rate (teacher correct ∧ student wrong) in the top-k selection, vs Random.

### 6.2 The decisive empirical fact: the two notebooks contradict each other, and the contradiction is explainable
**Notebook 1** (`all_models_analysis`): the *same* medium/large model provides both `Ĥt` and the "teacher correct" label. Averaged over all pairs at top-10%: **Window-Wiener/Log/Ratio win** (HV 44.4/44.2/44.2%, +62% lift), ER 43.7%, EG only 33.4% (+21.9%), and **plain student entropy H is worse than Random** (23.6% vs 27.4%).

**Notebook 2** (`using_larger_model_as_techer_proxy`): entropy from proxies, correctness labels from decoupled *true teachers* (gpt-oss-120B / Qwen3-235B, ~79% accurate, n=11,282). The ranking **inverts**: **H(s) wins** (HV 68.9%, +36.7%), H(t) 65.7%, EG 65.0% (+29.0%), LI 63.9% … and the **Wiener/SNR metrics fall below Random** (42.7–42.9%, −15%), actively selecting items the student already answers correctly (S.Acc ≈ 48% in their subsets vs 14.7% for H(s)). Consistent across both true teachers and both proxy ensembles.

**Why the flip:** in Notebook 1 the metrics that divide by `Ĥt` mechanically prefer items where the *label provider* is confident — inflating teacher-correctness in the subset (T.Acc 88% vs 79%) and therefore HV, by construction. It is label–metric circularity, not signal recovery. With decoupled labels from a strong teacher, that shortcut disappears: HV is then dominated by "find items the student gets wrong," which raw student entropy does best. **The SP metrics' apparent win is an evaluation artifact.** Anyone on the team still treating Window-Wiener as the best metric is optimizing the artifact.

Two secondary observations that survive the correction:
- **EG's real value is noise avoidance, not HV:** under true-teacher labels EG selects only 2.2% uncertainty-noise items vs 8.7% for H(s), at a modest HV cost. If wrong/noisy traces hurt training (plausible; unverified), EG could still win downstream — but that's a *different mechanism* than the one the lab measures.
- **The optimal metric depends on teacher reliability.** Weak label source (Notebook 1's 24–70B "teachers", 30–50% accurate on hard items) → proxy-confidence-seeking metrics pay; near-oracle teacher (Notebook 2, ~80%) → pure student uncertainty pays. This reconciles the two notebooks and is itself a publishable finding: a *phase diagram of selection metrics over teacher accuracy*.

### 6.3 Critique of the signal-processing formalism itself
1. **Semantic mismatch:** `Ĥs(1−Ĥs)` applies a Bernoulli-variance form to a normalized *entropy*, which is not a probability. The principled version of the "window" is in probability space: p(1−p) with p = student's probability of the gold answer — this is the actual variance/Fisher/gradient-signal quantity, it's computable from logits you already store, and it connects to established theory (expected gradient signal of CE/RL on an item peaks at p≈0.5; the same "learnability window" used in RL data selection by pass rate and matching Hypothesis 2 and Sorscher et al.).
2. **"Proxy entropy = noise power" is empirically false here.** Wiener/SNR require signal and noise to be additive and uncorrelated; H_s and H_t are strongly correlated (both track question difficulty), and H_t mixes aleatoric difficulty with the proxy's own epistemic gaps. Notebook 2 is a direct falsification of the noise model: metrics built on it drop below Random.
3. **No process, no spectrum:** each item contributes one static scalar pair; there is no ensemble/time structure over which Wiener filtering or channel capacity is defined. These are monotone 2D scoring heuristics wearing SP names — an SP-literate reviewer will say exactly this. Either formalize (see 6.4) or rename.
4. **Normalization fragility:** max-normalization makes every score dataset- and outlier-relative; `CONF_THR=0.30` and the noise taxonomy are defined on this fragile scale, and the taxonomy uses the same variable the metrics rank by (H(t)-based metrics get "0% confident noise" by construction — descriptive, not evidence).
5. **No uncertainty quantification:** at top-10% of n≈10–11k, k≈1k, HV differences of 1–3 pts are ~1–1.5 SE; several reported rank differences are within noise (and GPQA at n=546 → k=54 is pure noise).

### 6.4 What is salvageable — and genuinely promising
Ranked by value:
1. **Residualized student entropy (the "Wiener idea" done right).** The legitimate core intuition is *denoising*: remove the intrinsic-difficulty component from student uncertainty. The statistically sound version: regress `H_s` on `H_p` (isotonic or quantile regression across the pool) and select by the **residual** `H_s − E[H_s|H_p]`. This is scale-free (fixes the raw-nats vs normalized inconsistency and cross-vocab comparability), needs no clamping, has a shrinkage/BLUE interpretation, and is a strict generalization of entropy gain. One afternoon offline + one training run. This is the most promising SP-derived direction.
2. **Expected-utility factorization instead of formula zoo.** HV-optimal selection is by `P(student wrong | H_s) × P(teacher correct | H_t)` — estimate both calibration curves on a held-out slice and multiply. Every hand-crafted metric (EG, LI, Window-*) is an ad-hoc approximation of this product; the calibrated version is principled, interpretable, and extends naturally with a third factor for trainability (p(1−p) in probability space). Cheap, offline-testable.
3. **Learned 2D skyline:** fit a tiny logistic model on (Ĥs, Ĥt) → HV on held-out data to upper-bound what *any* hand-crafted 2D formula can achieve. If the skyline ≈ H(s), stop engineering metrics; if it's well above, the gap tells you what to design.
4. **The teacher-reliability phase diagram (original contribution):** systematically vary label-source accuracy (weak proxies → true teachers, or synthetic label corruption) and map which metric family wins where. Turns the notebooks' contradiction into a figure and answers a question practitioners actually have ("my teacher is imperfect — should I still chase student uncertainty?").
5. **Probability-space window p(1−p)** as the trainability term (see 6.3.1) — test as a selection score alongside the grid.

### 6.5 Gaps this analysis exposes (additions to §5)
- **The deployed metric contradicts the lab's own cleaner protocol.** Under true-teacher labels the design lab ranks H(s) ≥ EG on HV — yet `distillation_by_metrics` deployed EG and never ran H(s). Either EG's noise-avoidance advantage is real downstream (then say so and show it), or the production bet was placed on the wrong metric. The P0 grid adjudicates this; the lab evidence currently *predicts student entropy may win*, raising the stakes of risk #2 in §5.
- **The surrogate is unvalidated.** No artifact links HV@10% to downstream trained accuracy. With even 4–6 training runs from the grid, plot HV-of-selected-set vs final accuracy per metric — if the correlation is weak, the entire offline lab needs a different surrogate (e.g., one-epoch loss decrease on selected items).
- **Notebook 1's protocol should be retired or relabeled** (circular labels), and its GPQA slice (n=546) dropped from any averaged claim.
- **No CIs anywhere in the lab tables;** add binomial/bootstrap intervals before using HV rankings to make decisions.
- **Code hygiene:** ~500 duplicated lines between the two notebooks with divergent configs (DATA_DIR, metric sets) — one drifting copy already produced the normalized-vs-raw EG mismatch with the trainer; consolidate into `src/core` before the grid.

### 6.6 Paper framing consequence
Don't present the SP metric zoo as a contribution — present the **diagnosis** as one: "hand-crafted uncertainty-fusion metrics are unstable under label-source changes; a calibrated expected-utility score (or residualized entropy) is stable." That converts an embarrassing internal contradiction into the paper's most defensible analytical section, and it differentiates you from LARK/RHO-LOSS-style single-formula methods.

---

## 7. Entropy measurement and normalization — the proper way

*Current state: notebooks divide by the per-pool max (`H/(H.max()+eps)`); the trainer uses raw nats. Both are wrong in different ways.*

### 7.1 Why the current schemes fail
- **Per-pool max normalization** is set by a single outlier, is dataset-relative, and — fatally for the online loop — is recomputed each epoch on a drifting student, so the scale (and the `drop_non_positive` threshold riding on it) wobbles epoch to epoch.
- **Raw nats across models** are not comparable: Llama (128k vocab), Qwen (152k), Phi-4-mini (200k) have different supports, tokenizations of the option letters (`a` vs ` a` vs `A`), and formatting habits, all of which move entropy for reasons unrelated to question difficulty. Entropy gain *subtracts* two such numbers.

### 7.2 Why "divide by log |V|" is not the fix
`H/log|V|` (info-theoretic "efficiency") corrects the theoretical maximum, but log is slow: ln|V| = 11.76 / 11.93 / 12.21 for 128k/152k/200k — a ≤4% scale correction. Measured single-token entropies live at 0–2.5 nats and their cross-model differences are driven by **tail mass** scattered over the ~10⁵ non-answer tokens, which log|V| normalization leaves untouched. It is cosmetic here.

### 7.3 The recommended measurement stack
1. **Alphabet-restricted entropy (the base fix).** Sum first-token probability over all tokenizations of each valid option (a–j incl. leading-space/case variants), renormalize over the K options, compute the K-way entropy, divide by log K. Properties: bounded in [0,1]; semantically identical support across all tokenizers (student−proxy subtraction becomes meaningful); absolutely stable across epochs (no pool-relative rescaling); and the discarded mass P(valid answer token) becomes a free **format-compliance signal** — the quantity that currently degrades silently into NaN backfills as the student turns CoT-native. Caveat: verify option tokenization (numeric options collide, e.g. "1" vs "10" share a first token; letters a–j are safe).
2. **Temperature calibration (optional rigor layer).** Entropies of two models are only comparable if similarly calibrated; fit one temperature per model on a held-out slice (align confidence with accuracy), then compute the K-way entropy. Do this if calibrated vs uncalibrated EG rankings disagree; otherwise appendix. In depth in §7.5.
3. **Quantile normalization for the gain metric** — in depth below (§7.4).
4. **Residualization on the quantile scale (the unifying recipe).** Quantile-transform both entropies, then regress q_s on q_p (isotonic) and select by the residual. Strictly generalizes quantile-EG (which implicitly assumes the identity map q_p ↦ q_s is the right baseline) and absorbs asymmetric copula shape.

### 7.4 Quantile normalization, in depth

**Definition.** For model m with entropies {H_1..H_N} over the pool, replace each value by its empirical CDF position: `q_i = F̂_m(H_i)`. Implementation with mid-rank tie handling:
```python
from scipy.stats import rankdata
def ecdf_quantiles(h):                      # h: entropies over the pool
    return (rankdata(h, method="average") - 0.5) / len(h)   # in (0,1)
q_s = ecdf_quantiles(H_s)   # student: re-fit each epoch on the current pool
q_p = ecdf_quantiles(H_p)   # proxy: computed once, same question set
score = q_s - q_p           # quantile entropy-gain
```
(For out-of-sample items — eval sets, resumed epochs — map through the fitted ECDF by interpolating against the sorted training values, not by re-ranking a mixed pool. Never pool train and test when fitting.)

**What it buys.**
- *Invariance:* any strictly monotone transform of the raw score (units, log, max-scaling) leaves quantiles unchanged. All model-specific scale/offset/shape differences — vocab size, tail mass, sharpness — vanish. `q_s − q_p` asks the right question: "is this item unusually uncertain **for the student**, relative to how unusual it is **for the proxy**."
- *Copula view:* the pair (q_s, q_p) is exactly the copula of (H_s, H_p) — the dependence structure with both margins stripped. Quantile-EG selects the upper-left region of the copula plot. This is the clean formalization of what the Window/SNR metrics were groping toward.
- *Bounded, interpretable scores:* q_s − q_p ∈ (−1, 1); top-k thresholds are percentile statements that transfer across datasets and models.

**The key design decision in the online loop — re-fit vs fixed reference.** Re-fitting the student ECDF each epoch makes q_s *relative to the current epoch*: the top decile is always the top decile even as absolute entropies collapse. Consequences: (a) selection always has full dynamic range — no LR-schedule/`__len__` mismatch from a shrinking pool; (b) but the convergence signal (`selected_samples` shrinking as the student catches the proxy) disappears — with symmetric ranks, roughly half the items have q_s − q_p > 0 forever, so `drop_non_positive` no longer terminates. Recommendation: **re-fit per epoch for selection; monitor convergence separately with the absolute alphabet-normalized entropy** (mean, and P(valid-answer) mass). If instead you map every epoch through the *epoch-0* ECDF, you keep an absolute progress meter but reintroduce pool-shrinkage; that variant is better as a logged diagnostic than as the selector.

**Pitfalls to engineer around.**
1. *Ties and atoms:* single-token entropies pile up near 0 (easy items). Use mid-ranks (`method="average"`) so an atom maps to one quantile instead of arbitrary ordering; expect compressed resolution inside the atom.
2. *Loss of absolute meaning:* quantiles always nominate a "top 10%", even when the student has nothing left to learn. Pair the selector with an absolute floor (e.g., alphabet-entropy > ε) so the method can stop.
3. *Density distortion:* rank differences weight raw differences by local density — in crowded regions of the distribution, differences at the measurement-noise level become large rank jumps (churn in the selected set). If churn is high (check the §2 Jaccard diagnostic), smooth by averaging entropy over prompt paraphrases or option permutations before ranking.
4. *Dynamic range is governed by rank correlation:* for uniform margins, Var(q_s − q_p) = (1 − ρ_S)/6 where ρ_S is the Spearman correlation between student and proxy entropy. At ρ_S = 0.9 the score's SD is ≈0.13 on a (−1,1) range — selection then operates on small differences that may be noise-dominated. **Measure ρ_S first (Week-1 task):** it quantifies, in one number, how much information the proxy adds over plain student entropy — the crisp version of risk #2 in §5.
5. *Tail noise:* ECDF estimates at extreme quantiles carry the most sampling noise; at N≈9.6k the top-1% boundary is fuzzy — irrelevant at k=10%, relevant if you shrink the budget.
6. *Alignment:* rank within the question_id intersection of the two models' tables; if per-category coverage skew matters (§5.6), rank within category (stratified quantiles) so top-k cannot be monopolized by one subject's entropy distribution.

### 7.5 Temperature calibration, in depth

**What it is.** Temperature scaling (Guo et al. 2017, arXiv:1706.04599): divide the logits by a single scalar T before the softmax, `p(T) = softmax(z/T)`, with T fit on held-out labeled data to minimize NLL of the gold answers. T > 1 flattens an overconfident model; T < 1 sharpens an underconfident one. It has one parameter, is fit in seconds, and **never changes the argmax or the within-item token ranking** — accuracy is untouched; only the confidence profile moves.

**Why entropy gain needs it.** Entropy is a function of the probability vector's *shape*, and modern instruction-tuned LLMs are systematically miscalibrated — usually overconfident, and by amounts that differ across families (Phi vs Qwen vs Llama) and across training stages. If the student is more overconfident than the proxy, raw H_s is deflated relative to H_p, EG = H_s − H_p is biased downward, and `drop_non_positive` silently discards genuinely learnable items; an underconfident student inflates EG everywhere. After calibrating both models, "confidence ≈ probability of being correct" holds on both sides, so the subtraction compares *predictive uncertainty about the same event* rather than two sharpness idiosyncrasies. This is also what makes the §1.1 theory story well-founded: **calibrated** EG approximates the gap in expected correctness — the headroom a teacher trace can actually close. Uncalibrated EG approximates nothing in particular.

**Why quantile normalization does not subsume it.** Quantiles are invariant to monotone transforms of the entropy *value across items* — but temperature is not such a transform: it re-shapes each item's distribution individually and can **reorder items by entropy**. Example: item A = {0.9, 0.1} over two live options has H = 0.33 nats; item B = {0.97, 6×0.005} over seven has H = 0.19 — A ranks as more uncertain. Flatten both (large T): A → ln 2 = 0.69, B → ln 7 = 1.95 — the order inverts, because B's uncertainty is *wide but sharp* and A's is *narrow but soft*. Calibration decides which regime you are in before ranks are taken; quantiles then strip the remaining marginal shape. The layers are complementary: calibrate per item, quantile across items.

**Recipe for this codebase.**
1. Operate on the **alphabet-restricted K-way distribution** (§7.3.1) — calibrate the decision that matters, not the 150k-token tail.
2. Fit on the **validation split only** (never test, and not the selection pool itself — mild circularity otherwise). A 1-parameter fit is low-variance: ~500–1,000 labeled items suffice; your 10% val split (~1.2k) is enough.
3. Solve `T* = argmin_T Σ_i −log softmax(z_i/T)[y_i]` — a smooth 1-D problem; `scipy.optimize.minimize_scalar` over log T. Gold label = correct option.
4. Recompute the K-way probabilities at z/T*, take entropy, divide by log K. Then apply the quantile/residual layer (§7.4).
5. **Proxy: fit once. Student: re-fit every epoch.** Fine-tuning is known to *worsen* calibration (SFT on hard targets sharpens the answer distribution), so T_s drifts during training. The re-fit is one 1-token forward pass over the val slice — piggyback it on the existing per-epoch estimation subprocess; cost is negligible. The T_s trajectory itself is a diagnostic worth logging: strong drift means raw per-epoch entropies of the *same* student aren't comparable across epochs either, contaminating any fixed-reference analysis and the raw-nat `drop_non_positive` semantics.

**Decision rule — is the layer needed at all?** Compute per-model reliability diagrams/ECE on the K-way distributions, and the Jaccard overlap between the EG top-k selected with and without calibration. If the overlap stays ≳0.9 across epochs, calibration doesn't change selection → one appendix paragraph and move on. If it drops, the layer is load-bearing and belongs in the method. Either result is reportable.

**Limits and pitfalls.**
- A single T cannot fix *difficulty-dependent* miscalibration (models are typically most overconfident exactly on hard items). The residual miscalibration is one more reason to keep the quantile/residual layer on top rather than trusting calibrated absolute values alone. (Vector/matrix scaling or per-category temperatures exist, but with ~1.2k val items and K=10 they overfit; don't.)
- T fit on val is only valid under the same distribution — fine here (random split of one dataset), but re-fit if you move to a new dataset or a mid-training student (hence step 5).
- Items with near-zero mass on the valid alphabet can't be meaningfully calibrated — exclude them from the T fit and let the P(valid) signal handle them.

**Order of operations (final):** restrict to alphabet → temperature-calibrate (if the decision rule says it matters) → K-way entropy ÷ log K → per-epoch quantile transform → select by quantile difference or isotonic residual, with an absolute-entropy floor for stopping.

**Bottom line:** alphabet-restricted, log-K-normalized entropy as the measurement; per-epoch quantile transform (mid-ranks) for both models; select by q_s − q_p or, better, by the isotonic residual of q_s on q_p; keep an absolute-entropy floor and log P(valid answer) + Spearman ρ_S as diagnostics. Temperature calibration is the layer that makes *absolute* EG values meaningful (theory claims, stopping rules, cross-epoch monitoring); the ECE + top-k-Jaccard decision rule determines whether it graduates from appendix to method.

---

## Suggested sequencing (10–11 weeks, part-time, 2×H100)

1. **Week 1 (no GPU):** offline sanity — score-correlation/Jaccard analysis (risk #2), per-category coverage, cumulative-unique-selected accounting from existing artifacts; fix sampler hygiene (normalization, LR/len bug, duplicate class); seed everything. From §6: compute residualized-entropy and calibrated expected-utility scores offline, add CIs to the HV tables, and re-run the lab with the circular protocol retired — this decides which 1–2 extra scorers join the P0 grid.
2. **Weeks 2–4:** P0 grid (6 strategies × 3 seeds × 20 epochs, Qwen/direct traces) + static-vs-adaptive + RHO-LOSS row. This decides the paper's thesis.
3. **Weeks 4–6:** proxy-size ablation (offline + 3–4 runs); harmful-data demonstration; answer-hidden regeneration of corrected traces (API work, no GPU).
4. **Weeks 6–9:** winning method on GSM8K + second student (Phi-4-mini); OOD evals; 50-epoch extended runs for the headline configs only.
5. **Weeks 9–11:** writing, figures (compute-frontier main figure, entropy-plane cartography, selection-churn), significance tests, related-work table vs RHO-LOSS/LARK/ELAD.

**Decision gate after the grid (end of week 4):** if EG > student-entropy > random with significance → current framing holds. If EG ≈ student-entropy > random → pivot headline to the cheap self-referenced signal + recursion (proxy becomes an ablation). If nothing beats random-resampled → pivot the paper to "data rotation, not data choice, drives efficient distillation" (still publishable, honest, and novel against the selection literature).
