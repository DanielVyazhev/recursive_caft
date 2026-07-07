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
- **Unify normalization**: pick one (recommend rank/quantile normalization per model — robust to outliers and to vocab-size differences between student and proxy; raw-nat differences across different tokenizers/vocabs are not comparable) and use it in both analysis and training.
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

## Suggested sequencing (10–11 weeks, part-time, 2×H100)

1. **Week 1 (no GPU):** offline sanity — score-correlation/Jaccard analysis (risk #2), per-category coverage, cumulative-unique-selected accounting from existing artifacts; fix sampler hygiene (normalization, LR/len bug, duplicate class); seed everything.
2. **Weeks 2–4:** P0 grid (6 strategies × 3 seeds × 20 epochs, Qwen/direct traces) + static-vs-adaptive + RHO-LOSS row. This decides the paper's thesis.
3. **Weeks 4–6:** proxy-size ablation (offline + 3–4 runs); harmful-data demonstration; answer-hidden regeneration of corrected traces (API work, no GPU).
4. **Weeks 6–9:** winning method on GSM8K + second student (Phi-4-mini); OOD evals; 50-epoch extended runs for the headline configs only.
5. **Weeks 9–11:** writing, figures (compute-frontier main figure, entropy-plane cartography, selection-churn), significance tests, related-work table vs RHO-LOSS/LARK/ELAD.

**Decision gate after the grid (end of week 4):** if EG > student-entropy > random with significance → current framing holds. If EG ≈ student-entropy > random → pivot headline to the cheap self-referenced signal + recursion (proxy becomes an ablation). If nothing beats random-resampled → pivot the paper to "data rotation, not data choice, drives efficient distillation" (still publishable, honest, and novel against the selection literature).
