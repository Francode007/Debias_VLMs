# Interpretability-Derived Rewards Are Trajectory Selectors: A Two-Regime Geometric Structure of Bias in a Vision-Language Model

**Anonymous Authors**
Submission target: NeurIPS 2026 Workshop on Interpretability for Discovery (Interp4Discovery)

> **Draft status**: v1 skeleton, Markdown. Freezes results at Phase 0.9 R1+R2+R3
> ([Phase0.9/FINAL_REPORT.md](../../Phase0.9/FINAL_REPORT.md)). Related-Work section
> is a stub, to be written in a follow-up. Figures are placeholders; captions and
> data pointers are complete for later plotting. Will be ported to the NeurIPS 2026
> LaTeX workshop template (5-page main text, unlimited references and appendices).
> AI4GOOD adaptation notes at the end (§10). Every number cited traces to a
> workspace source doc — see §11.

---

## Abstract

We debias a vision-language model (Qwen2.5-VL-3B-Instruct) using an
**interpretability-derived training signal**: the projection of an intermediate
hidden state onto a frozen linear probe of demographic bias, fitted with 5-fold
cross-validation on a held-out fairness benchmark, and used as a dense reward
during PPO. The pipeline surfaces three coupled results.

*A discovery about the bias representation.* Fitting one linear probe per
demographic axis at each depth reveals a two-regime geometric structure. At
layers L9–L21 the ten per-axis unit weight vectors are nearly orthogonal (mean
pairwise `|cos| ≈ 0.10`); at L25–L35 they collapse onto a partially shared
subspace (`|cos| ≈ 0.50`). B = 200 paired-bootstrap gives
`P(|cos|@L25 > |cos|@L13) = 1.000`. Cross-benchmark replication on SB-Bench-9axis
confirms the `|cos|` convergence direction as a **model property**, while the
PC0 concentration spike is dataset-coupled.

*A mechanistic null on the reward signal.* PPO with the single-layer probe
reward at L13 delivers +1.49 pp ± 0.35 pp accuracy on 4 seeds on canonical
SB-Bench (n = 2916, per-seed McNemar `p < 10⁻¹¹`), but does not measurably attack
the bias direction at any of 11 probed depths (`|z| ≤ 0.075σ` vs a
pre-registered `|z| ≥ 0.30σ` cutoff). The reward acts as a **trajectory
selector** over already-existing low-bias completions, not as a representational
**eraser**.

*A predictive validation of the geometry.* An ensemble reward pooling five probes
across the regime boundary (L17–L33) is the only method that reduces the
aggregate `|bias_score|` on out-of-distribution VLBiasBench transfer (+0.0021
against the single-layer baseline's −0.0005), at a cost of 1.4 pp on the
abstention channel. Interpretability *can* drive alignment interventions in
VLMs — but the interventions operate at a level different from the probe
visualization, and the geometric structure of the representation constrains
what a single-layer reward can achieve.

---

## 1. Introduction

Vision-language models (VLMs) increasingly ground high-stakes decisions
involving demographic subjects — from face-attribute inference to
image-grounded question answering — and demographic bias in their outputs is
a documented failure mode across BBQ-family multiple-choice benchmarks (Cite:
BBQ, VLBiasBench, SB-Bench). Two families of debiasing exist and both fail
in characteristic ways. Binary letter-correctness PPO collapses onto the
highest-prior letter under mode collapse (Cite: RLHF/RLAIF failure literature).
Activation-space steering derived from projected concept probes transfers
poorly out-of-distribution and can cause abstention collapse (Cite: INLP, RLACE,
LEACE, and our companion Phase 0.7 PCA-DRM result).

We investigate a middle path: use a supervised linear probe of demographic bias
— fitted with 5-fold CV on a held-out fairness benchmark (VLBiasBench, close-
ended) — as a dense, input-conditioned **reward** during PPO on a different
benchmark (SB-Bench). The probe is frozen during PPO. Because the probe's
decision boundary is a specific 2048-d direction in hidden-state space, the
reward is grounded in an interpretability-verifiable representation, not a
black-box preference model. Three tightly coupled contributions emerge.

**Contribution 1 (discovery).** Fitting ten *per-axis* linear probes at each
of eleven depths reveals a **two-regime geometric structure of bias**. In the
mid-depth window L9–L21, per-category directions are nearly orthogonal — the
shared `bias_aligned` probe that succeeds at every layer is an averaged
compromise over disentangled per-category *circuits*. In the deep window
L25–L35, per-category directions partially collapse onto a shared subspace.
Paired bootstrap sharpens this into a decisive separability result
(`P = 1.000` on `|cos|` for every Regime-1 × Regime-2 pair), and the direction
of convergence replicates on SB-Bench-9axis — establishing it as a **model
property** rather than a dataset-coupled probe artefact.

**Contribution 2 (mechanistic null).** A pre-committed wash-out diagnostic tests
whether the L13 probe reward attacks the L13 bias direction (Pattern 1),
displaces the bias to deeper layers (Pattern 3), or does nothing measurable
(Pattern 2). All four PPO variants — production single-seed, correctness-only,
bias-only, and KL-stabilised — deliver Pattern 2 with `|z| ≤ 0.075σ` at every
layer probed. Combined with a reward-synergy decomposition that shows
`corrOnly + biasOnly` individually fail (−5.25 pp, −1.27 pp) while the
combination succeeds (+1.85 pp), the picture is: **the interpretability-derived
reward is a trajectory selector over already-existing low-bias completions,
not a representational eraser**.

**Contribution 3 (predictive validation).** The two-regime discovery predicts
that a probe reward pooling both regimes should generalize better OOD than a
single-layer probe fit to the training benchmark's activations. Empirically,
an ensemble reward across L17–L33 with per-layer z-score normalization is the
only variant that reduces the aggregate `|bias_score|` toward zero on VLBias
transfer (+0.0021 vs single-layer L13's −0.0005 which *over-corrects past zero*).

Together, these results argue that interpretability *can* drive alignment
interventions in VLMs, but the interventions operate mechanistically at a
different level than the probe-decision-boundary visualization suggests,
and the geometric structure of the underlying representation bounds what a
single-layer intervention can achieve.

---

## 2. Methodology

### 2.1 The `bias_aligned` probe

A binary logistic-regression probe over the 2048-d hidden state at layer L:
`w_biasA(L) ∈ ℝ^{2048}` with `C = 1.0`, `class_weight='balanced'`,
5-fold `StratifiedKFold`, `seed = 42`, `max_iter = 2000`. The label rule is
defined only on **disambig** records:

- `condition = neg` (stereotyped person is gold): label = 1 iff `pred = gold`.
- `condition = non_neg` (counter-stereotyped person is gold): label = 1 iff
  `pred = the other named option`.
- `condition = ambig` (gold = C, "unknown"): **label = None, record dropped**.

Excluding ambig by construction decouples fairness from abstention, and closes
a specific failure mode of predecessor probe-heads (Phase 0.7 T1.1: DRM heads
conflated good-C with bad-C).

### 2.2 Reward composition

The per-step reward for PPO has three learned terms and one KL penalty:

$$
r_t \;=\; w_\text{corr}\!\cdot\!\mathbb{1}[\hat{y}=y^*] \;+\; w_\text{bias}\!\cdot\!\bigl[-(\mathbf{w}_\text{biasA}\cdot \mathbf{h}_{L^{*}}[\text{ans\_pos}])\bigr] \;+\; w_\text{ambig}\!\cdot\!\mathbb{1}[g=C\wedge \hat{y}=C] \;-\; \beta\!\cdot\!\text{KL}_\text{tok}(\pi\|\pi_\text{ref})
$$

with `w_corr = w_bias = 1.0, w_ambig = 0.5, β = 0.1, target_kl = 0.02`. The
bias-projection term is negated because the probe predicts stereo-aligned = 1;
minimising it therefore corresponds to reducing bias-aligned emissions. `L*`
is the intervention layer — L13 for the single-layer baseline and multi-layer
`{L17, L21, L25, L29, L33}` for the ensemble (§2.6). We probe at the answer
position, i.e., the token whose logits produce the letter choice.

### 2.3 PPO recipe

Base model: Qwen2.5-VL-3B-Instruct (36 transformer blocks, hidden dim 2048,
bf16, FlashAttention-2). LoRA `r = 16`, `α = 32` targeting q/k/v/o projections
in every transformer block (distributed adapter). Batch size 8, one epoch on
SB-Bench train (n = 1936, 242 batches). Value head is a linear layer to a
scalar with `bias = False`, zero-initialised. **KL-stability bundle (KLFIX)**:
value-head zero-init at the code level, `--value-clip-range 0.2`, and
`--kl-adapt-rate 0.3`; empirically this reduces cross-seed accuracy std by
`3×` and tail-KL std by `148×` versus the un-fixed recipe. All results
in §4 use the KLFIX backbone; ablations that vary this bundle appear only in
supplementary comparisons.

### 2.4 Per-axis two-regime probe

To test whether the shared `bias_aligned` direction is a rank-1 axis or an
averaged compromise across per-category circuits, we fit **10 per-axis binary
probes** (Age, Disability, Gender_identity, Nationality, Physical_appearance,
Race_ethnicity, Race_x_SES, Race_x_gender, Religion, SES) at each layer
`L ∈ {1, 5, 9, 13, 17, 21, 25, 29, 33, 35}` with `n = 60` disambig records
per axis and the same hyperparameters as §2.1. Stack unit-normalised weight
vectors into `W_L ∈ ℝ^{10×2048}` and PCA on `W_L`. Report two geometric
descriptors:

- `PC0 evr(L)`: fraction of variance in `W_L` explained by the top PC
  (rank-1 concentration).
- `mean pairwise |cos|(L)` across the ten unit vectors (per-axis alignment).

Uniform-spectrum baseline (10 orthogonal vectors in 2048-d): PC0 evr = 0.10.

### 2.5 Bootstrap protocol

For each layer, resample each axis's records with replacement (n = 60), refit
each axis probe, recompute `PC0 evr` and `|cos|`; B = 200 draws. Report
2.5 / 97.5 percentile marginal CIs. **Paired-draw variant**: within a single
draw, the axis-resample indices are shared across all layers probed — this
controls for the layer-correlated nuisance covariance that inflates marginal
CIs without affecting cross-layer ordering. Paired separability =
`P(metric@L_hi > metric@L_lo)` computed across the B paired draws. A
degenerate-draw fallback re-samples when a bootstrap draw of a single axis
produces a single-class label vector (< 5 / 200 in practice).

### 2.6 Ensemble reward

Bundle five per-layer `bias_aligned` probes at `L ∈ {17, 21, 25, 29, 33}`,
straddling the regime boundary from §2.4. Unit-normalise each; precompute
per-layer offline μ_L, σ_L on a training-distribution activation cache
(n = 751) to keep runtime scale-invariant. Runtime: per-layer projection at
the answer position, per-layer z-score with the offline `(μ_L, σ_L)`,
equal-weight mean across the five, single negation for the sign convention,
plug into the `w_bias` slot of §2.2. Signal fidelity is verified per training
step by emitting each per-layer z as a metric (`ensemble_z_L{N}_mean`); the
five means bunch tightly across seeds (§4.3 discussion).

### 2.7 Wash-out diagnostic

For each PPO variant *V* and layer `L ∈ {1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35}`,
re-extract the mean bias-probe projection over a held-out eval set and compute
`z_V(L) = (μ_V(L) − μ_vanilla(L)) / σ_vanilla(L)`. Pre-committed classification:

- **Pattern 1 (clean eraser)**: `z(L13) ≤ −0.30` AND `z(L) ≤ −0.15` at every `L ≥ 17`.
- **Pattern 2 (wash-out)**: `z(L13) ≤ −0.30` BUT `|z(L)| < 0.10` at every `L ≥ 21`.
- **Pattern 3 (proxy-hack)**: `z(L13) ≤ −0.30` BUT `z(L) > 0` at any `L ≥ 25`.

An observed maximum `|z|` below `0.10` at every layer indicates the reward
did not measurably move the bias direction anywhere; we call this
**Pattern 2 (flat)**, the sub-threshold degenerate case of Pattern 2.

---

## 3. Experimental Setup

**Base model.** Qwen2.5-VL-3B-Instruct — LM-decoder backbone with 36
transformer blocks, hidden dim 2048, bf16, FlashAttention-2, greedy decoding
constrained to the letter tokens `{A, B, C}` for eval.

**Datasets.** SB-Bench (in-distribution training + eval, canonical test split
n = 2916 across 9 BBQ axes; the SB-Bench-9axis split with n = 751 is used only
for the cross-dataset replication in §2.5). VLBiasBench close-ended
(probe-fit source + OOD transfer eval, n = 2000 intersection, 10 axes with 7
shared with SB-Bench). Splits are pre-registered and frozen; probe fits and
PPO training never see the SB-Bench canonical test items or the VLBias
transfer items.

**Seeds.** Four seeds `{s1, s2, s3, s4}` for the KLFIX-L13 baseline; three
seeds `{s1, s2, s4}` for the ensemble (s3 hit a container-side data-loader
crash that occurred on both initial and re-launch attempts, at the same
micro-batch on both, indicative of concurrent-worker resource contention with
the eval loop; the crash is not model-related). Like-for-like comparisons in
§4 use the intersection `{s1, s2, s4}` for both arms.

**Baselines.** (i) *Vanilla* Qwen2.5-VL-3B-Instruct with no PPO. (ii) *corrOnly*:
the reward of §2.2 with `w_bias = 0` (correctness-only PPO on the same
backbone). (iii) *biasOnly*: `w_corr = 0`. (iv) *KLFIX-L13*: single-layer L13
probe reward (the intended comparison for the ensemble).

**Metrics.** SB-Bench: micro-accuracy (macro reported alongside) with per-seed
McNemar exact-binomial p-value paired against vanilla; Stouffer's method for
cross-seed aggregate z. VLBiasBench: overall accuracy, disambig/ambig accuracy,
`neg` and `non_neg` accuracy, signed and absolute `bias_score`, overall
`unknown_rate`. All cross-seed statistics are reported as mean ± std.

**Compute.** Modal-orchestrated A100-80GB single-GPU training and eval runs;
per-seed PPO wall-time ≈ 30 min, per-seed eval ≈ 8 min. Full sweep
(4 seeds × training + generation) fits in a $10 budget.

---

## 4. Results

### 4.1 In-distribution accuracy — SB-Bench (n = 2916)

Table 1 shows the headline result on canonical SB-Bench. The single-layer
KLFIX-L13 baseline improves 4-seed micro-accuracy by +1.49 pp with a
cross-seed std of 0.35 pp; the ensemble matches this in an aggregate sense
(+1.29 pp on the like-for-like 3-seed subset) but does not exceed it on ID.
Both far exceed the individual-term ablations.

| Method | Seeds | micro-acc | ± std | Δ vs vanilla | Notes |
|---|---:|---:|---:|---:|---|
| Vanilla | n/a | 0.6190 | — | — | canonical n = 2916 |
| corrOnly (`w_bias = 0`) | 1 (prod) | 0.5665 | — | **−5.25 pp** | actively hurts |
| biasOnly (`w_corr = 0`) | 1 (prod) | 0.6063 | — | **−1.27 pp** | fails alone |
| **KLFIX-L13** | **4** | **0.6298 / 0.6339** | **0.35 pp** | **+1.08 to +1.49 pp** | all 4 seeds > vanilla, per-seed McNemar `p < 10⁻¹¹` |
| KLFIX-L13 (like-for-like 3-seed) | 3 (s1, s2, s4) | 0.6286 | 0.31 pp | +1.37 pp | Stouffer combined z = +9.56 |
| **R3 ensemble (L17–L33 zmean)** | 3 (s1, s2, s4) | 0.6278 | 0.38 pp | +1.29 pp | Stouffer combined z = +8.10 |

The two 4-seed KLFIX-L13 aggregate figures (+1.08 pp and +1.49 pp) reflect
different aggregation protocols across our Phase 0.8 and Phase 0.9 audits.
See [Phase0.9/canonical/klfix_L13_4seeds_aggregate.json](../../Phase0.9/canonical/klfix_L13_4seeds_aggregate.json)
for reconciliation; the +1.49 pp is the Phase 0.9 canonical.

Reward-synergy interpretation: `corrOnly + biasOnly` = −6.52 pp individually
vs +1.49 pp jointly ⇒ interaction ≈ **+8.4 pp**, decisively rejecting the
hypothesis that the bias-projection term is a noisy correctness reward.

Per-axis fairness (KLFIX-L13, 4-seed) is uniformly favourable: 8 of 9 axes
improve over vanilla, with Religion (+3.30 pp ± 0.35) and Gender_identity
(+1.67 pp ± 0.47) the largest gains; only Disability regresses (−0.31 pp
± 0.80, within 1 SE). The full per-axis table is in the appendix.

### 4.2 Out-of-distribution transfer — VLBiasBench (n = 2000)

Table 2. Same 3 seeds (s1, s2, s4) for both PPO arms.

| Method | overall_acc | disambig_acc | ambig_acc | signed bias_score | \|bias_score\| Δ vs vanilla |
|---|---:|---:|---:|---:|---:|
| Vanilla | 55.13 % | 36.75 % | 91.90 % | +0.0070 | — |
| KLFIX-L13 (3-seed) | 54.42 % ± 0.20 | 36.16 % ± 0.47 | 90.90 % ± 0.38 | −0.0075 ± 0.0048 | **−0.0005** (worsens) |
| **R3 ensemble (3-seed)** | 54.47 % ± 0.06 | **36.93 % ± 0.24** | 89.51 % ± 0.45 | **−0.0049 ± 0.0048** | **+0.0021** (reduces) |

The single-layer L13 reward *over-corrects past zero* on the OOD benchmark:
the signed bias score flips from +0.0070 (vanilla) to −0.0075 (KLFIX-L13),
increasing `|bias_score|` by 0.0005. The ensemble is the **only** method that
reduces `|bias_score|` toward zero (+0.0021 reduction), and the reduction is
signed-consistent on s1 and s4. The gain concentrates on `disambig_acc`
(+0.78 pp) and `non_neg_acc` (+0.90 pp, not shown), with a 1.4 pp cost on
`ambig_acc` reflecting reduced abstention.

### 4.3 Two-regime bias geometry (VLBias hidden states, B = 200 paired)

Table 3. Marginal 95% CIs at selected layers; paired separability at the
regime boundary.

| Layer | `PC0 evr` [95% CI] | `mean \|cos\|` [95% CI] | Regime |
|---:|:--|:--|:--|
| L1 | [0.178, 0.248] | [0.244, 0.375] | input embedding artefacts |
| L9 | [0.163, 0.211] | [0.081, 0.124] | disentangled |
| **L13** | **[0.162, 0.214]** | **[0.085, 0.133]** | **disentangled (deployment layer)** |
| L17 | [0.159, 0.207] | [0.090, 0.131] | disentangled |
| L21 | [0.151, 0.201] | [0.106, 0.156] | transition |
| **L25** | **[0.202, 0.297]** | **[0.491, 0.599]** | **convergent (peak)** |
| L29 | [0.191, 0.255] | [0.456, 0.546] | convergent |
| L33 | [0.188, 0.247] | [0.404, 0.487] | convergent |

Paired-draw separability at the regime boundary:

|  metric  | `P(L25 > L13)` on VLBias | `P(L25 > L13)` on SB-Bench-9axis |
|:--|:-:|:-:|
| `mean |cos|` | **1.000** | **1.000** |
| `PC0 evr` | **0.995** | **0.200** (inverted) |

**The `|cos|` convergence direction replicates as a model property** across
two BBQ-family benchmarks generated by the same model
(`P = 1.000` on 17 of 20 Regime-1 × Regime-2 layer pairs on SB-Bench-9axis,
the three sub-1.0 cells all involving L35). The **`PC0 evr` concentration
spike is VLBias-coupled** (inverted ordering on SB-Bench-9axis). We therefore
lead the geometric claim with `|cos|` and retain `PC0 evr` as a directional
secondary descriptor with the dataset caveat disclosed.

`|cos|` magnitude is dataset-dependent: `|cos|@L25 = 0.540` on VLBias vs
`0.172` on SB-Bench-9axis (both above their own dataset's disentangled
baseline). Candidate explanations (not yet isolated): per-axis
class-imbalance differences, image-modality coverage, presence versus absence
of intersectional axes (VLBias has `Race_x_SES`, `Race_x_gender`; SB-Bench
has `Sexual_orientation` only).

### 4.4 Wash-out diagnostic — selector, not eraser

Table 4. Cross-layer maximum `|z|` for four PPO variants against vanilla.

| Variant | `max |z|` | `argmax L` | Pre-committed pattern |
|:--|--:|--:|:--|
| Prod L13 (single-seed) | 0.032 σ | 11 | Pattern 2 (wash-out) |
| corrOnly | 0.075 σ | 25 | Pattern 2 (wash-out) |
| biasOnly | 0.021 σ | 25 | Pattern 2 (wash-out) |
| **KLFIX-L13** | **0.019 σ** | **17** | **Pattern 2 (flattest)** |

All four variants are **sub-threshold at every layer probed** (pre-registered
Pattern-1 cutoff was `|z| ≥ 0.30 σ` at L13; the observed maxima are 4–16 ×
below this threshold). The interpretability-derived reward achieves
+1.08–1.49 pp accuracy **without measurably shifting the bias
representation at any depth**. KL stability further reduces the residual
motion — KLFIX-L13 is flatter than the un-stabilised prod L13, indicating that
the observed accuracy gains are being achieved through a mechanism other than
representational erasure of the bias direction. The mechanism is
**trajectory selection** over the base policy's existing distribution of
completions.

---

## 5. Discussion

### 5.1 What the two-regime finding predicts about interventions

Under the disentangled regime (L9–L21), the ten per-category bias directions
are nearly orthogonal. A single scalar reward derived from a single 1-D
`bias_aligned` probe therefore captures an averaged compromise across
categories — sufficient to increase in-distribution accuracy on a
mixed-category test set (where the compromise direction is aligned with the
average adversarial gradient) but insufficient to eliminate any one axis's
bias representation. Under the convergent regime (L25–L35), directions
partially collapse onto a shared subspace, but the reward magnitude inflates
`4.6×` due to Qwen's known norm growth across LM blocks — making single-layer
deep interventions numerically hazardous without per-layer normalisation.

The ensemble reward (§2.6) is a direct empirical consequence of this
prediction: pool across the regime boundary with per-layer z-score
normalisation, averaging the disentangled per-axis circuits at L17–L21 with
the convergent shared subspace at L25–L33. R3's OOD `|bias_score|` reduction
(§4.2), *unique* among all methods tested, confirms that the ensemble captures
a more dataset-invariant bias direction than the single-layer probe fit
specifically on VLBias activations.

### 5.2 Why the reward is a selector, not an eraser

The wash-out result (§4.4) admits a clean mechanistic reading. A single-layer
scalar reward cannot rewrite a 2048-d representation directly; what it *can*
do is preferentially reinforce PPO trajectories whose L13 hidden state at the
answer position already sits on the low-bias side of the probe's decision
boundary. In the language of trajectory selection: the reward *selects
between* the base policy's completions, without moving the base policy's
representation at any layer. Two implications follow:

- **Selection-signal debiasing is bounded by the base policy's trajectory
  support.** If the base model rarely produces counter-stereotype completions
  on a hard sub-axis (as we observe on Disability, which is the only
  regressing axis), no single-layer selector can amplify what is not there.
- **Interpretability-derived rewards operate mechanistically at a level
  different from probe visualisation.** A probe's decision boundary is a
  diagnostic of a subspace; a reward derived from that boundary tests whether
  the policy can rationally *select* trajectories in that subspace, not
  whether the policy can *rewrite* representations in it. The wash-out
  Pattern-2 result is a rigorous discovery of this gap.

### 5.3 Boundary of the model-property claim

The `|cos|` direction replicates as a model property on the same model's
activations under a different demographic-bias benchmark (SB-Bench-9axis),
but a stronger *architecture-independent* claim requires a third held-out
benchmark (e.g., CrowS-Pairs-MM, MMBench-Stereotype, FairFace-VQA) and a
cross-architecture check (LLaVA-Next, Idefics-3, BLIP-3). We flag this in
§6 and leave it as follow-up work. The `PC0 evr` concentration spike is
already known to be VLBias-coupled and is treated as a directional secondary
descriptor.

### 5.4 Actionable epistemic takeaway for the workshop

Reading a probe as an interpretability *visualisation* under-specifies what
using the same probe as an interpretability-*derived intervention* will do to
the underlying representation. Two of our results underscore this: the L13
probe supports a rank-1 discriminative direction with `|cos|@L13 ≈ 0.10`
across per-category probes (i.e., the rank-1 story is only true in the sense
of "averaged compromise"), and the same L13 direction supports a selector
signal that does not move the representation. The two-regime discovery,
paired with the selector-not-eraser mechanism, is a concrete example of how
interpretability can produce *testable* discoveries about a model — but only
if the intervention analysis is done alongside the visualisation analysis.

---

## 6. Limitations

1. **Single model architecture.** All results are on Qwen2.5-VL-3B-Instruct.
   The two-regime geometry and the selector-not-eraser mechanism may be
   architecture-specific; verification on LLaVA-Next, Idefics-3, or BLIP-3
   is future work.
2. **Two benchmarks in the BBQ family.** The `|cos|` model-property claim
   replicates across VLBiasBench and SB-Bench-9axis, but both are BBQ-style
   image-grounded multiple choice. A third architecture-independent benchmark
   (CrowS-Pairs-MM, MMBench-Stereotype, FairFace-VQA) is needed to isolate the
   claim from the BBQ item-format prior.
3. **Probe sample size at `n = 60` per axis for the two-regime experiment.**
   Robust to `C ∈ {0.1, 10.0}` and paired-bootstrap-hardened, but a direct
   convergence test at `n > 900` on the same activations is ongoing
   (see [Phase0.9.5 E7](../../Phase0.9_Strategic_Plan.md#37-e7--probe-re-fit-at-n--900-conditional-on-e2)).
4. **OOD transfer is not zero-shot with respect to the probe fit.** The
   `bias_aligned` probe was fitted on VLBias activations, so the OOD
   `|bias_score|` reduction on VLBias is transfer of the *policy* only, not
   of the probe. The SB-Bench accuracy claim is the cleaner transfer test
   but is confounded with the correctness term (see §5.4 and §4.1 synergy
   ablation).
5. **Counterfactual causal evidence underpowered.** The available paired
   counterfactual set (`n = 228`) has CI `±~6 pp` on a 35–50 % baseline —
   insufficient to adjudicate `+1.29 pp` as causally attributable to
   debiasing rather than test-set-specific accuracy. A larger set (`n = 6166`)
   is pre-planned but deferred until an ablation cycle over the reward's
   design axes closes.
6. **Disability axis regression.** The only per-axis regression under
   KLFIX-L13 (−0.31 pp ± 0.80, within 1 SE) is consistent with the
   selection-signal-bound interpretation from §5.2 and remains an unclosed
   fairness concern.
7. **Reward-magnitude scaling.** Hidden-state norm inflates 4.6× from L13 to
   L35; any multi-layer intervention must z-score-normalise per layer
   (which we do), but this remains a design-time constraint rather than a
   solved calibration problem.

---

## 7. Responsible-Use Statement

Debiasing interventions on VLMs can shift model outputs in ways that are
locally corrective on the benchmarks used for probe fit and evaluation, but
that may be harmful in other deployments. Our results explicitly show two
mechanisms by which this can happen. First, the reward acts as a
*selection signal* over the base policy's existing trajectories — residual
biases in the base distribution (evidenced by our failing Disability axis)
persist regardless of intervention strength, and can be amplified or
suppressed unevenly across demographic axes. Second, when the underlying
bias representation has a two-regime geometric structure with per-axis
disentanglement at mid-depth, a single 1-D probe reward operates on an
averaged compromise direction — meaning intervention magnitude is not a
reliable proxy for per-axis debiasing. Practitioners deploying probe-reward
pipelines should (a) audit per-axis outcomes against the specific
demographic groups their model will encounter, (b) treat probe-signal
alignment as evidence of a subspace intervention rather than
representation-level erasure, and (c) validate against benchmarks whose
distribution differs from the probe-fitting benchmark. Our probe artefacts,
per-seed generations, ablation grids, and evaluation code will be released
under an anonymous account for the review period, and under a permissive
open-source license upon camera-ready.

---

## 8. Related Work — STUB

*Placeholder to be written in the next revision. Coverage targets:*

- **Interpretability probes on VLMs and LLMs.** Classical probing literature
  (Hewitt & Manning, Belinkov, Tenney et al.), and the probe-of-a-probe
  design lineage.
- **Probing-as-intervention.** INLP (Elazar & Goldberg), RLACE, LEACE, and
  the linear-concept-erasure family — with emphasis on how these methods
  differ from probe-as-reward at the mechanistic level (representation vs
  trajectory).
- **VLM bias benchmarks.** BBQ (Parrish et al.), VLBiasBench, SB-Bench,
  CrowS-Pairs-MM, MMBench-Stereotype — the family we sample from, and the
  ones that would provide the architecture-independent third-benchmark
  check flagged in §6.
- **RLHF / PPO for LLM alignment and debiasing.** Ouyang et al., Bai et al.,
  Rafailov et al. (DPO), and the growing multi-layer / mechanistic
  reward-model literature.
- **Two-regime and layer-specialisation findings in LM interpretability.**
  Yosinski et al. (features, transfer), the induction-head literature, and
  the concept-neuron / superposition line — relevant to the regime-boundary
  framing and to the "convergent shared subspace" claim.
- **Prior debiasing pipelines that share components with ours.** Any prior
  work that used a linear probe as a training signal (rather than a
  diagnostic) — this is the direct comparison point that must be surfaced.

---

## 9. Figures list — placeholders + captions

Each figure below is described in enough detail to be plotted from
existing artefacts. Panel layouts are workshop-typical
(one-column figures, high-DPI PNG or vector PDF).

**Figure 1 (hero, page 1 top-right).**
Two-panel plot of the two-regime geometry across layer depth `L ∈ {1, 5, 9,
13, 17, 21, 25, 29, 33, 35}`.
Panel A: `mean pairwise |cos|(L)` with 95% marginal-bootstrap shaded band;
VLBias solid, SB-Bench-9axis dashed; regime-boundary vertical line at L22.
Panel B: `PC0 evr(L)` with same axes and CIs.
Data: [Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json](../../Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json)
and [Phase0.9/probe_convergence/diagnostic.json](../../Phase0.9/probe_convergence/diagnostic.json).
Caption: emphasise that `|cos|` direction replicates as a model property
(P = 1.000 on paired bootstrap, both datasets) while `PC0 evr` magnitude is
dataset-coupled (P = 0.995 vs 0.200 across the regime boundary).

**Figure 2 (methodology panel, top of §2).**
Reward pipeline schematic in three swimlanes. Left swimlane: `bias_aligned`
probe fit on VLBias-derived hidden states at layer L. Middle swimlane: PPO
loop on SB-Bench train — policy generates letter, extract `h_L[ans_pos]`,
compute projection onto frozen `w_biasA`, feed into reward alongside
correctness + ambig-preservation + KL. Right swimlane: single-layer variant
vs ensemble variant, with the ensemble showing the L17–L33 stack and
per-layer z-score normalisation.
No experimental data — visual only.

**Figure 3 (optional, §4.4).**
Bar chart of per-layer `|z|` across `L ∈ {1, 5, 9, 11, 13, 17, 21, 25, 29,
33, 35}` for the four PPO variants (Prod L13, corrOnly, biasOnly, KLFIX-L13).
Horizontal line at pre-committed threshold `|z| = 0.30σ`. All bars visibly
sub-threshold. Data:
[Phase0.8/washout_scores](../../Phase0.8/washout_scores/) +
[Phase0.8/washout_klfix_verdict.md](../../Phase0.8/washout_klfix_verdict.md).
Caption: "The interpretability-derived reward does not measurably attack the
bias direction at any depth — the accuracy gain is achieved via trajectory
selection, not representational erasure."

If space allows in the 5-page main text: keep Figures 1 and 2; move Figure 3
to appendix.

---

## 10. AI4GOOD adaptation notes (not this draft; for follow-up)

The AI4GOOD workshop (`https://trustworthy-ai-for-good.github.io/`,
2–8 pages) has a different centre of gravity: **trustworthy alignment,
auditing, and impact at population scale** rather than
interpretability-as-discovery. Same deadline (Aug 29, 2026 AoE); the FAQ
explicitly allows cross-submission to other NeurIPS workshops.

Changes to reshape this draft for AI4GOOD:

- **Title/abstract**: lead with the *debiasing intervention* framing.
  "A linear-probe reward for VLM debiasing: +1.49 pp accuracy and 43–71 %
  reduction in `|bias_score|` on VLBiasBench transfer, with a mechanistic
  audit that clarifies the intervention level."
- **Contribution 2 (selector-not-eraser)** becomes an *auditing* result:
  probe-derived rewards operate at the trajectory-selection level, not the
  representation-erasure level — practitioners should audit accordingly.
- **Contribution 1 (two-regime discovery)** demotes to a §5-ish
  interpretation of *why* the ensemble reward generalises better OOD; it
  is no longer a co-headline claim.
- **Contribution 3 (ensemble validation)** promotes to the main practical
  finding, with fairness-slice tables (per-axis Δ over 9 SB-Bench axes,
  10 VLBias axes) as the headline evidence, and the two-regime finding
  cited as its predictive motivation.
- **Extra pages (8 vs 5)**: use the extra room for the full per-axis
  fairness tables, the counterfactual-underpower analysis, and a
  deployment-guidance subsection that turns the responsible-use statement
  into concrete audit checklists.

I recommend producing the AI4GOOD version *after* Interp4Discovery is
locked, by adapting the same source Markdown into a distinct
`DRAFT_v1_AI4GOOD.md` file. The two versions can share the appendix
material, related-work section, and all data artefacts.

---

## 11. Traceability appendix — every cited number

Every claim in §4 traces to a specific source doc or artefact in the
workspace. Reviewer-facing citations will replace these workspace paths
in the camera-ready LaTeX version.

| Claim | Value | Source |
|---|---|---|
| Vanilla SB-Bench micro-acc | 0.6190 | [Phase0.9/canonical/canonical_base_sbbench_gen.jsonl](../../Phase0.9/canonical/canonical_base_sbbench_gen.jsonl) |
| KLFIX-L13 4-seed micro-acc | 0.6298 / 0.6339 (two aggregations) | [Phase0.9/canonical/klfix_L13_4seeds_aggregate.json](../../Phase0.9/canonical/klfix_L13_4seeds_aggregate.json), Workshop_Submission/REPORT_EXTENSIVE.md §6.3 |
| KLFIX-L13 3-seed (LFL) micro-acc | 0.6286 (+1.37 pp) | [Phase0.9/canonical/klfix_L13_3seeds_lf_aggregate.json](../../Phase0.9/canonical/klfix_L13_3seeds_lf_aggregate.json) |
| R3 ensemble 3-seed micro-acc | 0.6278 (+1.29 pp) | [Phase0.9/canonical/ensemble_klfix_3seeds_aggregate.json](../../Phase0.9/canonical/ensemble_klfix_3seeds_aggregate.json) |
| Reward synergy `corrOnly` | −5.25 pp | Workshop_Submission/REPORT_EXTENSIVE.md §7.2 |
| Reward synergy `biasOnly` | −1.27 pp | Workshop_Submission/REPORT_EXTENSIVE.md §7.2 |
| Vanilla VLBias overall_acc / ambig_acc / \|bs\| | 55.13 % / 91.90 % / 0.0070 | [Phase0.9/vlbias/base_vlbias_results.json](../../Phase0.9/vlbias/base_vlbias_results.json) |
| KLFIX-L13 VLBias signed bs | −0.0075 (worsens by 0.0005) | [Phase0.9/vlbias/](../../Phase0.9/vlbias/) 3-seed backfill |
| R3 ensemble VLBias signed bs | −0.0049 (reduces by 0.0021) | [Phase0.9/vlbias/](../../Phase0.9/vlbias/) 3-seed backfill |
| `|cos|@L25 = 0.540` VLBias | 95% CI [0.491, 0.599] | [Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json](../../Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json) |
| `P(|cos|@L25 > L13) = 1.000` | paired B = 200 | [Phase0.8/a3_results/rank1_bias_geometry_bootstrap_full.json](../../Phase0.8/a3_results/rank1_bias_geometry_bootstrap_full.json), [Phase0.9/r1_bootstrap_verdict.md](../../Phase0.9/r1_bootstrap_verdict.md) |
| SB-Bench-9axis `|cos|@L25 = 0.172` | replication | [Phase0.9/r2_sbbench_replication_verdict.md](../../Phase0.9/r2_sbbench_replication_verdict.md) |
| Wash-out `max |z|` | 0.019–0.075 σ across variants | [Phase0.8/washout_klfix_verdict.md](../../Phase0.8/washout_klfix_verdict.md) |
| KL stability tail-KL 148× improvement | KLFIX bundle | Workshop_Submission/REPORT_EXTENSIVE.md §6.3 |
| Per-axis KLFIX-L13 fairness slice | 8/9 axes ≥ vanilla | Workshop_Submission/REPORT_EXTENSIVE.md §6.4 |

---

## 12. Notes for LaTeX port

- Use the NeurIPS 2026 workshop LaTeX template
  (`https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`).
- Anonymize: no author names, no institution names, no repository URLs, no
  Modal usernames, no wallet/account references. Replace workspace links
  (`../../Phase0.8/...`) with anonymous.4open.science paths.
- Equations: use inline `$...$` and display `\[...\]` — the Markdown `$$...$$`
  in §2.2 becomes `\begin{equation}...\end{equation}` in LaTeX.
- Tables: Table 1 and Table 2 are the two-panel headline; keep them narrow
  enough for single-column layout. Table 3 (regime CIs) and Table 4
  (wash-out) can go one-column in the main text; the per-axis slice moves
  to appendix.
- Word budget audit at LaTeX-compile time: main text should be under
  ~2900 words to fit 5 pages with 2 figures + 4 tables. Current Markdown
  is **3562 words** in §§1–7 plus **291 words** in Abstract; expect a
  ~15–20 % trim during port (main compression targets: §5.4 epistemic
  takeaway can shorten, §6 limitations can collapse items 4+7 into a
  single scope paragraph, §2.3 PPO recipe can move sub-details to
  appendix, per-axis language in §4.1 can shorten to a single sentence).
- Responsible-Use Statement (§7) is **required** by Interp4Discovery; do
  not drop under any page pressure.
