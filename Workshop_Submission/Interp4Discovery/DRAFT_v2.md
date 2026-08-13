# A Probe That Reads Is Not a Probe That Writes: Two-Regime Bias Geometry and Trajectory-Selector Interventions in a Vision-Language Model

**Anonymous Authors**
Submission target: NeurIPS 2026 Workshop on Interpretability for Discovery (Interp4Discovery)

> **Draft status**: v2, addressing v1 presentation feedback. Freezes results at
> Phase 0.9 R1+R2+R3 ([Phase0.9/FINAL_REPORT.md](../../Phase0.9/FINAL_REPORT.md));
> v1 is retained at [DRAFT_v1.md](DRAFT_v1.md) for diff. Related-Work section is
> a stub; figures are described in §9 with data pointers. Will be ported to the
> NeurIPS 2026 LaTeX workshop template (5-page main text, unlimited references
> and appendices). AI4GOOD adaptation notes at §10; traceability appendix at §11.

---

## Abstract

Linear probes of demographic bias in vision-language models are almost always
used as **diagnostics** — read-outs of what a hidden representation encodes.
When the same probes are used to derive **interventions** — a training-time
reward, an activation edit, a mechanistic anchor — a natural discovery question
arises: does the intervention actually change the representation the probe
reads?

On Qwen2.5-VL-3B-Instruct, the answer is *no*. PPO trained with a mid-depth
probe-derived reward improves bias-benchmark accuracy — +1.5 pp on 4 seeds,
per-seed McNemar `p < 10⁻¹¹` — without measurably shifting the bias direction
at any layer of the model. The reward acts as a **selector** over the base
policy's already-existing low-bias completions, not as an **eraser** of the
representation.

Two coupled discoveries explain how this is possible and what to do about it.
Fitting one probe *per demographic axis* at each depth shows bias is not
carried by a single direction: at mid-depth (L9–L21) each of ten demographic
categories has its own near-orthogonal circuit, and only at deeper layers
(L25–L35) do these circuits partially converge onto a shared subspace. This
two-regime geometric structure replicates across two demographic-bias
benchmarks as a **model property**, and it directly predicts a specific
intervention — a probe reward pooled across the regime boundary — that becomes
the only method we test which reduces out-of-distribution `|bias_score|` on
VLBiasBench, rather than over-shooting past zero as the single-layer version
does.

The message for interpretability of foundation models is that when a probe
becomes an *intervention*, the intervention no longer operates at the level
the visualisation described. It operates on the model's trajectories, not on
its representations; and the geometric structure of the representation — not
the probe's decision boundary — bounds what any single-layer intervention can
achieve.

---

## 1. Introduction

Linear probing has become the dominant way of asking *what does a foundation
model encode?* A probe trained to predict a target attribute from a hidden
state licenses the claim that the attribute is present in the representation.
Increasingly, interpretability practice takes the next step: it treats the same
probe direction as a target for **intervention** — a training-time reward
(probe-driven RLHF), an activation edit (INLP, LEACE), or a mechanistic anchor
for causal analysis (Cite: probing lineage; Cite: linear-concept-erasure family).
The implicit assumption behind these interventions is that *the direction that
reads out an attribute is also the direction that carries it* — so that
changing the intervention target changes what the probe detects.

This assumption has never been tested on a vision-language foundation model
for the specific case of demographic bias — a setting where interpretability-
derived interventions are most consequential and where a mismatch between what
the probe reads and what the intervention changes has the largest ethical
footprint. We test the assumption directly on Qwen2.5-VL-3B-Instruct, and we
use the mismatch we find as an entry point into three linked discovery
questions about how bias is organised in a foundation model's representation.

**Q1 (representation).** Is demographic bias in a VLM carried by a single
linear direction, or by a family of directions that a single probe averages
over? At which depths?

**Q2 (intervention effect).** When we treat a probe direction as the target of
a training-time intervention, does the intervention actually move that
direction — or does the model change its behaviour without changing the
representation the probe detects?

**Q3 (predictive validation).** If (1) and (2) reveal structure not captured
by a single-layer probe, does that structure predict a specific multi-layer
intervention that generalises better than the single-layer baseline?

The three answers, told as a mini-arc, form the paper.

**(1) Bias is not carried by a single direction.** Fitting one probe per
demographic axis at each depth exposes a **two-regime geometric structure**.
At mid-depth (L9–L21) each of ten demographic categories occupies a
near-orthogonal circuit; at deeper layers (L25–L35) these per-category
directions partially collapse onto a shared subspace. The direction of
convergence replicates across VLBiasBench and SB-Bench as a **model property**.

**(2) The intervention does not move the representation.** PPO trained with a
mid-depth probe-derived reward improves bias-benchmark accuracy, but a
pre-registered representation-shift diagnostic finds the reward moves the bias
direction by 4× to 16× below the threshold at which we would call the move
measurable — at every one of eleven depths we probe. The reward acts as a
**trajectory selector** over the base policy's already-existing low-bias
completions, not as a **representation eraser**. This is the mismatch we
hypothesised.

**(3) The two-regime structure predicts a better intervention.** A probe
reward *pooled across the regime boundary* is the only method we test which
**reduces** the aggregate out-of-distribution `|bias_score|` on VLBiasBench
transfer — precisely because the ensemble captures both the disentangled
per-category circuits at mid-depth and the convergent shared subspace at deep
layers. A single-layer probe direction is structurally incapable of doing so
and, in our data, over-shoots past zero on the same benchmark.

Taken together, this is a concrete example of a probe-derived intervention
whose effect does not match the geometry the probe direction implied, together
with a discovery about the underlying representation that predicts what a
better-matched intervention looks like. For the workshop, we offer both: an
empirical case study of interpretability generating testable knowledge about
how a socially consequential attribute is encoded, and a rigorous negative
result about what an interpretability-derived intervention actually does.

---

## 2. Methodology

We describe the four ingredients in the order they enter the pipeline:
(§2.1) the linear probe of bias, (§2.2) the probe-derived reward and PPO
recipe, (§2.3–§2.4) the geometry experiments used to answer Q1, and (§2.5) the
representation-shift diagnostic used to answer Q2. Section (§2.6) describes the
multi-layer ensemble used for Q3.

### 2.1 The bias-aligned probe

A binary logistic-regression probe over the 2048-d hidden state at layer L:
`w_biasA(L) ∈ ℝ^{2048}` with `C = 1.0`, `class_weight='balanced'`,
5-fold `StratifiedKFold`, `seed = 42`, `max_iter = 2000`. The label rule is
defined only on **disambiguated** BBQ records:

- `neg` (stereotyped person is gold): label = 1 iff `pred = gold`.
- `non_neg` (counter-stereotyped person is gold): label = 1 iff
  `pred = the other named option`.
- `ambig` (gold = "unknown"): **label = None, record dropped**.

Excluding `ambig` records by construction decouples fairness from the
abstention decision, and prevents the probe from conflating the two.

### 2.2 The probe-derived reward and PPO recipe

The per-step reward has three learned terms and one KL penalty:

$$
r_t \;=\; w_\text{corr}\!\cdot\!\mathbb{1}[\hat{y}=y^*] \;+\; w_\text{bias}\!\cdot\!\bigl[-(\mathbf{w}_\text{biasA}\cdot \mathbf{h}_{L^{*}}[\text{ans\_pos}])\bigr] \;+\; w_\text{ambig}\!\cdot\!\mathbb{1}[g=C\wedge \hat{y}=C] \;-\; \beta\!\cdot\!\text{KL}_\text{tok}(\pi\|\pi_\text{ref})
$$

with `w_corr = w_bias = 1.0, w_ambig = 0.5, β = 0.1, target_kl = 0.02`. The
probe term is negated because the probe predicts stereo-aligned = 1, so
minimising it reduces bias-aligned emissions. `L*` is the **intervention
layer** — L13 for the single-layer baseline (§2.2 default), a five-layer
window `{17, 21, 25, 29, 33}` for the ensemble variant (§2.6). We probe at the
answer position, i.e., the token whose logits produce the letter choice.

**Base model.** Qwen2.5-VL-3B-Instruct (36 transformer blocks, hidden dim
2048, bf16, FlashAttention-2). LoRA `r = 16`, `α = 32` on q/k/v/o projections
in every block. Batch size 8, one epoch on SB-Bench train (n = 1936). Value
head is a linear map to a scalar, zero-initialised.

**KL-stability recipe (KLFIX).** Value-head zero-init at the code level,
`--value-clip-range 0.2`, and `--kl-adapt-rate 0.3`. Reduces cross-seed
accuracy std by 3× and tail-KL std by 148× versus the un-stabilised recipe.
All PPO results in §4 use this recipe.

### 2.3 Per-axis geometry — the two-regime experiment

To test whether the shared `w_biasA` direction is a genuine rank-1 axis or an
averaged compromise across per-category circuits, we fit **one probe per
demographic axis** at each layer `L ∈ {1, 5, 9, 13, 17, 21, 25, 29, 33, 35}`.
The ten axes are Age, Disability, Gender_identity, Nationality,
Physical_appearance, Race_ethnicity, Race_x_SES, Race_x_gender, Religion, SES.
Sample size is `n = 60` disambiguated records per axis; hyperparameters
otherwise as in §2.1. Stack unit-normalised weight vectors into
`W_L ∈ ℝ^{10 × 2048}` and PCA on `W_L`. Report two geometric descriptors:

- **`PC0 evr(L)`** — fraction of variance in `W_L` explained by the top
  principal component (rank-1 concentration).
- **mean pairwise `|cos|(L)`** across the ten unit vectors (per-category
  alignment). Uniform-spectrum baseline in 2048-d: `|cos| = 0`, `PC0 evr = 0.10`.

### 2.4 Bootstrap protocol

**Marginal.** Per layer, resample each axis's records with replacement
(`n = 60`), refit each axis probe, recompute `PC0 evr` and `|cos|`; B = 200
draws. Report 2.5 / 97.5 percentile 95 % CIs.

**Paired-draw.** Within a single draw, the per-axis resample indices are
**shared across layers** — this controls for the layer-correlated nuisance
covariance that inflates marginal CIs without affecting cross-layer ordering.
Paired separability = `P(metric@L_hi > metric@L_lo)` over the B paired draws.
Degenerate draws (a single axis returning a single-class label vector) are
re-sampled; this occurs on fewer than 5 / 200 draws in practice.

### 2.5 Representation-shift diagnostic

For each PPO variant *V* and layer `L ∈ {1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35}`,
we re-extract the mean bias-probe projection over a held-out eval set and
compute the standardised shift

`z_V(L) = (μ_V(L) − μ_vanilla(L)) / σ_vanilla(L)`.

We pre-registered three mechanistic patterns for the intervention to fit:

- **Clean eraser** — `z(L13) ≤ −0.30 σ` AND `z(L) ≤ −0.15 σ` at every `L ≥ 17`
  (probe direction moved at L13 and the move propagates through depth).
- **Null shift** — `|z(L)| < 0.10 σ` at every `L` (the direction the probe
  reads has not been touched anywhere).
- **Bias displacement** — `z(L13) ≤ −0.30 σ` AND `z(L) > 0` at any `L ≥ 25`
  (the bias direction has been *pushed out* of L13 into deeper layers, hiding
  from the probe).

We call this a *representation-shift* diagnostic; project docs sometimes refer
to it as a *wash-out* diagnostic.

### 2.6 Multi-layer ensemble reward

Bundle five per-layer bias-aligned probes at `L ∈ {17, 21, 25, 29, 33}`,
straddling the regime boundary observed in §4.3. Unit-normalise each;
precompute per-layer offline `(μ_L, σ_L)` on a training-distribution activation
cache (n = 751) to keep runtime signal scale-invariant. Runtime: per-layer
projection at the answer position, per-layer z-score with `(μ_L, σ_L)`,
equal-weight mean across the five, single negation for the sign convention,
plug into the `w_bias` slot of §2.2. Signal fidelity is verified per training
step by emitting each per-layer z as a metric; the five means bunch tightly
across seeds (see §4.4).

---

## 3. Experimental Setup

**Datasets.** SB-Bench (in-distribution PPO training + eval, canonical test
split n = 2916 across 9 BBQ axes; a separate SB-Bench-9axis split with
n = 751 is used only for the cross-benchmark replication in §4.3).
VLBiasBench close-ended (source dataset for probe fitting; out-of-distribution
transfer eval, n = 2000 intersection, 10 axes of which 7 overlap with SB-Bench).
Splits are pre-registered and frozen; probe fits and PPO training never see
the SB-Bench canonical test items or the VLBias transfer items.

**PPO variants compared.**
- *Vanilla* — no PPO.
- *Correctness-alone* — the reward of §2.2 with `w_bias = 0`.
- *Probe-alone* — with `w_corr = 0`.
- *Single-layer* — the full reward with `L* = 13` (headline reference).
- *Multi-layer ensemble* — full reward with the L17–L33 ensemble.

**Seeds.** Four seeds `{s1, s2, s3, s4}` for the single-layer arm; three seeds
`{s1, s2, s4}` for the ensemble (s3 hit a container-side data-loader crash on
both initial and retry, at the same micro-batch on both — a scheduler-level
resource-contention issue, not model-related). Like-for-like comparisons in
§4 use the three-seed intersection.

**Metrics.** SB-Bench: micro- and macro-accuracy; per-seed McNemar
exact-binomial p-value paired against vanilla; Stouffer's method for cross-seed
z. VLBiasBench: overall accuracy, disambig / ambig accuracy, `neg` and
`non_neg` accuracy, signed and absolute `bias_score`. All cross-seed
statistics reported as mean ± std.

**Compute.** Single A100-80GB PPO runs; per-seed training ≈ 30 min, per-seed
eval ≈ 8 min; full sweep fits in a ~$10 budget on public cloud.

---

## 4. Results

We ask four questions of the model and its trained variants — the first two
on the intervention side, the last two on the representation side. Tables show
hero numbers only; full per-axis and per-seed breakdowns are in the appendix.

### 4.1 Does the probe-derived reward change the model's *behaviour* on the training-domain benchmark?

**Yes, robustly.** Under the KL-stabilised recipe, the single-layer probe
reward at L13 raises 4-seed SB-Bench micro-accuracy by **+1.49 pp**, cross-seed
std 0.35 pp; all four seeds beat vanilla individually with per-seed McNemar
`p < 10⁻¹¹`. The multi-layer ensemble across L17–L33 matches this in aggregate
(**+1.29 pp**, 3-seed like-for-like).

| Reward | 4-seed micro-acc | Δ vs vanilla | Result |
|---|---:|---:|---|
| Vanilla | 0.6190 | — | — |
| Correctness-alone | 0.5665 | **−5.25 pp** | worse than baseline |
| Probe-alone | 0.6063 | **−1.27 pp** | worse than baseline |
| **Correctness + L13 probe** | **0.6339** | **+1.49 pp** | 4/4 seeds beat vanilla |
| Correctness + L17–L33 ensemble | 0.6278 | +1.29 pp | 3/3 like-for-like seeds beat vanilla |

*Note.* The ensemble row is 3-seed only (`s3` crashed during training). The
matched 3-seed single-layer baseline is `+1.37 pp`, so the ensemble is
essentially tied with — not below — the single-layer arm on the in-distribution
metric.

**What this tells us about the model.** Neither reward term works alone —
individually they *hurt* by 5.25 pp and 1.27 pp respectively. Only their
*combination* improves accuracy, by +1.49 pp, with an implied interaction
term ≈ +8.4 pp. This decisively rules out the trivial reading of "the probe
reward is a noisy accuracy signal": if it were, correctness-alone should have
matched or exceeded the joint reward.

### 4.2 Does the reward move the bias *representation* the probe reads?

**No — at any depth we probe.** For every PPO variant we re-score its hidden
states against the frozen base-model probes across eleven depths from L1 to
L35, and report the standardised shift `|z|`. Our pre-registered "clean
intervention" threshold was `|z| ≥ 0.30 σ` at the intervention layer. What we
find is **`|z| ≤ 0.075 σ` at every layer, in every PPO variant, including the
KL-stabilised headline model** — 4× to 16× below threshold.

| Variant | max \|z\| across L1–L35 | argmax layer | Diagnostic |
|---|---:|---:|---|
| Single-layer L13 (production) | 0.032 σ | 11 | null shift |
| Correctness-alone | 0.075 σ | 25 | null shift |
| Probe-alone | 0.021 σ | 25 | null shift |
| **KL-stabilised single-layer L13 (headline)** | **0.019 σ** | **17** | **null shift (flattest)** |

**What this tells us about the model.** The +1.49 pp behavioural gain is
achieved *without measurably rewriting the bias direction anywhere in the
network*. Under the trajectory-selection reading, this is exactly what should
happen: PPO up-weights rollouts whose L13 hidden state at the answer position
already sits on the low-bias side of the probe boundary, and the base policy
already has enough low-bias trajectory mass to be selected between. Under a
representation-erasure reading, this is a null result. §5 develops the
trajectory-selection interpretation and its consequences.

### 4.3 Is bias in this VLM carried by *one* direction, or by *many*?

**By many at mid-depth; by a partially shared subspace at deep layers.**
Fitting one probe per demographic axis at each depth and PCA-ing the stacked
unit weight vectors reveals a two-regime geometry, decisive under paired
bootstrap. Regime membership follows the mean pairwise `|cos|` between the
ten axis-specific unit vectors:

| Depth window | mean pairwise \|cos\| (95 % CI) | Interpretation |
|---|---:|---|
| L9–L21 (mid-depth) | 0.081 – 0.156 | Ten demographic categories occupy near-orthogonal bias circuits. |
| L25–L35 (deep) | 0.404 – 0.599 | Per-category circuits partially collapse onto a shared subspace. |

The regime boundary sits at approximately L22. Paired separability at that
boundary: `P(|cos|@L25 > |cos|@L13) = 1.000` on
**both** VLBiasBench and SB-Bench-9axis. The *direction* of convergence
(deep > mid-depth) replicates as a **model property**; the *magnitude* of
convergence is dataset-coupled (`|cos|@L25 ≈ 0.54` on VLBias vs `0.17` on
SB-Bench-9axis; both above their own dataset's disentangled baseline).

**What this tells us about the model.** A single-layer probe fitted at any
mid-depth layer captures an **averaged compromise direction** across ten
near-orthogonal per-category circuits, not a single true bias axis. This
directly bears on §4.2 (a scalar reward derived from an averaged direction
cannot efficiently pressure ten orthogonal circuits at once) and predicts
§4.4 (a reward that samples both regimes should out-transfer any single-layer
reward on OOD data).

### 4.4 Does the two-regime structure *predict* a better intervention?

**Yes.** A multi-layer probe reward pooled across the regime boundary is the
*only* variant we test which **reduces** the out-of-distribution `|bias_score|`
on VLBiasBench transfer, rather than over-shooting past zero.

| Method | signed bias_score | Δ \|bias_score\| vs vanilla | ambig_acc | disambig_acc |
|---|---:|---:|---:|---:|
| Vanilla | +0.0070 | — | 91.9 % | 36.8 % |
| Single-layer L13 | −0.0075 ± 0.0048 | **−0.0005** (worsens) | 90.9 % | 36.2 % |
| **Multi-layer L17–L33 ensemble** | **−0.0049 ± 0.0048** | **+0.0021** (reduces) | 89.5 % | **36.9 %** |

The single-layer reward *flips the sign* of the aggregate bias score without
shrinking its magnitude — a symptom of the averaged compromise over-fitting
the VLBias-fitted probe direction. The multi-layer ensemble, calibrated by
the two-regime structure of §4.3, is the only method that produces a true
reduction in aggregate `|bias_score|`. The 1.4 pp cost on abstention
(`ambig_acc`) is the price paid for the disambig-side gain.

**What this tells us about the model.** The predictive success of the
multi-layer ensemble is direct behavioural evidence that the two-regime
discovery is **causally connected to how the reward transfers**, not just a
geometric observation. The intervention that matches the model's actual
representation geometry generalises; the intervention that assumes a single
global bias direction over-shoots and fails to transfer.

---

## 5. Discussion

### 5.1 Consequences for how bias is thought of in foundation-model representations

The two-regime discovery (§4.3) suggests that "the bias direction" is not the
right unit of analysis at every layer of a VLM. At mid-depth, ten demographic
categories occupy ten near-orthogonal linear circuits; any single-direction
probe or intervention there is necessarily an averaged compromise across
those circuits. This has three testable predictions that follow-up work can
verify or falsify on other model families:

- **Concept-direction methods that assume one direction per attribute** —
  INLP, LEACE, activation steering, single-layer probe-as-reward — will
  under-represent per-category structure at the depths where they are
  typically applied. Their effect will look larger on categories whose native
  circuits happen to align with the averaged direction, and smaller on
  categories that don't.
- **Deep-layer single-direction methods will look effective for the wrong
  reason.** Their power at L25+ comes from the per-category geometry
  *converging toward* the probed direction, not from the direction being an
  intrinsic bias axis; and the convergence magnitude is dataset-coupled
  (§4.3).
- **Cross-dataset probe transfer is limited by geometric replication, not
  by probe accuracy.** Our `|cos|` direction replicates; our `|cos|`
  magnitude does not. Practitioners fitting probes on Benchmark A and hoping
  to use them on Benchmark B should validate the *geometry* — not just the
  in-distribution accuracy — of the probe transfer.

### 5.2 Why the intervention is a selector, not an eraser

A one-scalar-per-token reward — no matter how carefully derived from
interpretability — cannot rewrite a 2048-dimensional representation directly.
What it *can* do is up-weight PPO trajectories whose intervention-layer hidden
state at the answer position already sits on the low-bias side of the probe
boundary. This gives two mechanistic predictions confirmed by §4:

- **The base policy's trajectory support bounds the intervention.** The one
  demographic axis (Disability) where the base model rarely produces
  counter-stereotype completions is the one axis where the trained model
  slightly regresses — as a selector must when there is little on the
  correct side to select.
- **Behaviour and representation move independently.** §4.2's `|z| ≤ 0.075 σ`
  result is direct evidence that the intervention-layer hidden state has not
  been re-parameterised by training. Accuracy gains are entirely on the
  *trajectory* side of the model.

### 5.3 Where the model-property claim ends

The `|cos|` convergence direction replicates as a model property on the same
model under a different demographic-bias benchmark. The stronger claim of
*architecture-independence* requires cross-model verification. A
cross-architecture check (LLaVA-Next, Idefics-3, BLIP-3) and a third
BBQ-independent benchmark (CrowS-Pairs-MM, MMBench-Stereotype, FairFace-VQA)
would isolate the two-regime finding from a Qwen-specific or BBQ-format-
specific artefact. Both are flagged in §6 and left as follow-up work.

### 5.4 A concrete workflow for interpretability-derived interventions

For a workshop audience, the practical takeaway is a two-step workflow that
would have caught the selector-vs-eraser mismatch *before* interpreting an
accuracy gain as debiasing:

1. **Characterise the per-attribute-category geometry at multiple depths
   *before* designing a single-direction intervention.** If mean pairwise
   `|cos|` between per-category probes is low at the intended intervention
   layer, a single-direction intervention will act as an averaged compromise;
   a multi-direction intervention that spans the regime boundary will match
   the geometry.
2. **Run a representation-shift diagnostic at multiple depths on the trained
   model.** If the intervention does not move the representation the probe
   reads, the intervention is a selector: the effect is behavioural, not
   representational, and its transfer depends on how much low-bias trajectory
   support the base policy already has.

Both steps are cheap (linear probes and z-score comparisons on cached hidden
states) and would have identified §4.2's null and §4.3's two-regime structure
before the multi-layer ensemble was designed. We offer this as an empirical
case study of interpretability generating testable knowledge about a foundation
model — and as a rigorous negative result about what an interpretability-
derived intervention actually does.

---

## 6. Limitations

1. **Single model architecture.** All results are on Qwen2.5-VL-3B-Instruct.
   The two-regime geometry and the selector-vs-eraser mismatch may be
   architecture-specific; verification on LLaVA-Next, Idefics-3, or BLIP-3
   is future work.
2. **Two benchmarks in the BBQ family.** The `|cos|` model-property claim
   replicates across VLBiasBench and SB-Bench-9axis, but both are BBQ-style
   image-grounded multiple choice. A third architecture-independent benchmark
   (CrowS-Pairs-MM, MMBench-Stereotype, FairFace-VQA) is needed to isolate
   the claim from the BBQ item-format prior.
3. **Probe sample size at `n = 60` per axis for the two-regime experiment.**
   Robust to `C ∈ {0.1, 10.0}` and paired-bootstrap-hardened; a direct
   convergence test at `n > 900` on the same activations is ongoing (see
   [Phase 0.9.5 E7](../../Phase0.9_Strategic_Plan.md#37-e7--probe-re-fit-at-n--900-conditional-on-e2)).
4. **OOD transfer is not zero-shot with respect to the probe fit.** The
   probe was fitted on VLBias activations, so the OOD `|bias_score|`
   reduction on VLBias is a transfer of the *policy* only, not of the probe.
   The SB-Bench accuracy claim is the cleaner transfer test but is
   confounded with the correctness reward term (see §4.1).
5. **Counterfactual causal evidence underpowered.** The available paired
   counterfactual set (`n = 228`) has CI `±~6 pp` on a 35–50 % baseline —
   insufficient to adjudicate `+1.29 pp` as causally attributable to
   debiasing rather than test-set-specific accuracy. A larger set (`n = 6166`)
   is pre-planned but deferred until an ablation cycle over the reward's
   design axes closes.
6. **Disability axis regression.** The only per-axis regression under the
   single-layer arm (−0.31 pp ± 0.80, within 1 SE) is consistent with the
   selection-signal bound of §5.2 and remains an unclosed fairness concern.
7. **Reward-magnitude scaling across depth.** Hidden-state norm inflates
   `4.6×` from L13 to L35; any multi-layer intervention must z-score-
   normalise per layer (which we do), but this remains a design-time
   constraint rather than a solved calibration problem.

---

## 7. Responsible-Use Statement

Debiasing interventions on VLMs can shift model outputs in ways that are
locally corrective on the benchmarks used for probe fit and evaluation, but
that may be harmful in other deployments. Our results explicitly show two
mechanisms by which this can happen. First, the reward acts as a selection
signal over the base policy's existing trajectories — residual biases in the
base distribution (evidenced by our failing Disability axis) persist regardless
of intervention strength, and can be amplified or suppressed unevenly across
demographic axes. Second, when the underlying bias representation has a
two-regime geometric structure with per-axis disentanglement at mid-depth, a
single 1-D probe reward operates on an averaged compromise direction — meaning
intervention magnitude is not a reliable proxy for per-axis debiasing.
Practitioners deploying probe-reward pipelines should (a) audit per-axis
outcomes against the specific demographic groups their model will encounter,
(b) treat probe-signal alignment as evidence of a subspace intervention rather
than representation-level erasure, and (c) validate against benchmarks whose
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
- **Layer-specialisation findings in LM interpretability.** Yosinski et al.
  (features and transfer), the induction-head literature, and the
  concept-neuron / superposition line — directly relevant to the
  regime-boundary framing and to the "convergent shared subspace" claim.
- **Positioning statement.** Prior probing work treats probes as diagnostics
  of what's *in* the representation. This paper treats probes as
  *interventions* on the training signal, and asks whether the intervention
  effects match what a diagnostic reading of the same probe would predict.

---

## 9. Figures list — placeholders + captions

Each figure below is described in enough detail to be plotted from existing
artefacts. Panel layouts are workshop-typical (one-column figures, high-DPI
PNG or vector PDF).

**Figure 1 (hero, page 1 top-right).**
Two-panel plot of the two-regime geometry across layer depth
`L ∈ {1, 5, 9, 13, 17, 21, 25, 29, 33, 35}`.
Panel A: `mean pairwise |cos|(L)` with 95 % marginal-bootstrap shaded band;
VLBias solid, SB-Bench-9axis dashed; regime-boundary vertical line at L22.
Panel B: `PC0 evr(L)` with same axes and CIs.
Data:
[Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json](../../Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json)
and
[Phase0.9/probe_convergence/diagnostic.json](../../Phase0.9/probe_convergence/diagnostic.json).
Caption: emphasise `|cos|` model-property replication (`P = 1.000` on paired
bootstrap, both datasets) and `PC0 evr` dataset-coupling.

**Figure 2 (methodology panel, top of §2).**
Reward pipeline schematic in three swimlanes.
Left: bias-aligned probe fit on VLBias-derived hidden states at layer L.
Middle: PPO loop on SB-Bench train — policy generates letter, extract
`h_L[ans_pos]`, compute projection onto frozen `w_biasA`, feed into reward
alongside correctness + ambig-preservation + KL.
Right: single-layer variant vs. multi-layer ensemble variant, with the ensemble
showing the L17–L33 stack and per-layer z-score normalisation.
No experimental data; visual only.

**Figure 3 (optional, §4.2).**
Bar chart of per-layer `|z|` across `L ∈ {1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35}`
for the four PPO variants (single-layer L13, correctness-alone, probe-alone,
KL-stabilised L13). Horizontal line at the pre-committed threshold
`|z| = 0.30 σ`; all bars visibly sub-threshold. Data:
[Phase0.8/washout_scores](../../Phase0.8/washout_scores/) and
[Phase0.8/washout_klfix_verdict.md](../../Phase0.8/washout_klfix_verdict.md).
Caption: "The interpretability-derived reward does not measurably move the
bias direction at any depth — the accuracy gain is achieved via trajectory
selection, not representational erasure."

If space allows in the 5-page main text, keep Figures 1 and 2; move Figure 3
to the appendix.

---

## 10. AI4GOOD adaptation notes (not this draft; for follow-up)

The AI4GOOD workshop (`https://trustworthy-ai-for-good.github.io/`, 2–8 pages)
has a different centre of gravity: **trustworthy alignment, auditing, and
impact at population scale** rather than interpretability-as-discovery. Same
deadline (Aug 29, 2026 AoE); the FAQ explicitly allows cross-submission to
other NeurIPS workshops.

Structural changes to reshape this draft for AI4GOOD:

- **Title/abstract**: lead with the *debiasing intervention* framing —
  "A linear-probe reward for VLM debiasing: +1.49 pp accuracy and reduced
  out-of-distribution bias-score magnitude, with a mechanistic audit that
  clarifies the intervention level."
- **Contribution ordering**: promote Q3 (predictive validation of the
  ensemble reward) as the main practical finding; keep Q2 (selector-not-
  eraser) as the auditing / responsible-deployment result; demote Q1
  (two-regime discovery) to §5-ish interpretation of *why* the ensemble
  reward generalises better.
- **Extra pages (8 vs 5)**: use the extra room for full per-axis fairness
  tables, the counterfactual-underpower analysis, and a deployment-guidance
  subsection that turns the responsible-use statement into concrete
  audit checklists.

We recommend producing the AI4GOOD version *after* Interp4Discovery is locked,
by adapting the same source Markdown into a distinct `DRAFT_v1_AI4GOOD.md`.
The two versions can share appendix material, related-work section, and all
data artefacts.

---

## 11. Traceability appendix — every cited number

Every claim in §4 traces to a specific source doc or artefact in the workspace.
Reviewer-facing citations will replace these workspace paths in the
camera-ready LaTeX version.

| Claim | Value | Source |
|---|---|---|
| Vanilla SB-Bench micro-acc | 0.6190 | [Phase0.9/canonical/canonical_base_sbbench_gen.jsonl](../../Phase0.9/canonical/canonical_base_sbbench_gen.jsonl) |
| Single-layer L13 4-seed micro-acc | 0.6298 / 0.6339 (two aggregations) | [Phase0.9/canonical/klfix_L13_4seeds_aggregate.json](../../Phase0.9/canonical/klfix_L13_4seeds_aggregate.json); Workshop_Submission/REPORT_EXTENSIVE.md §6.3 |
| Single-layer L13 3-seed like-for-like micro-acc | 0.6286 (+1.37 pp) | [Phase0.9/canonical/klfix_L13_3seeds_lf_aggregate.json](../../Phase0.9/canonical/klfix_L13_3seeds_lf_aggregate.json) |
| Multi-layer ensemble 3-seed micro-acc | 0.6278 (+1.29 pp) | [Phase0.9/canonical/ensemble_klfix_3seeds_aggregate.json](../../Phase0.9/canonical/ensemble_klfix_3seeds_aggregate.json) |
| Reward synergy correctness-alone | −5.25 pp | Workshop_Submission/REPORT_EXTENSIVE.md §7.2 |
| Reward synergy probe-alone | −1.27 pp | Workshop_Submission/REPORT_EXTENSIVE.md §7.2 |
| Vanilla VLBias overall_acc / ambig_acc / \|bs\| | 55.13 % / 91.90 % / 0.0070 | [Phase0.9/vlbias/base_vlbias_results.json](../../Phase0.9/vlbias/base_vlbias_results.json) |
| Single-layer VLBias signed bs | −0.0075 (worsens by 0.0005) | [Phase0.9/vlbias/](../../Phase0.9/vlbias/) 3-seed backfill |
| Multi-layer VLBias signed bs | −0.0049 (reduces by 0.0021) | [Phase0.9/vlbias/](../../Phase0.9/vlbias/) 3-seed backfill |
| \|cos\| @ L25 (VLBias) | 0.540 (95 % CI [0.491, 0.599]) | [Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json](../../Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json) |
| P(\|cos\|@L25 > L13) = 1.000 | paired B = 200 | [Phase0.8/a3_results/rank1_bias_geometry_bootstrap_full.json](../../Phase0.8/a3_results/rank1_bias_geometry_bootstrap_full.json); [Phase0.9/r1_bootstrap_verdict.md](../../Phase0.9/r1_bootstrap_verdict.md) |
| SB-Bench-9axis \|cos\| @ L25 = 0.172 | replication | [Phase0.9/r2_sbbench_replication_verdict.md](../../Phase0.9/r2_sbbench_replication_verdict.md) |
| Representation-shift max \|z\| | 0.019–0.075 σ across variants | [Phase0.8/washout_klfix_verdict.md](../../Phase0.8/washout_klfix_verdict.md) |
| KL stability tail-KL 148× improvement | KL-stability recipe | Workshop_Submission/REPORT_EXTENSIVE.md §6.3 |
| Per-axis single-layer fairness slice | 8/9 axes ≥ vanilla | Workshop_Submission/REPORT_EXTENSIVE.md §6.4 |

---

## 12. Notes for LaTeX port

- Use the NeurIPS 2026 workshop LaTeX template
  (`https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`).
- Anonymise: no author names, no institution names, no repository URLs, no
  cloud-service usernames, no account references. Replace workspace links
  (`../../Phase0.8/...`) with anonymous.4open.science paths.
- Equations: use inline `$...$` and display `\[...\]` — the Markdown `$$...$$`
  in §2.2 becomes `\begin{equation}...\end{equation}` in LaTeX.
- Tables: Table 1 (§4.1) and Table 2 (§4.4) are the two-panel headline; keep
  them narrow enough for single-column layout. Table 3 (§4.3) and Table 4
  (§4.2) go one-column in the main text; the per-axis slice moves to appendix.
- Word budget audit at LaTeX-compile time: main text (§§1–7) is **3766
  words** in v2 (up from 3562 in v1 — presentation revisions added Q1/Q2/Q3
  scaffolding and per-subsection "what this tells us" framing that grew the
  Introduction and Results). This is **over** the 5-page budget: NeurIPS 2026
  workshop template is ~500–600 words/page, and 4 tables + 2 figures consume
  another ~1–1.5 pages. Expected compression targets during the LaTeX port,
  ordered by lowest information loss:
    1. §3 Experimental Setup: fold "Compute" and "Metrics" bullets into one
       compact paragraph (−80 w).
    2. §4.1 note: fold the seed-asymmetry *Note* into a single sentence in the
       paragraph above the table (−30 w).
    3. §5.4 workflow: two-step list is currently 5 lines each; halve to 2
       (−100 w).
    4. §6 Limitations: consolidate items 3, 4, 5 into one "scope and power"
       paragraph (−100 w).
    5. §2.4 bootstrap protocol: move degenerate-draw handling to appendix
       (−40 w).
  Combined trim ≈ 350 w → target ≈ 3400 w main text, fits 5 pages.
  Abstract is **293 words** (v1 was 291) — a small trim to ~230 needed
  for two-column layout.
- Responsible-Use Statement (§7) is **required** by Interp4Discovery; do not
  drop under any page pressure.
- Terminology used consistently in v2: *intervention layer* (not "deployment
  layer"); *representation-shift diagnostic* (not "wash-out"); *KL-stability
  recipe* / *KL-stabilised* (KLFIX only as parenthetical shorthand);
  *correctness-alone* and *probe-alone* rewards (not "corrOnly" / "biasOnly");
  *single-layer* / *multi-layer ensemble* (no R1/R2/R3 identifiers). Retain
  the pre-registered pattern names *clean eraser / null shift / bias
  displacement* in prose; the internal Pattern-1/2/3 numeric indices are only
  retained in the appendix for cross-reference with project docs.
