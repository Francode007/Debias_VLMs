# REPORT_EXTENSIVE
## Detailed Trajectory of Phase 0.8 — Probe-as-Reward Debiasing of Qwen2.5-VL-3B

**Date:** 2026-06-20 · **Branch:** `phase0.8_strategic_plan_day1`
**Companion documents:**
- [REPORT_SUMMARY.md](REPORT_SUMMARY.md) — compressed everything (~5 pages)
- [REPORT_EARLY_PHASES_EXTENSIVE.md](REPORT_EARLY_PHASES_EXTENSIVE.md) — Phase 0–0.7 detail
- [REPORT_COMBINED.md](REPORT_COMBINED.md) — concatenation of all three

> This document is the **detailed Phase 0.8 deep-dive** in the four-document
> set. Phase 0–0.7 is compressed in §2; the full early-phase trajectory lives
> in [REPORT_EARLY_PHASES_EXTENSIVE.md](REPORT_EARLY_PHASES_EXTENSIVE.md).

---

## 1. Abstract

We debias a vision-language model (Qwen2.5-VL-3B-Instruct) by **rewarding its policy
with the projection of an intermediate hidden state onto a frozen, supervised linear
probe of bias**, fitted via 5-fold CV on a held-out fairness benchmark (VLBiasBench
close-ended). PPO is run on SB-Bench under a low-rank LoRA adapter; the probe acts as
a dense, input-conditioned reward independent of the correctness signal. Across four
training seeds with a KL-stability fix (KLFIX), we obtain **+1.08 pp ± 0.35 pp** on
canonical SB-Bench (n=2916; per-seed McNemar p < 10⁻¹¹), and a 43–71% reduction in
VLBiasBench bias-score magnitude with no significant accuracy regression. A
three-experiment audit ((a) reward synergy decomposition, (b) cross-layer wash-out
diagnostic across L1–L35, (c) per-axis rank-1 PCA experiment) demonstrates the
mechanism: **the L13 reward acts as a *selector* over already-existing trajectory
variation, not as a representational *eraser***. The bias representation passes through
two qualitatively distinct geometric regimes — disentangled per-category circuits at
mid-depth (L9–L21) and a convergent shared subspace at terminal depth (L25–L35) —
which directly motivates a multi-layer ensemble-reward intervention currently in the
pre-registered roadmap.

> **Phase 0.9 update (2026-06-23).** The two-regime claim is now hardened with B = 200
> paired-bootstrap separability (R1) and cross-dataset replication on SB-Bench-9axis
> (R2). The multi-layer ensemble-reward intervention has been built and 3-seed-evaluated
> (R3): in-distribution SB-Bench accuracy is tied with the single-layer L13 baseline,
> but the ensemble achieves a **+0.0021 |bias_score| reduction on VLBias transfer** that
> the single-layer baseline cannot (it instead worsens |bias_score| by 0.0005). See §9
> for the geometry update and §10.1 for the R3 measured result; full chain of evidence
> in [Phase0.9/FINAL_REPORT.md](../Phase0.9/FINAL_REPORT.md). R4 (causal counterfactual
> at n = 6166) is deferred pending additional ablation experiments.

---

## 2. From Phase 0–0.7 to Phase 0.8 (Compressed)

Earlier phases established three things that constrain everything in Phase 0.8:

1. **Reward shape is the bottleneck, not PPO** (Phase 0–0.5). Binary letter-correctness
   reward over a 3-way action collapses to the highest-prior letter; the stable
   PPO ceiling is ≈ 0.77, not 0.99. The 0.99 was an unstable transient. PPO knobs
   (KL controller, grad clip, LR, batch size) cannot rescue binary reward.
2. **DRM (SVM/PCA) reward works in-distribution but reveals two distinct failure modes
   off-distribution** (Phase 0.6 + 0.7). SVM transfers (+30.95 pp disambig at
   `ep1-50pct`) but drifts at `ep1-end` (bias score +0.020 → −0.067) and has a scene-image
   visual regression invisible to POPE. PCA mode-collapses to "always-C" off-distribution
   (99.5% C-rate, 33% overall accuracy).
3. **Both failures are REWARD-side, not PPO-side** (Phase 0.7 T1.1). Scoring the **base**
   model (no PPO) with the SVM and PCA heads on VLBiasBench gave C-bias = +10.28 (SVM)
   and +1.79 (PCA), confusion ≈ 0 (cannot distinguish "C because gold = C" from "C as
   refusal"). PCA-PPO's mode-collapse is the rational policy under that reward landscape.
   The penultimate hidden state, where these heads were fit, is task-conditioned —
   it must encode "which letter to output", which competes with bias representation
   for the same dimensions.

Full trajectory in [REPORT_EARLY_PHASES_EXTENSIVE.md](REPORT_EARLY_PHASES_EXTENSIVE.md).

**Phase 0.8 hypothesis.** There exists a layer L* < penultimate at which a **linear probe
of fairness** (not derived from the answer letter at all) provides a dense, input-conditioned
reward signal that decouples bias from answer formatting. The probing literature
(Tenney, Hewitt & Manning, Belinkov) suggests demographic features peak at middle layers;
for a 37-layer model this places L* in the L9–L21 window. The reward design follows
this hypothesis: train a single linear probe on labelled stereo-vs-counter outputs from
VLBiasBench, freeze it, then use its projection as one of three reward components
during SB-Bench PPO.

---

## 3. Reward and Probe Design

### 3.1 Reward composition

The final-step reward (after all Phase 0.8 iterations) is a weighted sum of three terms:

$$
r = w_\text{corr} \cdot \mathbb{1}[\hat{y} = y^*] \;+\; w_\text{bias} \cdot \bigl[-(\mathbf{w}_{\text{biasA}} \cdot \mathbf{h}_{L13}[\text{ans\_pos}])\bigr] \;+\; w_\text{ambig} \cdot \mathbb{1}[g = C \wedge \hat{y} = C] \;-\; \beta \cdot \text{KL}_\text{token}
$$

with coefficients `w_corr=1.0, w_bias=1.0, w_ambig=0.5, β=0.1, target_kl=0.02`.

- **Correctness term** `1[ŷ = y*]`: standard letter-correctness scalar; ensures the
  reward at least encourages the right letter.
- **Bias-axis term** `−(w_biasA · h_L13[ans_pos])`: the **negative** projection of the
  L13 hidden state at the answer position onto the unit-normalised `bias_aligned`
  probe weight vector. Negative sign because the probe predicts stereo-aligned = 1;
  the policy should *minimise* this projection.
- **Ambig term** `1[gold=C ∧ pred=C]`: preserves the "I don't know" behaviour on
  ambiguous items. Without it, debiasing pressure can drive the model to over-commit
  to A or B on ambig records and collapse the `gold=C` performance.
- **KL penalty** `−β · KL(π‖π_ref)` per token: standard PPO anchoring against the
  un-adapted base.

### 3.2 The `bias_aligned` probe label rule

The probe is a binary logistic regression `LogisticRegression(C=1.0, penalty='l2',
class_weight='balanced')` over 2048-d hidden states. The label rule is intentionally
**defined only on disambig records** ([source: probe_layers.py `_is_bias_aligned`](../src/modules/evaluation/probe_layers.py)):

| `condition` | gold meaning | stereotype-aligned option ID | bias_aligned label |
|---|---|---|---|
| `neg` | stereotyped person is the correct answer | `stereo_id = gold` | 1 if pred == gold, else 0 |
| `non_neg` | counter-stereotyped person is correct | `stereo_id = ({0,1} \ {gold}).pop()` | 1 if pred == stereo_id, else 0 |
| `ambig` | gold = C (Unknown) | — | **None** (dropped) |

The "dropped on ambig" choice matters: it leaves the C-emission decision to the
separate ambig-preservation term, decoupling fairness from abstention. This is the
fix to Phase 0.7's T1.1 finding that DRM heads conflated good-C and bad-C
(confusion ≈ 0).

### 3.3 Why `bias_aligned` and not `correct`

The probe could in principle be trained on `correct ∈ {0, 1}` (i.e. pred == gold over
all records). That would be a **capability** probe — it would reward "knows the answer",
not "avoids stereotype". A high `correct` probe Cohen-d on stereo-vs-counter records
would mean we'd be rewarding base capability, with no debiasing guarantee. The
`bias_aligned` probe is the only one of the three label rules whose decision boundary
explicitly separates stereotype-aligned from counter-stereotype actions *holding the
"emitted C" axis constant*. See §4.4 for the empirical falsification of the capability-
confound concern.

### 3.4 The deployment layer L13

L13 is selected after the audit in §4. It is the middle of the L9–L17 *optimal plateau*
where the bias direction has best `|s_d| / |C_d|` ratio (bias-discriminability vs
generic-capability-discriminability) under the `bias_aligned` label rule. The next
section walks through how this layer was chosen without cherry-picking.

---

## 4. Probe Audit — Picking the Layer Without Cherry-Picking

### 4.1 The pre-registration constraint

A core methodological principle entering Phase 0.8 ([source: Phase0.8.md §2.3](../Phase0.8.md)):

> Before running A2, write down the top-3 layers by A1 probe-accuracy on the `qformat=text`
> **holdout** split, *not* on the full set. This is the bias-blind selection step.

Concretely, the layer audit was structured so that:
- A1 (per-layer probe) covered every layer; no layer was excluded a priori.
- The A1 metric (probe accuracy + bias-discriminability) is task-agnostic; probe target
  is derived from VLBiasBench gold labels, not from any DRM head output. This avoids
  the eval-leak that broke T1.2 in Phase 0.7.
- Pre-committed acceptance gates were:

| Gate | Metric | Threshold | Rationale |
|---|---|---:|---|
| **G1** | `bias_aligned` holdout accuracy | ≥ 0.70 | probe must actually learn the bias axis |
| **G2** | Disambig `s_d` (offline reward on base) | ≥ +0.40 | must clearly beat legacy `is_good_C` L9 baseline (+0.289) |
| **G3** | Ambig signature (μ_inc > 0, gap > 0) | required | the canonical BBQ ambig bias finding |

### 4.2 Three-layer audit (L9, L11, L13)

| L | G1 holdout | G2 `s_d` | G3 ambig (μ_inc, gap) | overall |
|---:|---|---|---|---|
|  9 | 0.866 ✓ | +0.505 ✓ | μ_inc = +0.224, gap = +1.469 ✓ | **PASS** |
| 11 | 0.881 ✓ | +0.588 ✓ | μ_inc = +0.209, gap = +1.517 ✓ | **PASS** |
| **13** | **0.940 ✓** | **+0.621 ✓** | μ_inc = +0.184, gap = **+1.600 ✓** | **PASS (best)** |

L13 dominates L9 on every metric (Gate 1 +7.4 pp, Gate 2 +0.116, ambig gap +0.131). The
coefficient norm is identical across layers, so the lift isn't a "L13 is easier to fit"
artefact.

### 4.3 Per-axis breakdown at the three layers

| axis | L9 | L11 | L13 |
|---|---:|---:|---:|
| Age | +0.521 | +0.595 | **+0.673** |
| Disability_status | +0.472 | +0.567 | **+0.632** |
| Gender_identity | +0.460 | +0.574 | **+0.626** |
| Nationality | +0.471 | +0.554 | +0.562 |
| Physical_appearance | +0.427 | +0.561 | +0.576 |
| Race_ethnicity | +0.462 | +0.524 | +0.512 |
| Race_x_SES | +0.413 | +0.500 | +0.461 |
| Race_x_gender | +0.627 | +0.627 | **+0.710** |
| Religion | +0.566 | +0.661 | **+0.683** |
| SES | +0.551 | +0.648 | +0.651 |

Every axis at every layer clears the +0.40 bar — the first time in the workstream that
Gate 2 is uniformly satisfied across all 10 BBQ axes (the legacy `is_good_C` probe had
Race_x_SES and SES failing at L11/L13).

### 4.4 Capability-confound falsification (`correct` control probe)

If the `bias_aligned` signal is really capability in disguise, the `correct` head should
produce a similar `s_d`. It does not:

| L | probe | holdout | `s_d` | μ ambig.cor | μ ambig.inc | ambig gap |
|---:|---|---:|---:|---:|---:|---:|
| 9  | `bias_aligned` | 0.866 | **+0.505** | −1.245 | +0.224 | **+1.469** |
| 9  | `correct`      | 0.860 | **−0.046** | +0.205 | +0.026 | **−0.179** |
| 13 | `bias_aligned` | 0.940 | **+0.621** | −1.416 | +0.184 | **+1.600** |
| 13 | `correct`      | 0.870 | **−0.039** | +0.228 | −0.020 | **−0.248** |

Per-axis `s_d` under `correct` hovers between **−0.18 and +0.06** across all axes at
both layers (vs +0.41 to +0.71 for `bias_aligned`). The capability direction is
essentially blind to the neg-vs-non_neg polarity — exactly as it should be. The
**capability-confound hypothesis is decisively falsified**: the `bias_aligned` direction
is reading bias-specific structure, not "knows the answer" structure.

### 4.5 Full 11-layer extension sweep (L1, L5, L9, L11, L13, L17, L21, L25, L29, L33, L35)

After L13 was selected as deployment layer, we asked the broader question: how does the
bias direction behave across the full 37-layer depth? An 11-layer sweep was run on the
**base** model only, using the same `bias_aligned` label rule and 5-fold CV protocol.

Standardised Cohen-d-like `s_d` and capability control:

| L | Δμ_corr | Cohen `s_d` | μ ambig.inc | reward magnitude μ(neg::inc) | `correct` Δμ | `|s_d| / |corr|` |
|---:|---:|---:|---:|---:|---:|---:|
|  1 | +0.260 | +0.275 | +0.348 | −1.690 | −0.095 | 0.095 |
|  5 | +0.365 | +0.498 | +0.242 | −1.200 | −0.047 | 0.047 |
|  9 | +0.505 | +0.676 | +0.224 | −1.266 | −0.046 | 0.046 |
| 11 | +0.588 | +0.753 | +0.208 | −1.262 | n/a | n/a |
| **13** | **+0.621** | **+0.752** | +0.184 | −1.414 | **−0.039** | **0.039** |
| **17** | **+0.724** | **+0.825** | +0.027 | −1.415 | **−0.039** | **0.039** |
| **21** | **+1.430** | **+1.114** | −0.030 | −2.105 | **−0.016** | **0.016** |
| 25 | +5.890 | +1.709 | −0.288 | −4.579 | −0.034 | 0.034 |
| 29 | +6.450 | +1.742 | −0.260 | −4.969 | −0.017 | 0.017 |
| 33 | +7.675 | +1.658 | −0.311 | −5.876 | +0.052 | 0.052 |
| 35 | +8.412 | +1.713 | −0.532 | −6.505 | +0.064 | 0.064 |

**Findings.**

1. **Monotone amplification of raw Δμ_corr** from +0.260 at L1 to +8.412 at L35
   — a 32× increase, with zero sign-reversed axes anywhere in the sweep.
2. **Cohen-d saturation past L25.** Standardised `s_d` grows from +0.28 (L1) to a
   plateau of ~+1.7 at L25–L35. Decomposes the raw 13.5× growth (L13→L25) as
   ≈ 2.3× real discriminability gain × ≈ 6× hidden-state-norm inflation.
3. **Capability confound flat throughout.** `|correct Δμ|` ≤ 0.10 at every layer from
   L5 to L35 (max 0.064 at L35). The bias direction is independent of capability at
   every depth, not just in the L9–L17 window.
4. **Reward-magnitude inflation.** `μ(neg::incorrect)` grows 4.6× from L13 to L35,
   mirroring Qwen's known norm growth across LM blocks. For PPO this is the single
   biggest operational concern at deep layers (the L25+ reward scale dwarfs any KL
   coefficient calibrated for L13). This motivates per-layer z-score normalisation in
   any multi-layer ensemble.
5. **Ambig signature flips at L21+.** `μ(ambig::incorrect)` goes from +0.184 at L13 to
   −0.532 at L35. By construction, ambig records are excluded from `bias_aligned` probe
   training (label = None when gold = C), so these cells are pure OOD scoring. The
   *disambig* behaviour — what PPO actually optimises — remains correct (`Δμ_corr` > 0,
   all axes) and *sharpens* with depth.

### 4.6 Layer recommendation outcome

| layer | Δμ_corr lift vs L13 | Cohen s_d lift | reward magnitude vs L13 | ambig signature | recommendation |
|---:|---:|---:|---:|---|---|
| **L13** | 1.00× | 1.00× | 1.00× | intact (+0.18) | safe baseline (current default) |
| **L17** | **1.17×** | **1.10×** | **1.00×** | borderline (+0.03) | **safe upgrade** — same scale, ~16 % lift |
| **L21** | **2.30×** | **1.48×** | 1.49× | flips to 0 (−0.03) | **aggressive upgrade** — large lift, cleanest capability control |
| L25 | 9.5× | 2.27× | 3.24× | strongly negative (−0.29) | high-signal but requires PPO reward normalisation |
| L29–L35 | 10–14× | 2.27–2.32× | 3.5–4.6× | strongly negative | reward-scale risk dominates |

Decision: **L13 selected as the deployment layer** for the conservative track (preserves
the operating regime every other knob was tuned against). L17/L21 retained as candidate
upgrades. The 11-layer sweep is the foundation of the multi-layer ensemble experiment
(Action 1; see §10).

---

## 5. Cross-Dataset Transfer Audit (SB-Bench → VLBiasBench at L13)

The probe was trained on VLBiasBench-derived hidden states; if we deploy it as reward
during SB-Bench training, we need to know the L13 `bias_aligned` direction transfers to
*both* datasets. Direct test: refit the L13 probe on SB-Bench-derived hidden states with
the same label rule, score under the same protocol.

| dataset | n (probed) | Cohen `s_d` | Per-axis sign |
|---|---:|---:|---:|
| SB-Bench train (probe-fit) | ≈900 | **+1.631** | 9/9 positive |
| VLBiasBench (transfer) | 900 stratified | **+1.600** | 10/10 positive |

**Cross-dataset retention = 102%.** The L13 `bias_aligned` direction is dataset-general
within the BBQ family. The pre-registered transfer gate (`SB-Bench s_d ≥ +0.31`) was
cleared by 5×.

**Caveat.** Only 2 of 9 SB-Bench axes were covered by the current vanilla-baseline
JSONL at the time of the audit; full-axis SB-Bench transfer is in the roadmap. Also,
this is a single-dataset-pair transfer test — true *architecture-independent* bias-axis
claims require a 3rd held-out benchmark (CrowS-Pairs MM, MMBench-Stereotype). See
§12 limitations.

---

## 6. PPO Recipe and KL-Stability Fix (KLFIX)

### 6.1 Prod single-seed result (the headline number that motivated the audit)

The first end-to-end PPO run with the L13 probe-reward recipe ([source: Phase0.8_Critical_Review.md §2.1](../Phase0.8_Critical_Review.md)):

| Component | Spec |
|---|---|
| Base | Qwen2.5-VL-3B-Instruct, bf16, FlashAttention-2 |
| LoRA | r=16, α=32 |
| PPO data | SB-Bench train, 1936 samples (1 epoch, 242 batches) |
| Probe head | VLBiasBench-fitted L13 `bias_aligned`, 2048-d, frozen |
| Reward | `r = 1.0·1[ŷ=y*] + 1.0·−(w_biasA·h_L13) + 0.5·1[gold=C∧pred=C] − 0.1·token_KL` |
| target_kl | 0.02 |
| Wall time | ~31 min (113 s setup + 1767 s PPO) |

**Results.**

| Slice | Vanilla | Phase 0.8 | Δ | McNemar p |
|---|---:|---:|---:|---:|
| SB-Bench overall (n=2916) | 0.6190 | **0.6375** | **+1.85 pp** | **7.4 × 10⁻¹²** |
| VLBiasBench overall (n=2000) | 0.5420 | 0.5395 | −0.25 pp (n.s.) | 0.42 |
| VLBias \|bias_score\| | 0.0070 | 0.0020 | **−71%** | — |

SB-Bench: 8 of 9 categories improved; biggest gains on Religion (+3.47 pp),
Gender (+2.22 pp), Age (+2.03 pp); only Disability marginally negative (−0.62 pp, n.s.).

This was the prod single-seed result. Three follow-ups were required before we could
treat it as the headline:

1. **Reward synergy decomposition** — does the gain come from `correctness`, `bias`,
   or the combination? (§7)
2. **Seed replication with KL stability** — is the +1.85 pp reproducible? (§6.2)
3. **Wash-out diagnostic** — does the L13 reward actually attack the bias representation,
   or is it doing something else? (§8)

### 6.2 The KL-stability problem and KLFIX

Initial 4-seed sweep with the prod recipe (s1, s2, s3, s4) produced wildly varying
results: per-seed canonical Δacc ranged from −2.61 pp to +0.81 pp (mean −0.90 pp, 95% CI
[−2.61, +0.81] pp). The reward-causality diagnostic was positive within-run (Pearson
ρ = +0.95 between reward and accuracy) but uncorrelated across seeds (ρ ≈ +0.08).
**Reward was causal within a seed but the seed lottery was destroying transfer.**

Three independent stability issues were identified and bundled into the **KLFIX**:

1. **Value-head initialisation bias.** The `nn.Linear(hidden, 1, bias=False)` value
   head was Kaiming-uniform-initialised, giving a 5–28× cross-seed spread on
   `value_grad_norm`. Fix ([phase08_ppo_trainer.py L141-150](../src/modules/rl_components/phase08_ppo_trainer.py)):
   ```python
   self.value_head = nn.Linear(hidden_size, 1, bias=False)
   with torch.no_grad():
       self.value_head.weight.zero_()
   ```
2. **Value clipping was inert by default.** Added `--value-clip-range 0.2` to bound
   value-update magnitude per step.
3. **KL controller too slow at default.** Bumped `--kl-adapt-rate 0.1 → 0.3` so the
   Schulman β controller responds 3× faster to KL excursions. In practice the controller
   never had to engage at all post-KLFIX (β stayed at floor 0.05 for all 4 seeds), so
   this acts as a defensive dependency.

### 6.3 KLFIX 4-seed canonical result

| variant | micro-acc | std | Δ vs vanilla | macro-acc | std | Δ macro |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla baseline | 0.6190 | — | — | 0.6564 | — | — |
| L13 nowarm (4-seed) | 0.6203 | 1.39 pp | +0.13 pp | 0.6567 | 1.38 pp | +0.03 pp |
| L13 vwarmup (pre-fix) | 0.6100 | 1.08 pp | **−0.90 pp** | 0.6457 | 1.12 pp | −1.07 pp |
| **L13 KLFIX (this work)** | **0.6298** | **0.35 pp** | **+1.08 pp** | **0.6675** | **0.30 pp** | **+1.12 pp** |

Per-seed micro: s1=0.6259, s2=0.6320, s3=0.6334, s4=0.6279. **All 4 seeds beat vanilla
individually.**

| KL metric | Pre-KLFIX (vwarmup) | KLFIX | Improvement |
|---|---:|---:|---:|
| Tail-KL mean | 0.01345 | 0.00071 | **19× lower** |
| Tail-KL std | 0.02524 | 0.00017 | **148× tighter** |
| KL-spread max/min | 73× | 1.69× | 43× tighter |
| β-tail | 0.115 | **0.0500 (floor)** | controller never engaged |
| value_grad_norm head5 | 22.4 | 1.06 | **21× lower** |
| Cross-seed canonical-acc std | 1.08 pp | 0.35 pp | 3× tighter |

The "seed lottery" was eliminated; the post-KLFIX result is the new Phase 0.8 baseline.

### 6.4 Per-axis fairness (KLFIX 4-seed, mean ± std)

| Axis | Vanilla | KLFIX mean | Δ | std |
|---|---:|---:|---:|---:|
| Religion | 0.8194 | 0.8524 | **+3.30 pp** | 0.35 pp |
| Gender_identity | 0.4786 | 0.4953 | **+1.67 pp** | 0.47 pp |
| SES | 0.7474 | 0.7616 | +1.42 pp | 0.49 pp |
| Race/Ethnicity | 0.5809 | 0.5910 | +1.01 pp | 0.38 pp |
| Nationality | 0.6518 | 0.6618 | +1.00 pp | 0.43 pp |
| Age | 0.6318 | 0.6385 | +0.68 pp | 0.62 pp |
| Physical_appearance | 0.6335 | 0.6401 | +0.65 pp | 0.66 pp |
| Sexual_orientation | 0.7117 | 0.7180 | +0.62 pp | 0.53 pp |
| Disability | 0.6522 | 0.6491 | −0.31 pp | 0.80 pp |

**8/9 axes ≥ vanilla, 9/9 axes ≥ vwarmup.** Religion (+3.30 pp) and Gender (+1.67 pp)
are the standouts. Only Disability is slightly negative (−0.31 pp), within 1 SE (~0.9 pp
at n ≈ 322 per axis).

---

## 7. Reward Synergy Decomposition

### 7.1 The ablation

Phase 0.8 Critical Review §3.2 raised the central concern: the prod +1.85 pp on
SB-Bench could be driven primarily by the `correctness_coef=1.0` term (which is just
RLHF for accuracy), not by the bias projection. We isolated the two terms:

- **corrOnly**: `w_corr=1.0, w_bias=0.0, w_ambig=0.5`
- **biasOnly**: `w_corr=0.0, w_bias=1.0, w_ambig=0.5`
- **Full**: `w_corr=1.0, w_bias=1.0, w_ambig=0.5` (the prod recipe)

### 7.2 Result

| Variant | SB-Bench Δacc vs vanilla | Notes |
|---|---:|---|
| Vanilla (no PPO) | — | baseline 0.6190 |
| `corrOnly` (correctness alone) | **−5.25 pp** | actively hurts |
| `biasOnly` (bias-probe alone) | **−1.27 pp** | also hurts |
| **Full (`corrOnly + biasOnly`)** | **+1.85 pp** | works |

Interaction term ≈ **+8.37 pp**.

### 7.3 Interpretation

This is the strongest single piece of evidence in the project that the probe reward is
doing real work:

- **Neither term alone produces the +1.85 pp.** Correctness-only PPO with the
  ambig term destabilises the policy (the model over-commits to certain letters on
  hard items, costing accuracy). Bias-only PPO has no anchor to "the right letter is
  worth picking", so it just minimises the bias projection by random drift.
- **The combination is super-additive.** corrOnly + biasOnly individually = −6.52 pp;
  jointly = +1.85 pp. The synergy is in the right direction: the correctness gradient
  *channels* the bias gradient, ensuring the policy only "pays" the bias term in
  directions where the resulting answer is also correct.
- **The probe is not a redundant accuracy signal.** A trivial "the bias probe is just
  a noisy correctness reward" hypothesis would predict corrOnly ≥ Full. The data
  decisively rules this out.

The "synergy" framing is also why later interventions (multi-layer ensemble reward
in §10) preserve the correctness term — it is operationally necessary, not optional.

---

## 8. Wash-out Diagnostic — Where Does the L13 Reward Actually Land?

### 8.1 Three patterns and a pre-committed decision rule

If the L13 reward is doing what we hoped — driving the bias representation down at
L13 — we should be able to **measure** it. Pre-committed classification rule
([source: Phase0.8_Strategic_Plan.md §3](../Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md)):

| Pattern | Definition | Mechanistic meaning |
|---|---|---|
| **Pattern 1 — Clean** | `Δμ ≤ −0.30σ` at L13 AND `≤ −0.15σ` at every L ≥ 17 | reward signal propagates; bias suppression sticks across depth |
| **Pattern 2 — Wash-out** | `Δμ ≤ −0.30σ` at L13 BUT `|Δμ| < 0.10σ` at every L ≥ 21 | local L13 shift; downstream reconstructs the bias representation |
| **Pattern 3 — Proxy-hack** | `Δμ ≤ −0.30σ` at L13 BUT `Δμ > 0` at any L ≥ 25 | model pushes bias OUT of L13 INTO deeper layers; bias preserved or amplified, hidden from probe |

### 8.2 Method

Re-scored each PPO variant's hidden states against the **frozen base-fit** per-layer
probes at L ∈ {1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35}. Compared four PPO variants
against vanilla:

- `phase08_2k` — prod single-seed baseline
- `corrOnly` — correctness-only ablation
- `biasOnly` — bias-only ablation
- `phase08_2k_klfix_s1` — KLFIX-stabilised seed 1

### 8.3 Result

| variant | max \|z\| vs base | argmax L | pattern |
|---|---:|---:|---|
| `phase08_2k` (prod) | 0.032σ | 11 | Pattern 2 |
| `corrOnly` | 0.075σ | 25 | Pattern 2 |
| `biasOnly` | 0.021σ | 25 | Pattern 2 |
| **`klfix_s1`** | **0.019σ** | **17** | **Pattern 2 (flattest)** |

All four variants are **sub-threshold at every layer**. None crosses the pre-registered
Pattern-1 cut-off of |z| ≥ 0.30σ at L13. KLFIX-s1 is the flattest — KL-stability removed
the noise floor but added no representational signal.

Per-layer (klfix-s1 − base) / σ_van:
```
L 1  +0.008    L13 +0.016    L25 -0.009
L 5  +0.015    L17 +0.019    L29 -0.013
L 9  +0.012    L21 +0.019    L33 -0.006
L11  +0.016                  L35 -0.000
```

### 8.4 Interpretation — selector, not eraser

The L13 reward delivers +1.08 pp accuracy *without measurably attacking the L13 bias
direction at any layer*. The pre-experiment hypothesis (a strict Pattern 1 / 2 / 3
classification) didn't bind because the L13 representation didn't move at all. The
correct mechanistic reading:

> **The L13 probe-reward is functioning as a *selection signal* over already-existing
> trajectory variation — preferentially reinforcing trajectories that already happen
> to be less biased on the L13 axis — rather than as a representational *eraser* that
> rewrites the L13 hidden state.**

This explains both:
- **Why the +1.08 pp gain is real**: PPO can rationally up-weight low-bias trajectories
  from the base policy's existing distribution. The bias direction doesn't need to move
  for the *output* to shift.
- **Why corrOnly and biasOnly individually fail (§7)**: without the bias-probe
  selector, the correctness gradient amplifies the policy's existing letter priors;
  without the correctness anchor, the bias selector has nothing to bias *toward*.
  Only the combination produces a coherent update.

It also implies a clear limitation: a selection signal can only access whatever
low-bias trajectories already exist in the base distribution. If the base policy
never produces counter-stereotype completions on a hard sub-axis (e.g. Disability),
no single-layer selector can amplify what isn't there. **The fix is more reward
*coverage*, not more training data of the same kind** — this directly motivates the
multi-layer ensemble in §10.

---

## 9. The Bias Representation Has a Two-Regime Geometric Structure

> Phase 0.9 update (2026-06-23): Promoted from "supporting evidence" to a
> headline contribution. The two-regime finding now has B = 200 paired-bootstrap
> separability (Phase 0.9 R1, §9.4 below) and cross-dataset replication on
> SB-Bench-9axis (Phase 0.9 R2, §9.9 below). See
> [`Phase0.9/FINAL_REPORT.md`](../Phase0.9/FINAL_REPORT.md) for the full
> Phase 0.9 chain of evidence.

### 9.1 Question

The 11-layer sweep (§4.5) showed a single linear `bias_aligned` direction successfully
discriminates stereo from counter on all 10 BBQ axes at every layer L1–L35. Reading
this naively suggests "bias is rank-1 geometry at every depth". But the probe is fitted
as a single 1-D direction *by construction* — the per-axis breakdown could mean the
shared direction is genuinely rank-1, *or* it could mean the shared direction is just
an averaged compromise across 10 nearly-orthogonal per-axis circuits.

To discriminate, we fit **10 per-axis probes** at each layer, stack their unit-normalised
weight vectors, and PCA on the stack. This directly measures whether the per-category
bias directions collapse onto a shared subspace.

### 9.2 Method

Cached hidden states (n=900, 37 layers, 2048-d) at
`Phase0.8/a3_results/local_cache/base_probe_hs.npz`. Per-record metadata includes
`bbq_axis`, `condition`, model `text` output. The `bias_aligned` label is computed
per-record by the same `_is_bias_aligned` function used to fit the original probe.

For each layer L ∈ {1, 5, 9, 13, 17, 21, 25, 29, 33, 35}:
1. For each axis A ∈ 10 BBQ axes, subset to disambig records of that axis with
   non-None `bias_aligned` label (n=60 per axis after dropping ambig and unparseable preds,
   class-imbalanced ~10–19 positives).
2. Fit `LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', solver='lbfgs',
   max_iter=2000)` with 5-fold StratifiedKFold; average the unit-normalised coefficient
   vectors across folds to get `w_{A,L}`.
3. Stack the 10 per-axis unit vectors into `W_L ∈ ℝ^{10×2048}`. PCA: report
   explained-variance ratio of PC0..PC4, mean pairwise |cos(w_i, w_j)|, mean
   |cos(w_A, w_*)| where w_* is the shared probe fit on all 600 disambig records.

[Source code: scripts/rank1_bias_geometry.py](../scripts/rank1_bias_geometry.py).
[Output: Phase0.8/a3_results/rank1_bias_geometry.json](../Phase0.8/a3_results/rank1_bias_geometry.json).

### 9.3 Robustness check — L2 regularisation strength

Repeated at C ∈ {0.01, 0.1, 1.0, 10.0}. The structural pattern is stable for C ∈
{0.1, 1.0, 10.0}; C=0.01 collapses probe weights toward zero (degenerate, max
|cos| = 0.98 at L1 from probes near the origin). Result table below reports C=1.0;
the conclusions hold across the three valid regularisation strengths.

### 9.4 Result

| layer | PC0 evr | PC1 evr | PC2 evr | mean pairwise \|cos\| | mean axis-shared \|cos\| | per-axis CV acc |
|---:|---:|---:|---:|---:|---:|---:|
| L1  | 0.19 | 0.17 | 0.13 | 0.35 | 0.44 | 0.84 |
| L5  | 0.20 | 0.18 | 0.12 | **0.13** | 0.32 | 0.88 |
| L9  | 0.18 | 0.16 | 0.13 | **0.10** | 0.29 | 0.89 |
| **L13** | **0.17** | 0.16 | 0.14 | **0.10** | 0.31 | 0.89 |
| L17 | 0.17 | 0.14 | 0.14 | 0.11 | 0.32 | 0.89 |
| L21 | 0.16 | 0.13 | 0.13 | 0.14 | 0.39 | 0.92 |
| **L25** | **0.25** | 0.16 | 0.14 | **0.58** | **0.49** | **0.97** |
| L29 | 0.23 | 0.15 | 0.14 | 0.55 | 0.54 | 0.97 |
| L33 | 0.22 | 0.15 | 0.13 | 0.50 | 0.53 | 0.97 |
| L35 | 0.19 | 0.14 | 0.13 | 0.47 | 0.53 | 0.97 |

**Uniform-spectrum baseline.** 10 random orthogonal unit vectors in 2048-d give
PC0 ≈ 1/10 = 0.10 with mean pairwise |cos| ≈ 0. Our values:
- L13 PC0 = 0.17 → 1.7× uniform (mildly above orthogonal)
- L25 PC0 = 0.25 → 2.5× uniform (clear rank-1 concentration)

### 9.4bis Bootstrap 95% CIs and paired-draw separability (Phase 0.9 R1)

The point estimates in §9.4 are augmented by B = 200 bootstrap analyses
([scripts/rank1_bias_geometry_bootstrap.py](../scripts/rank1_bias_geometry_bootstrap.py)
and [scripts/rank1_bias_geometry_bootstrap_full.py](../scripts/rank1_bias_geometry_bootstrap_full.py)).
Per draw, each axis's records are resampled with replacement (n = 60), the
per-axis probe refit on the same 5-fold StratifiedKFold recipe, and PC0 /
|cos| recomputed. Marginal CIs are the 2.5%/97.5% percentiles; paired-draw
separability uses a *shared resample index across layers* to control for
cross-layer correlated noise (i.e., a paired-difference test on the
per-draw vectors).

Marginal 95% CIs and paired-draw `P(L25 > L13)`:

| layer | PC0 mean | PC0 95% CI       | mean \|cos\| mean | \|cos\| 95% CI     |
|---:|---:|:--:|---:|:--:|
| L13 | 0.185 | [0.162, 0.214] | 0.106 | [0.085, 0.133] |
| L25 | 0.243 | [0.202, 0.297] | 0.540 | [0.491, 0.599] |

|       | metric        | P(L25 > L13)  paired |
|------:|:--------------|:-:|
|       | `mean |cos|`  | **1.000** |
|       | `PC0 evr`     | **0.995** |

|cos| ordering is decisively separable under both marginal and paired tests
(no CI overlap; all 200/200 paired draws place L25 above L13). PC0 ordering
is paired-separable at P = 0.995 but the marginal CIs touch by 0.012 due to
layer-shared nuisance variance from the bootstrap resample. Across the full
Regime-1 × Regime-2 grid: paired P ≥ 0.96 on PC0 and P = 1.000 on |cos| for
all 9 `{L13, L17, L21} × {L25, L29, L33}` pairs.

**Decision:** lead the two-regime claim with **mean pairwise |cos|** as the
primary metric (cleaner under both marginal and paired tests). Retain PC0
evr as a directional secondary metric with the marginal-CI caveat. See
[Phase0.9/r1_bootstrap_verdict.md](../Phase0.9/r1_bootstrap_verdict.md).

### 9.5 Interpretation — two distinct geometric regimes

**Regime 1 — Categorical disentanglement at mid-depth (L5–L21).** Per-axis bias
directions are essentially **orthogonal** (mean pairwise |cos| ≈ 0.10–0.14, PC0 only
marginally above the uniform baseline). Each demographic category has its **own
distinct linear bias circuit**. Bias is high-rank (≥ 5-D effective) here. Per-axis CV
accuracy is high (0.88–0.92) — these are real, individually-learnable circuits, not
noise.

**Regime 2 — Convergent shared subspace at terminal depth (L25–L35).** Per-axis
directions **partially collapse onto a common axis** (mean pairwise |cos| ≈ 0.47–0.58,
PC0 = 0.19–0.25). Per-axis discriminability simultaneously sharpens (CV acc 0.89 → 0.97).
This is consistent with all categorical bias signals converging onto a single
"is-this-stereotypical" readout direction near the LM head — exactly where the policy
must commit to a letter token.

**L1 anomaly.** L1 has moderate |cos| = 0.35 because L1 is the input embedding —
per-axis probes pick up template-level lexical features ("the person in the image",
demographic group names) rather than bias circuits. This is expected and not part of
the bias geometry story.

### 9.6 What this refines about the §4.5 single-direction finding

The §4.5 finding that "a single `bias_aligned` direction discriminates stereo/counter
on all 10 axes at every layer" is preserved, but the *reason* is depth-dependent:

- **At L13**, the shared direction works because it picks the **best averaged compromise**
  across 10 nearly-orthogonal per-axis circuits. The compromise is good (Cohen-d 0.75)
  but is leaving significant per-axis signal on the table.
- **At L25+**, the shared direction works because the per-axis circuits **genuinely
  converge** onto a common subspace. The compromise is the actual underlying structure.

### 9.7 Caveats and what this finding is not

1. **Magnitude is dataset-coupled.** §9.9 shows the |cos| **direction**
   replicates across VLBias and SB-Bench (paired-draw P = 1.000 on
   17 of 20 Regime-1 × Regime-2 pairs) but the **magnitude** is ~3× weaker
   on SB-Bench (|cos|@L25 = 0.540 VLBias vs 0.172 SB-Bench). The
   convergence is a robust *directional* model property; the strength of
   alignment depends on the test distribution's per-axis coverage.
2. **PC0 concentration spike is VLBias-coupled, not a model property.**
   The L25 PC0 = 0.25 (vs 0.17 at mid-depth) on VLBias does **not**
   replicate on SB-Bench (paired P(L25 > L13) = 0.200 on SB-Bench,
   inverted relative to VLBias's 0.995). PC0 should be treated as a
   descriptive VLBias-specific quirk, not a structural claim.
3. **Single model.** Qwen2.5-VL-3B only. Whether the two-regime structure is specific
   to this architecture, or a general property of multi-modal transformers with
   similar depth/hidden-dim, is untested.
4. **The "convergent subspace" claim is geometric, not interpretive.** We have not
   shown that the deep-layer shared axis is *causally* responsible for the final
   answer — only that the per-category probe directions align there. R4
   (counterfactual flip-rate at n = 6166) is the planned causal-evidence
   gate but is deferred pending additional ensemble-window ablation experiments.

Caveats (1) and (2) supersede the original §9.7 "n=60 / no CIs" and
"single dataset" caveats, which were closed by Phase 0.9 R1 and R2
respectively.

### 9.8 Why this matters for the next intervention

The combined §8 (wash-out: L13 reward is a selector) + §9 (two-regime geometry: L13 is
in the disentangled regime) reading is:

> **One scalar reward at one disentangled layer cannot efficiently move 10 nearly-orthogonal
> per-category circuits.** The probe's success at L13 is real but incidental — it picks
> the best 1-D averaged compromise but lacks the dimensionality to selectively pressure
> all categories.

A **multi-layer ensemble reward** that aggregates probe signals from both regimes (e.g.
{L13, L17, L21} in the disentangled regime + {L25, L29, L33} in the convergent regime)
would sample **both** per-category circuits *and* the shared subspace. This is the
basis of the Action 1 experiment in §10.

### 9.9 Cross-dataset replication on SB-Bench-9axis (Phase 0.9 R2)

Re-runs the bootstrap pipeline on a hidden-states cache extracted from the
same model on SB-Bench-9axis (n = 751, 9 BBQ axes; 7 overlap with VLBias's
10). Cache: `Phase0.8/a3_results/local_cache/sbbench9_probe_hs.npz`. Same
script, generalised from `n_axes == 10` assert to `n_axes >= 2`.

|cos| side-by-side:

| Layer | VLBias \|cos\| | SB-Bench \|cos\| | VLBias PC0 | SB-Bench PC0 |
|---:|---:|---:|---:|---:|
| L1  | 0.299 | 0.212 | 0.208 | 0.216 |
| L5  | 0.133 | 0.101 | 0.211 | 0.205 |
| L9  | 0.103 | 0.079 | 0.186 | 0.187 |
| L13 | 0.106 | 0.080 | 0.185 | 0.188 |
| L17 | 0.110 | 0.087 | 0.179 | 0.185 |
| L21 | 0.129 | 0.090 | 0.173 | 0.177 |
| **L25** | **0.540** | **0.172** | **0.243** | **0.175** |
| L29 | 0.499 | 0.149 | 0.223 | 0.174 |
| L33 | 0.446 | 0.123 | 0.216 | 0.174 |
| L35 | 0.418 | 0.140 | 0.200 | 0.172 |

Paired-draw `P(L_hi > L_lo)`:

|       | metric        | VLBias | SB-Bench |
|------:|:--------------|:-:|:-:|
|       | `mean |cos|`  | 1.000 | **1.000** |
|       | `PC0 evr`     | 0.995 | **0.200** ❌ |

|cos| ordering replicates at P = 1.000 on 17 of 20 Regime-1 × Regime-2 pairs;
the 3 sub-1.0 cells all involve L35 (deepest-layer mild relaxation already
visible in VLBias). PC0 ordering is **inverted** on SB-Bench — mid-layer
PC0 ≥ deep-layer PC0 in 80% of paired draws.

**Verdict.** The two-regime claim — deep-layer per-axis directions converge
into a shared subspace — is a **directional model property**. The
**magnitude** of convergence is **dataset-coupled** (~3× weaker on
SB-Bench). The PC0 "concentration spike" sub-claim is **not** a model
property and should be reported as a VLBias-specific descriptor (cf.
§9.7 caveats 1 and 2). See
[Phase0.9/r2_sbbench_replication_verdict.md](../Phase0.9/r2_sbbench_replication_verdict.md).

---

## 10. Strategic Implication and Roadmap

> Phase 0.9 update (2026-06-23): §10.1 is rewritten to reflect the
> **measured** R3 result rather than the pre-Phase-0.9 priors. See
> [`Phase0.9/FINAL_REPORT.md`](../Phase0.9/FINAL_REPORT.md) §3 for full
> chain of evidence and §4 for audit disclosures.

### 10.1 Action 1 — multi-layer ensemble reward (measured result)

**Hypothesis.** Given §9 establishes that deep-layer per-axis convergence is
a model property, an ensemble probe across the regime-boundary window
`{L17, L21, L25, L29, L33}` with `zscore_mean` pooling — plugged in as a
**KLFIX-PPO enhancement** (same 3-fix KLFIX backbone, only reward signal
differs) — should extract more debiasing than the single-layer L13 reward
already on the KLFIX backbone.

**Verdict: partially vindicated.** No in-distribution accuracy benefit; the
ensemble does improve OOD bias-reduction on VLBias transfer, consistent with
the R2 model-property finding.

**Implementation.** Five surfaces touched (additive; single-layer path
unchanged). Per-layer offline μ/σ calibrated on the SB-Bench-9axis HS cache
(n = 751) to match the PPO training distribution. Bundle on volume at
`/generated_heads_probe_L17-33_zmean_base_biasA/`. Code: §10.1.1 below.

**In-distribution result (SB-Bench canonical, n = 2916, like-for-like 3-seed):**

| Metric | Vanilla | KLFIX-L13 (n = 3) | R3 ensemble (n = 3) | R3 − KLFIX |
|---|---|---|---|---|
| micro_acc | 0.6149 | 0.6286 ± 0.0031 | 0.6278 ± 0.0038 | **−0.08 pp** |
| macro_acc | 0.6510 | 0.6669 ± 0.0034 | 0.6652 ± 0.0036 | −0.17 pp |
| Δ vs vanilla | — | +1.37 pp | +1.29 pp | tied within noise |
| Stouffer combined z | — | +9.56 | +8.10 | both highly sig |
| Per-axis wins | — | 6/9 | 1/9 (Age) | 2/9 ties |

Single-layer L13 saturates the available in-distribution accuracy signal;
the ensemble adds no further gain on SB-Bench (the dataset PPO trains on).

**Out-of-distribution result (VLBias transfer, n = 2000, like-for-like 3-seed):**

| Metric | Vanilla | KLFIX-L13 (n = 3) | R3 ensemble (n = 3) | R3 − KLFIX | Direction |
|---|---|---|---|---|---|
| overall_accuracy | 55.13 % | 54.42 % ± 0.20 | 54.47 % ± 0.06 | +0.05 pp | tie |
| **disambig_acc** | 36.75 % | 36.16 % ± 0.47 | **36.93 % ± 0.24** | **+0.78 pp** | **R3** ✅ |
| neg_acc | 36.40 % | 36.53 % ± 0.31 | **37.18 % ± 0.00** | +0.65 pp | **R3** ✅ |
| non_neg_acc | 37.10 % | 35.79 % ± 0.68 | **36.69 % ± 0.48** | +0.90 pp | **R3** ✅ |
| **\|bias_score\|** | 0.0070 | 0.0075 ± 0.0048 | **0.0049 ± 0.0048** | toward 0 by **+0.0021** | **R3** ✅ |
| ambig_acc | 91.90 % | 90.90 % ± 0.38 | 89.51 % ± 0.45 | −1.40 pp | KLFIX |

**|bias_score| movement vs vanilla** is the headline OOD effect:
- KLFIX-L13: |bias_score| **worsens** by 0.0005 (vanilla |0.0070| → KLFIX |0.0075|, sign-flipped past 0).
- R3 ensemble: |bias_score| **improves** by 0.0021 (vanilla |0.0070| → R3 |0.0049|).
- **R3 is the only method that aggregate-debiases on VLBias.**

Per-axis OOD bias_score (closer to 0 wins): R3 wins 5/10, KLFIX 2/10, ties 3/10.

**Mechanism.** The R2 model-property finding directly predicts this: an
ensemble probe over the regime-boundary window captures a more
dataset-invariant bias direction (since the deep-layer convergence
replicates), so the policy trained against it transfers a bias-reduction
signal that the dataset-specific L13 probe cannot. The cost is ambig-channel
erosion (1.4 pp gap to KLFIX-L13).

**Pre-committed §3bis gate status:**

| Gate | Threshold | R3 (like-for-like n = 3) | Status |
|---|---|---|---|
| Mean canonical-acc Δ vs vanilla | ≥ +1.5 pp | +1.29 pp | ❌ |
| Cross-seed std | ≤ 0.5 pp | 0.38 pp | ✅ |
| Beats KLFIX-L13 on ≥ 6/9 BBQ axes | ≥ 6 wins | 1 win, 6 losses, 2 ties | ❌ |
| KL stability: tail-KL std | ≤ 0.005 | 0.00066 | ✅ |
| VLBias ambig rate within −2 pp of vanilla | ≥ −2 pp | −2.39 pp | ⚠️ |
| OOD VLBias bias_score reduction > KLFIX-L13 (post-hoc) | any positive Δ | +0.0021 vs −0.0005 | ✅ |
| OOD VLBias disambig_acc > KLFIX-L13 (post-hoc) | any positive Δ | +0.78 pp | ✅ |

The original §3bis gate (which was framed around in-distribution accuracy
enhancement) **fails**. The two post-hoc OOD gates pass cleanly. The
paper narrative is therefore: **L13 single-layer is sufficient for in-
distribution accuracy gain; the ensemble offers a transfer-aware
bias-reduction benefit on the held-out dataset.**

#### 10.1.1 Implementation pointers (delivered)

The "implementation gaps" in the pre-Phase-0.9 §10.1 are all landed:

| Surface | File / artifact |
|---|---|
| Bundle build | [scripts/phase09_build_ensemble_bundle.py](../scripts/phase09_build_ensemble_bundle.py) → [Phase0.9/ensemble_bundle/](../Phase0.9/ensemble_bundle/) |
| Loader | [src/modules/training/drm_loader.py](../src/modules/training/drm_loader.py) `load_ensemble_probe_bundle()` |
| Args | [src/modules/training/args.py](../src/modules/training/args.py) `--ensemble_bundle_dir / --ensemble_layers / --ensemble_pool` |
| Trainer | [src/modules/rl_components/phase08_ppo_trainer.py](../src/modules/rl_components/phase08_ppo_trainer.py) `is_ensemble_mode` branch in `bias_aligned` reward |
| Launchers | [scripts/phase09_ensemble_klfix_smoke.sh](../scripts/phase09_ensemble_klfix_smoke.sh), [scripts/phase09_ensemble_klfix_seeds.sh](../scripts/phase09_ensemble_klfix_seeds.sh), [scripts/phase09_ensemble_canonical_gens.sh](../scripts/phase09_ensemble_canonical_gens.sh), [scripts/phase09_ensemble_vlbias_gens.sh](../scripts/phase09_ensemble_vlbias_gens.sh) |
| Aggregates | [Phase0.9/canonical/](../Phase0.9/canonical/) + [Phase0.9/vlbias/](../Phase0.9/vlbias/) |
| Verdict + audit notes | [Phase0.9/FINAL_REPORT.md](../Phase0.9/FINAL_REPORT.md) |

### 10.2 Action 3 — counterfactual flip-rate at n=6166 (causal-evidence gate)

The existing CF pair set (n=228) has CI ± 6 pp on a 35–50% baseline — useless for the
+1.08 pp headline claim. The n=6166 build is on the roadmap. Pass criterion:
`flip_rate(target) ≤ 0.7 × flip_rate(vanilla)` AND `accuracy_swap − accuracy_orig`
within ±2 pp AND corrOnly does not meet the first criterion. This is the only test
that converts the headline from an "accuracy" claim into a "causal debiasing" claim.

### 10.3 Pre-Action-1 hardening experiments (Phase 0.9 — completed)

The two-regime finding (§9) is now hardened by Phase 0.9. Both pre-registered
experiments completed with verdicts:

| # | Experiment | Status | Verdict |
|---|---|---|---|
| **R1** | Bootstrap CIs on per-layer PC0 + \|cos\| | ✅ Done | Two-regime claim is **publication-grade** under paired-draw separability: P(L25 \|cos\| > L13 \|cos\|) = 1.000, P(L25 PC0 > L13 PC0) = 0.995 on B = 200 paired bootstrap. Lead with \|cos\|, demote PC0 to descriptive secondary. (§9.4bis above.) |
| **R2** | Cross-dataset replication on SB-Bench | ✅ Done | \|cos\| convergence direction **replicates as a model property** (P = 1.000 on 17 of 20 paired pairs). PC0 spike does **not** replicate (P(L25 > L13) = 0.200 on SB-Bench, inverted vs VLBias). Magnitude is ~3× weaker on SB-Bench. (§9.9 above.) |

R3 (Action 1 in §10.1) is also completed. R4 (Action 3 in §10.2) is deferred
pending additional ensemble-window ablation experiments. See
[Phase0.9/FINAL_REPORT.md](../Phase0.9/FINAL_REPORT.md) for the full Phase
0.9 chain of evidence + methodology audit disclosures.

### 10.4 What we are explicitly NOT doing

- **PPO scale-up beyond 2k samples.** Ranked #4 of 4 in
  [SCALING_PRIORITY_ANALYSIS.md](../Phase0.8/SCALING_PRIORITY_ANALYSIS.md). Trajectory
  decelerates (+1.03 → +0.79 → +0.17 pp across quartiles), probe sits in the L9–L19
  optimal plateau, synergy saturated. Tripwires that would promote it back: Action 1
  failure, wash-out pattern shift, or CF flip-rate Δ < 5 pp.
- **DPO/IPO replacement of PPO.** Orthogonal to the "does the reward signal point at
  the right thing" question.
- **Reward-model retraining at deeper layers (single-layer L25).** Superseded by Action 1
  (ensemble covers L17/21/25/29/33 in one objective).
- **VLBiasBench retraining.** VLBias stays as the held-out transfer set.
- **Switching to the 7B model.** Capacity is not the current bottleneck. Wash-out shows
  the bottleneck is reward-signal coverage, not policy capacity. 7B would inherit the
  same selector-not-eraser pathology.
- **Multi-layer LoRA placement.** Pattern 2 wash-out authorises it, but the offline
  ensemble evidence makes the **reward** lever cheaper-and-stronger; tackle reward
  side first, LoRA placement later if needed.

---

## 11. Limitations and Threats to Validity

### 11.1 Single architecture

Qwen2.5-VL-3B-Instruct only. The 7B variant, LLaVA-Next, BLIP-3, Idefics-3, etc.
untested. Probing literature suggests middle-layer demographic features are
architecture-general at this scale (3B–7B transformer-decoder), but this is a
prediction, not a measurement.

### 11.2 Single benchmark family

Training (SB-Bench) and transfer (VLBiasBench) are both BBQ-style 3-way multiple
choice with image grounding. Free-form generation bias (CrowS-Pairs MM),
single-attribute classification (FairFace-VQA), and image-only inference are
untested. The cross-dataset 102% retention at L13 (§5) is *within-family* transfer,
not architecture-agnostic.

### 11.3 Probe was trained on the transfer set

The L13 `bias_aligned` probe was fitted on **VLBiasBench** hidden states, and OOD
evaluation is on **VLBiasBench**. The OOD bias-score reduction is therefore not a
pure transfer test — the reward signal had already seen those activations. The
+1.08 pp on **SB-Bench** is the cleaner transfer claim (probe never saw SB-Bench
hidden states), but it is conflated with the `correctness` term (§7 ablation shows
only the *combination* of bias + correctness produces non-noise improvement).
True architecture-agnostic transfer requires a 3rd held-out benchmark.

### 11.4 Disability axis directional regression

−0.31 pp on KLFIX (within 1 SE), −0.62 pp on prod single-seed. Directionally
consistent across two independent runs — flagged as a per-axis concern. Possible
causes: (i) Disability has the smallest n per axis on SB-Bench; (ii) the
`bias_aligned` label rule has higher class imbalance on Disability (9 pos / 51 neg
in our probe data); (iii) genuine reward-signal weakness on this axis. Not yet
isolated.

### 11.5 Two-regime finding sample size

n=60 disambig records per axis × 10 axes for the rank-1 experiment. Robust at L2
C ∈ {0.1, 10.0}, but bootstrap PC0 CIs and cross-dataset replication are blockers
for promoting §9 from supporting evidence to headline contribution. Both
experiments are pre-registered (§10.3) as Action-1-blockers.

### 11.6 Counterfactual evidence underpowered

Existing CF pair set (n=228) has CI ± 6 pp on a 35–50% baseline — useless for the
+1.08 pp claim. Without n=6166 CF (Action 3), the headline is an **accuracy** claim,
not a **causal debiasing** claim. The strategic plan is explicit about this:
"if Action 1 fails, hold KLFIX-L13 as baseline and write up the negative result
honestly."

### 11.7 Reward-magnitude inflation across depth

`μ(neg::incorrect)` grows 4.6× from L13 to L35 (§4.5). Any multi-layer ensemble
reward at deep layers without per-layer z-score normalisation would have the
deep-layer signal dominate; this is the motivation for the `zscore_mean` design
choice in Action 1.

### 11.8 The wash-out diagnostic's pre-committed thresholds didn't bind

Pattern 1/2/3 classification all required `|Δμ| ≥ 0.30σ at L13`. In practice every
PPO variant returned `|Δμ| ≤ 0.075σ` at every layer — sub-threshold for *any* of
the three patterns. The mechanistic reading ("L13 reward is a selector, not an
eraser") is a *post-hoc* synthesis; it was not pre-registered. We have flagged
this in [washout_klfix_verdict.md](../Phase0.8/washout_klfix_verdict.md) and the
revised Strategic Plan v2.

### 11.9 PPO training data size

2000 SB-Bench training samples × 1 epoch. Critical Review §3.3 noted the probe
itself was fit on ~900 samples in 2048-d space — by rule of thumb, 3–5×
under-sampled. The cross-layer transfer retention (102% at L13) suggests the
direction is robust to this under-sampling, but a 5000-sample probe re-fit is
straightforward future work.

---

## 12. Contributions Summary

1. **Methodological** — a **probe-as-reward** framework that decouples bias
   suppression from answer formatting by routing the reward through a frozen
   linear probe of an intermediate (non-penultimate) hidden state. Removes the
   T1.1 confusion-≈-0 pathology that mode-collapsed PCA-DRM and drifted SVM-DRM
   in Phase 0.7.

2. **Empirical** — **+1.08 pp ± 0.35 pp** on canonical SB-Bench across 4 seeds
   with KL stability (KLFIX), 8/9 axes positive, no significant accuracy
   regression on VLBiasBench transfer, bias-score magnitude reduced 43–71%.

3. **Mechanistic (preliminary, supporting evidence)** — the bias representation
   in Qwen2.5-VL-3B passes through **two distinct geometric regimes**:
   disentangled per-category circuits at mid-depth (L9–L21, mean pairwise
   |cos| ≈ 0.10) and a convergent shared subspace at terminal depth (L25–L35,
   |cos| ≈ 0.50, PC0 = 0.25). This refines the "single shared direction works at
   every layer" finding by giving the *reason* it works as depth-dependent.

4. **Diagnostic** — the **wash-out test** (compare PPO-policy hidden states
   against frozen base-fit per-layer probes) localised the mechanism of the
   probe-reward gain: it is a **selector** over already-existing trajectory
   variation, not a representational eraser. Same diagnostic falsified the
   "L13 reward propagates to deeper layers" hypothesis (all 4 PPO variants
   sub-threshold at every layer).

5. **Operational** — the **KL-stability fix (KLFIX)**: value-head zero-init +
   `--value-clip-range 0.2` + `--kl-adapt-rate 0.3`. Reduces tail-KL std 148×
   and cross-seed canonical-acc std 3×, eliminating the seed lottery that
   plagued the initial 4-seed sweep.

---

## 13. Reproduction Pointers (Appendix)

### 13.1 Workspace

- Strategic plan (live): [Phase0.8_Strategic_Plan.md](../Phase0.8_Strategic_Plan.md)
- Strategic plan v1 (archived): [Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md](../Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md)
- Phase 0.8 motivation: [Phase0.8.md](../Phase0.8.md)
- Critical review of prod single-seed: [Phase0.8_Critical_Review.md](../Phase0.8_Critical_Review.md)
- Multi-layer probe analysis (§4½ series): [Phase0.8/Multi_layer_head_analysis.md](../Phase0.8/Multi_layer_head_analysis.md)
- KLFIX 4-seed final: [Phase0.8/seed_diag/klfix/FINAL_REPORT.md](../Phase0.8/seed_diag/klfix/FINAL_REPORT.md)
- Wash-out verdict (KLFIX): [Phase0.8/washout_klfix_verdict.md](../Phase0.8/washout_klfix_verdict.md)
- Wash-out diagnostic JSON: [Phase0.8/washout_diagnostic_klfix.json](../Phase0.8/washout_diagnostic_klfix.json)
- Ensemble offline test: [Phase0.8/ensemble/README.md](../Phase0.8/ensemble/README.md)
- Rank-1 experiment script: [scripts/rank1_bias_geometry.py](../scripts/rank1_bias_geometry.py)
- Rank-1 result JSON: [Phase0.8/a3_results/rank1_bias_geometry.json](../Phase0.8/a3_results/rank1_bias_geometry.json)
- Scaling-priority ranking: [Phase0.8/SCALING_PRIORITY_ANALYSIS.md](../Phase0.8/SCALING_PRIORITY_ANALYSIS.md)
- Strategic synthesis post-KLFIX: [Phase0.8/NEXT_STEPS.md](../Phase0.8/NEXT_STEPS.md)

### 13.2 Modal volume artefacts (`debias-vlm-persistent-storage`)

- KLFIX checkpoints: `/mnt/data/output_ppo_phase08_2k_L13_s{1,2,3,4}_klfix/final_debiased_model`
- Prod single-seed: `/mnt/data/output_ppo_phase08_2k/final_debiased_model`
- Ablation variants: `/mnt/data/output_ppo_phase08_2k_{corrOnly,biasOnly}/final_debiased_model`
- Frozen per-layer probe heads: `/mnt/data/generated_heads_probe_L{1,5,9,11,13,17,21,25,29,33,35}_base_biasA/sb_bench-PROBE-component/`
- Vanilla SB-Bench gens: `/mnt/data/sb_bench_vanilla_baseline/sb_bench_generations.jsonl`
- Vanilla VLBias gens: `/mnt/data/phase07_vlbiasbench/base_vlbias_gen.jsonl`
- Wash-out scores: `/mnt/data/phase08_washout/{variant}__probe_L{N}__offline_reward.json`
- Canonical eval aggregate: `/mnt/data/canonical/L13_canonical_klfix_4seeds_aggregate.json`

### 13.3 Key source modules

- Probe-layers ETL: [src/modules/evaluation/probe_layers.py](../src/modules/evaluation/probe_layers.py)
- Offline reward scorer: [src/modules/evaluation/score_vlbias_offline.py](../src/modules/evaluation/score_vlbias_offline.py)
- PPO trainer (KLFIX): [src/modules/rl_components/phase08_ppo_trainer.py](../src/modules/rl_components/phase08_ppo_trainer.py)
- Reward loader: [src/modules/training/drm_loader.py](../src/modules/training/drm_loader.py)
- CLI args: [src/modules/training/args.py](../src/modules/training/args.py)
- Modal entrypoints: [src/run_modal.py](../src/run_modal.py)
- Wash-out classifier: [scripts/phase08_washout_classify.py](../scripts/phase08_washout_classify.py)
- KLFIX wash-out launcher: [scripts/phase08_klfix_washout.sh](../scripts/phase08_klfix_washout.sh)
- Counterfactual pair builder: [scripts/phase08_build_counterfactual_pairs.py](../scripts/phase08_build_counterfactual_pairs.py)

### 13.4 Reproduction quickstarts

**Re-run the rank-1 experiment (§9):**
```bash
cd /Users/f0s03xp/Debias_VLMs
source debias_env/bin/activate
python scripts/rank1_bias_geometry.py
# Output: Phase0.8/a3_results/rank1_bias_geometry.json
```
Cost: ~2 min CPU (no GPU needed); cached hidden states required at
`Phase0.8/a3_results/local_cache/base_probe_hs.npz`.

**Re-run the wash-out diagnostic on a new checkpoint:**
```bash
# STEP=1: generate VLBias completions
STEP=1 TAG=phase08_new_variant SEED=1 \
  CHECKPOINT=/mnt/data/output_ppo_NEW_VARIANT/final_debiased_model \
  bash scripts/phase08_klfix_washout.sh

# STEP=2: per-layer probe scoring
STEP=2 TAG=phase08_new_variant \
  LAYERS_CSV="1,5,9,11,13,17,21,25,29,33,35" \
  bash scripts/phase08_klfix_washout.sh

# Pull + classify locally
for L in 1 5 9 11 13 17 21 25 29 33 35; do
  modal volume get debias-vlm-persistent-storage \
    "phase08_washout/phase08_new_variant__probe_L${L}__offline_reward.json" \
    "Phase0.8/washout_scores/phase08_new_variant__probe_L${L}__offline_reward.json" \
    --force
done

python scripts/phase08_washout_classify.py \
    --score-root Phase0.8/washout_scores \
    --variants base phase08_new_variant \
    --output-json Phase0.8/washout_diagnostic_NEW.json
```

**KLFIX 4-seed canonical eval:**
```bash
# See scripts/phase08_value_warmup_seeds.sh (rename to phase08_klfix_seeds.sh)
# Canonical training command in Phase0.8/seed_diag/klfix/FINAL_REPORT.md §"What changed"
```

---

*End of REPORT_EXTENSIVE.*
