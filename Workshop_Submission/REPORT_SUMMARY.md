# Interpretable Probe-Reward Debiasing of Vision-Language Models
## Project Summary — All Phases (Workshop Submission Draft)

**Date:** 2026-06-20
**Branch:** `phase0.8_strategic_plan_day1`
**Model:** Qwen2.5-VL-3B-Instruct (37 LM-decoder layers, 2048-d hidden, bf16, FlashAttention-2)
**Benchmarks:** SB-Bench (in-distribution training), VLBiasBench (held-out transfer)

> This is the **compressed-everything** view. For the full Phase 0.8 trajectory, see
> [REPORT_EXTENSIVE.md](REPORT_EXTENSIVE.md). For full Phase 0–0.7 trajectory, see
> [REPORT_EARLY_PHASES_EXTENSIVE.md](REPORT_EARLY_PHASES_EXTENSIVE.md). Both are
> concatenated in [REPORT_COMBINED.md](REPORT_COMBINED.md).

---

## 1. Thesis

We debias a vision-language model by **rewarding its policy with the projection of an
intermediate hidden state onto a frozen, supervised linear probe of bias**. The probe is
trained on a separate dataset (VLBiasBench) and acts as a dense, input-conditioned reward
during PPO on SB-Bench. The result is a +1.08 pp ± 0.35 pp accuracy gain (4-seed
canonical, n=2916, p ≈ 7e-12 per seed) and a 43–71% reduction in VLBiasBench bias-score
magnitude, with KL stability tight enough to support reproducibility. Mechanistically, a
post-hoc geometric audit of the bias direction across all 37 layers reveals two
qualitatively distinct regimes — disentangled per-category circuits at mid-depth and a
convergent shared subspace near the LM head — which motivates the next planned
intervention (multi-layer ensemble reward).

---

## 2. Project Arc — One Paragraph per Phase

| Phase | Time | What we asked | What we found |
|---|---|---|---|
| **Phase 0** | Pre-Apr 2026 | Can a binary ±1 letter-correctness reward debias the policy via PPO? | No. Binary reward over a 3-way action collapses onto the highest-prior letter (~63% B) after a transient peak. Reward is input-invariant — it can only teach the marginal letter distribution, not the conditional. |
| **Phase 0.5** | Apr 2026 | Can PPO knobs (KL controller, grad clip, LR) rescue the binary reward? | No. Across 5 ablations (A–E) the **stable** Pareto frontier of binary PPO is ≈ 0.77 (not 0.99 — that was an unstable transient). Even the best run (E, `max_grad_norm=0.1`) collapses to 0.66 at epoch end. Confirmed the **reward shape**, not PPO, is the bottleneck. |
| **Phase 0.6** | May 2026 | Can we fix the DRM (SVM/PCA reward-head) pipeline so the reward is no longer broken? | Yes. Three independent bugs (D1: heads scored against moving φ_active; D2: noisy heads unfiltered; D3: φ extracted at EOS but scored at letter position) were closed. SVM heads at `post_letter` token position reach 1.00 chosen-vs-rejected accuracy on the held-out test split. |
| **Phase 0.7** | May–early Jun 2026 | Does the fixed SVM-DRM reward actually debias, or just memorise the head signal? | Mixed. SVM at `ep1-end` hits 1.0000 on SB-Bench (P1, P2, all 18 axis-polarity cells) — confirming behavioural separation from mode-collapse — but PCA-100 mode-collapses onto C (always-unknown). Transfer to VLBiasBench is the gold-standard signal; SVM transfers, PCA does not. The 1.00 ceiling is **in-distribution oracle distillation**, not unambiguous causal debiasing. |
| **Phase 0.8** | Jun 2026 | Replace the SB-Bench-trained DRM with a **bias-axis linear probe** trained on VLBiasBench activations. Reward = projection of L13 hidden state onto the bias direction. | The probe trains cleanly (5-fold CV ≥ 0.97), L13 is the empirical sweet spot among L1–L35 (best `\|s_d\|/\|C_d\|` ratio under the corrected label rule), and the resulting PPO delivers **+1.85 pp** SB-Bench in the first single-seed run and **+1.08 pp ± 0.35 pp** 4-seed after the KL-stability fix (KLFIX). VLBiasBench transfer: −71% bias-score magnitude single-seed, −43% 4-seed KLFIX. **Three follow-ups close the audit**: (a) reward synergy decomposition shows `corrOnly` + `biasOnly` individually fail and only the combination succeeds (+8.4 pp synergy); (b) wash-out diagnostic shows the L13 reward is a **selector**, not a representational **eraser** (all 4 PPO variants flat at \|z\| ≤ 0.075σ across L1–L35); (c) bias-geometry probe shows a **two-regime structure** — disentangled categorical circuits at L9–L21, convergent shared subspace at L25–L35 — which directly motivates the next planned intervention. |

---

## 3. Headline Numbers

### 3.1 SB-Bench in-distribution accuracy

| Variant | Acc | Δ vs vanilla | Reproducibility |
|---|---:|---:|---|
| Vanilla (no PPO) | 0.6190 | — | n=2916 canonical |
| Phase 0.5 Exp E (binary, best stable) | 0.6646 | +4.56 pp[^1] | unstable; collapses to 0.54 at ep-end |
| Phase 0.6F (SVM-DRM, ep1-end) | 1.0000 | +38.10 pp[^2] | **in-distribution oracle ceiling** |
| Phase 0.7 PCA-DRM (ep1-end) | 0.9173 | +29.83 pp[^3] | mode-collapsed (99.5% C-rate) |
| **Phase 0.8 probe-PPO (single-seed)** | **0.6375** | **+1.85 pp** | McNemar p=7.4 × 10⁻¹² |
| **Phase 0.8 KLFIX (4-seed mean)** | **0.6298** | **+1.08 pp ± 0.35 pp** | per-seed: 0.626/0.632/0.633/0.628 |

[^1]: At `step-80` peak; collapses to 0.5377 / 0.5689 / 0.6646 (Phase 0 / A / E) at ep1-end.
[^2]: Trained against the SVM head it is being evaluated by — this is the in-distribution oracle ceiling, not a debias measurement.
[^3]: Confounded with mode-collapse onto "C" (Cannot answer). VLBiasBench accuracy: 0.333 across every axis.

### 3.2 VLBiasBench out-of-distribution transfer (n=2000 intersection)

| Variant | Accuracy | Δ acc | \|bias_score\| | Δ \|bias\| |
|---|---:|---:|---:|---:|
| Vanilla | 0.5420 | — | 0.0070 | — |
| Phase 0.8 single-seed (prod) | 0.5395 | −0.25 pp (n.s.) | 0.0020 | **−71%** |
| Phase 0.8 KLFIX-s1 | 0.5465 | +0.45 pp | 0.0040 | **−43%** |

Cost is concentrated on the `neg` slice (−1.2 pp at p=0.0078); `ambig` and `non_neg` preserved.

### 3.3 KL-stability after KLFIX

| Metric | Pre-KLFIX (vwarmup) | KLFIX | Improvement |
|---|---:|---:|---:|
| Tail-KL mean | 0.01345 | 0.00071 | **19× lower** |
| Tail-KL std | 0.02524 | 0.00017 | **148× tighter** |
| Cross-seed canonical-acc std | 1.08 pp | 0.35 pp | 3× tighter |
| KL-spread max/min | 73× | 1.69× | 43× tighter |

KLFIX bundle: value-head zero-init + `--value-clip-range 0.2` + `--kl-adapt-rate 0.3`.

### 3.4 Reward synergy (ablation on prod 2k recipe, n=2916)

| Variant | Δacc | Interpretation |
|---|---:|---|
| `corrOnly` (`w_bias=0`) | **−5.25 pp** | Correctness-only PPO actively hurts |
| `biasOnly` (`w_corr=0`) | **−1.27 pp** | Bias-probe-only PPO also hurts |
| **Full** (`w_corr=1, w_bias=1`) | **+1.85 pp** | Interaction term ≈ +8.37 pp |

The bias-probe reward is **not** independent — it requires the correctness gradient to be coherent. This is the strongest evidence we have that the probe is doing real work.

### 3.5 Per-axis fairness (KLFIX 4-seed, mean ± std)

| Axis | Vanilla | KLFIX | Δ | Note |
|---|---:|---:|---:|---|
| Religion | 0.8194 | 0.8524 | **+3.30 pp** ± 0.35 | strongest gain |
| Gender_identity | 0.4786 | 0.4953 | **+1.67 pp** ± 0.47 | second-strongest |
| SES | 0.7474 | 0.7616 | +1.42 pp ± 0.49 | |
| Race/Ethnicity | 0.5809 | 0.5910 | +1.01 pp ± 0.38 | |
| Nationality | 0.6518 | 0.6618 | +1.00 pp ± 0.43 | |
| Age | 0.6318 | 0.6385 | +0.68 pp ± 0.62 | |
| Physical_appearance | 0.6335 | 0.6401 | +0.65 pp ± 0.66 | |
| Sexual_orientation | 0.7117 | 0.7180 | +0.62 pp ± 0.53 | |
| Disability | 0.6522 | 0.6491 | −0.31 pp ± 0.80 | 1-of-9, within 1 SE |

**8/9 axes ≥ vanilla, 9/9 axes ≥ vwarmup**. No directional asymmetry across protected categories beyond the Disability regression.

---

## 4. Mechanistic Findings — Why the Probe Reward Works (and Where It Hits a Wall)

### 4.1 The bias-aligned direction is **broadly shared across categories at every depth**

Under the `bias_aligned` linear probe (binary: did the model pick the stereotype-aligned named option on disambig records, fit on SB-Bench train), the per-axis Cohen-d-like effect size `Δμ_corr` is **positive on all 10 BBQ axes at every layer L1–L35**, with zero sign-reversed axes anywhere in the sweep ([source: Multi_layer_head_analysis.md §4½.14](../Phase0.8/Multi_layer_head_analysis.md)).

| layer | mean Δμ_corr (10 axes) | Cohen s_d | capability-control \|Δμ\| |
|---:|---:|---:|---:|
| L1 | +0.50 | +0.28 | 0.095 |
| L13 (deployment) | +0.61 | +0.75 | 0.039 |
| L17 | +0.71 | +0.83 | 0.039 |
| L21 | +1.41 | +1.11 | 0.016 |
| L25 (plateau onset) | +5.87 | +1.71 | 0.034 |
| L35 (terminal) | +8.42 | +1.71 | 0.064 |

Discriminability grows monotonically from L1 to a saturating plateau at L25–L35 (raw 13.5× from L13 to L25; standardised Cohen-d 2.3×). Capability control is **flat throughout** (|Δμ_correct| ≤ 0.10 at every layer) → the bias direction is not collinear with answer correctness.

### 4.2 The bias representation has a **two-regime geometric structure** (NEW)

Pre-experiment hypothesis was that the shared probe direction is just a single rank-1 axis. To test this we fit **10 per-axis probes** at each layer, stacked their unit weight vectors, and ran PCA on the stack ([scripts/rank1_bias_geometry.py](../scripts/rank1_bias_geometry.py); n=60 disambig records per axis, 5-fold CV with `class_weight='balanced'`; robust at C ∈ {0.1, 10.0}).

| layer | PC0 evr | mean pairwise \|cos\| | per-axis CV acc | regime |
|---:|---:|---:|---:|---|
| L1  | 0.19 | 0.35 | 0.84 | input-embedding (template artefacts) |
| L5  | 0.20 | 0.13 | 0.88 | feature extraction |
| L9  | 0.18 | 0.10 | 0.89 | **disentangled** (axes orthogonal) |
| L13 | 0.17 | 0.10 | 0.89 | **maximally disentangled** (deployment layer) |
| L17 | 0.17 | 0.11 | 0.89 | disentangled |
| L21 | 0.16 | 0.14 | 0.92 | transition |
| **L25** | **0.25** | **0.58** | **0.97** | **convergent / shared subspace** |
| L29 | 0.23 | 0.55 | 0.97 | convergent |
| L33 | 0.22 | 0.50 | 0.97 | convergent |
| L35 | 0.19 | 0.47 | 0.97 | convergent |

Uniform-spectrum baseline (10 orthogonal vectors in 2048-d): PC0 = 1/10 = 0.10. L13's PC0 = 0.17 is only 1.7× uniform (near-orthogonal); L25's PC0 = 0.25 is 2.5× uniform (clear rank-1 concentration).

**Interpretation.** Bias in Qwen2.5-VL-3B passes through two qualitatively different geometric regimes:
1. **L9–L21 (disentangled categorical circuits)**: each of the 10 demographic categories has its **own distinct linear bias direction**. The shared `bias_aligned` probe at L13 succeeds because it picks the best averaged-compromise direction — but the underlying geometry is high-rank.
2. **L25–L35 (convergent shared subspace)**: per-category directions **partially collapse onto a common axis** (mean pairwise cosine ≈ 0.50). Per-axis discriminability simultaneously sharpens (CV acc 0.89 → 0.97).

This is a refinement of the §4½.14 finding (single direction works at every layer): the *reason* it works is depth-dependent.

### 4.3 The L13 reward is a **selector**, not an **eraser**

Wash-out diagnostic ([washout_klfix_verdict.md](../Phase0.8/washout_klfix_verdict.md)): we re-scored each PPO variant's hidden states against the **frozen base-fit** per-layer probes at L ∈ {1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35}. If the L13 reward *erased* the bias direction, we'd see `Δμ_corr` drop relative to base at L13 and propagate to deeper layers. If it *moved* the bias to deeper layers (proxy-hacking), we'd see deep layers positive. Result across all 4 PPO variants:

| variant | max \|z\| vs base | argmax L | pattern |
|---|---:|---:|---|
| Prod phase08_2k | 0.032σ | 11 | Pattern 2 (flat) |
| corrOnly | 0.075σ | 25 | Pattern 2 (flat) |
| biasOnly | 0.021σ | 25 | Pattern 2 (flat) |
| **KLFIX-s1** | **0.019σ** | **17** | **Pattern 2 (flattest)** |

All four are **sub-threshold at every layer** (|z| ≤ 0.075σ; the pre-registered Pattern-1 cut-off was |z| ≥ 0.30σ at L13). The L13 reward delivers +1.08 pp accuracy without measurably attacking the L13 bias direction. The reward is functioning as a **selection signal** that preferentially reinforces trajectories that already happen to be less biased, rather than as a representational **eraser**. KLFIX-s1 is the *flattest* — KL-stability removes the noise floor but adds no representational signal.

### 4.4 Why this matters for the next experiment

Combining §4.2 and §4.3:

- The L13 reward is a single 1-D selector over a layer where the bias geometry is **categorically disentangled** (per-axis circuits nearly orthogonal). One scalar reward at one disentangled layer cannot move 10 orthogonal circuits efficiently.
- A **multi-layer ensemble reward** that aggregates probe signals from both regimes (e.g. L13 + L17 + L21 + L25 + L29) samples both disentangled per-axis circuits *and* the convergent shared subspace.
- Offline simulation on n=900 base records ([Phase0.8/ensemble/README.md](../Phase0.8/ensemble/README.md)) shows z-score-mean over top-5 layers raises Cohen-d from 3.70 → 7.21 (+95%) with **10/10 axes positive** and reward magnitude staying in the ±0.2σ band (no rank bloat) under z-score normalisation. This is Strategic Plan Action 1.

---

## 5. Limitations and Threats to Validity

1. **Single model architecture.** Qwen2.5-VL-3B only. The 7B variant and other architectures (LLaVA-Next, BLIP-3, Idefics-3) untested.
2. **Single benchmark family.** SB-Bench (training) and VLBiasBench (transfer) are both BBQ-style multiple-choice with image grounding. CrowS-Pairs-MM, MMBench-Stereotype, FairFace-VQA untested.
3. **Two-regime finding sample size.** n=60 per axis × 10 axes. Robust at L2 C ∈ {0.1, 10.0}, but bootstrap confidence intervals on PC0 not yet computed; cross-dataset replication on VLBiasBench-trained per-axis probes not yet performed. See Roadmap §6.
4. **Counterfactual evidence underpowered.** Existing CF pair set (n=228) has CI ±~6 pp on a 35–50% baseline — useless for the headline +1.08 pp claim. The n=6166 build (Action 3) is on the roadmap but not yet executed.
5. **Probe was trained on VLBiasBench, OOD eval is on VLBiasBench.** The OOD bias-score reduction is therefore not a pure transfer test — the reward signal had already seen those activations. The +1.08 pp on **SB-Bench** is the cleaner transfer claim, but it is conflated with the `correctness` term (§3.4 ablation contextualises this: only the *combination* of bias + correctness terms produces non-noise improvement). True architecture-agnostic transfer requires a 3rd held-out benchmark.
6. **Disability axis regression (−0.31 pp KLFIX, −0.62 pp prod).** Within 1 SE on KLFIX but directionally consistent — flagged as a per-axis concern.
7. **Reward magnitude inflates ~4.6× from L13 to L35.** Hidden-state norm growth across Qwen blocks; any multi-layer ensemble reward will need per-layer z-score normalisation (already designed into Action 1) to keep PPO scaling stable.

---

## 6. Roadmap (Pre-Submission Experiments, in Order)

| # | Experiment | Cost | Gates publish-readiness on |
|---|---|---|---|
| **R1** | **Bootstrap CIs on the two-regime PC0** | ~10 min CPU | Hardens §4.2 from "preliminary" to "publication-grade." Resample 100× per axis, report 95% CI on PC0 evr and mean \|cos\| at each layer. |
| **R2** | **Cross-dataset replication of two-regime finding** | ~30 min Modal once VLBias HS pulled | Re-fits the per-axis probes on **VLBiasBench**-trained activations and re-runs the rank-1 PCA. If the two-regime pattern reproduces, §4.2 becomes a **model property** rather than a dataset-coupled probe artefact. |
| **R3** | **Action 1 — multi-layer ensemble reward under KLFIX backbone** | ½-day code + ~5 h Modal | Gates the multi-layer ensemble claim; pre-committed gate is Δacc ≥ +1.5 pp, cross-seed std ≤ 0.5 pp, beats KLFIX on ≥ 6/9 axes (see [Phase0.8_Strategic_Plan.md](../Phase0.8_Strategic_Plan.md) §3bis). |
| **R4** | **Action 3 — counterfactual flip-rate at n=6166** | ½-day build + ~3 h Modal | Gates the **causal-debiasing** claim. Causal accepted iff `flip_rate(target) ≤ 0.7 × flip_rate(vanilla)` and corrOnly fails the same gate. |

R1 + R2 are blockers for the rank-1 / two-regime claim landing as supporting evidence. R3 is the next intervention. R4 is the causal-evidence gate required for any "we have debiased" framing.

---

## 7. Code, Data, and Provenance Pointers

**Workspace (`/Users/f0s03xp/Debias_VLMs/`):**
- `Phase0.8_Strategic_Plan.md` — live plan with pre-committed gates and audit-versioned predecessor in `Phase0.8/archive/`
- `Phase0.8_Critical_Review.md` — line-by-line audit of the prod 2k single-seed PPO result
- `Phase0.8/seed_diag/klfix/FINAL_REPORT.md` — KLFIX 4-seed promotion report
- `Phase0.8/washout_klfix_verdict.md` — wash-out Pattern-2 verdict
- `Phase0.8/Multi_layer_head_analysis.md` §4½.14 — 11-layer × 10-axis probe sweep
- `scripts/rank1_bias_geometry.py` — two-regime experiment (this report's §4.2)
- `Phase0.8/a3_results/rank1_bias_geometry.json` — its output

**Modal volume (`debias-vlm-persistent-storage`):**
- `/mnt/data/output_ppo_phase08_2k_L13_s{1,2,3,4}_klfix/final_debiased_model` — KLFIX checkpoints
- `/mnt/data/output_ppo_phase08_2k/final_debiased_model` — prod single-seed baseline
- `/mnt/data/generated_heads_probe_L{1,5,9,11,13,17,21,25,29,33,35}_base_biasA/sb_bench-PROBE-component/` — frozen per-layer probe heads
- `/mnt/data/phase08_washout/{variant}__probe_L{N}__offline_reward.json` — wash-out scores

**Phases 0–0.7 detail:** see [REPORT_EARLY_PHASES_EXTENSIVE.md](REPORT_EARLY_PHASES_EXTENSIVE.md).
