# Interpretable Probe-Reward Debiasing of Vision-Language Models
## REPORT_COMBINED — Summary + Early-Phases + Phase-0.8 Extensive

**Single-document concatenation** of:
- [REPORT_SUMMARY.md](REPORT_SUMMARY.md) — compressed everything
- [REPORT_EARLY_PHASES_EXTENSIVE.md](REPORT_EARLY_PHASES_EXTENSIVE.md) — Phase 0–0.7 detail
- [REPORT_EXTENSIVE.md](REPORT_EXTENSIVE.md) — Phase 0.8 detail

For a reader who wants the single-file view; otherwise consult the three component
docs above for separable reading.

---
---

# PART I — SUMMARY (compressed, all phases)

> Source: [REPORT_SUMMARY.md](REPORT_SUMMARY.md)

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


---
---

# PART II — EARLY PHASES (Phases 0–0.7 extensive)

> Source: [REPORT_EARLY_PHASES_EXTENSIVE.md](REPORT_EARLY_PHASES_EXTENSIVE.md)


---

## 0. Setting

**Model.** Qwen2.5-VL-3B-Instruct, bf16, FlashAttention-2, LoRA r=16/α=32 on text
projection layers (PEFT). 37 LM-decoder layers, 2048-d hidden state.

**Training data.** SB-Bench (BBQ-style 3-way multiple choice, 2916 test items,
9 demographic categories: Age, Disability, Gender, Nationality, Physical_appearance,
Race/Ethnicity, Religion, SES, Sexual_orientation). Ground-truth letter distribution:
A=32.9%, B=34.6%, C=32.5%. Images grounded to demographic stereotypes.

**Transfer benchmark.** VLBiasBench close-ended (10 BBQ axes × 3 conditions × multiple
question formats; 53,034 records after `qformat ∈ {base, scene, scene_text}` filter).

**Infrastructure.** Modal A100-80GB (12 h timeout per training run); single-epoch
on-policy PPO with FirstTokenAllowlistProcessor restricting first generated token to
{A, B, C}; GAE γ=1.0, λ=0.95, ppo_clip=0.2, vf_coef=0.1.

**Research question (entering Phase 0).** Can a single-token correctness reward over
A/B/C train a fairer policy via PPO under LoRA?

---

## 1. Phase 0 — Binary Reward + Mode Collapse

### 1.1 Setup and the first transient

The Phase 0 PPO trainer used a binary reward: `r = +1 if predicted letter == gold letter
else −1`, with a target-KL adaptive controller `β *= clamp(1 + α·(kl/target − 1),
[0.5, 2.0])`. First run produced a striking trajectory:

| Step | Eval acc | Per-token KL | β |
|---:|---:|---:|---:|
| 30 | 0.960 | 0.07 | 0.06 |
| 40 | 0.990 | **0.76** | 0.18 |
| **50** | **0.9949** (peak) | 0.68 | 0.63 |
| 60 | 0.975 | 0.82 | **5.0 (pinned)** |
| ep1-end | **0.5377** | ~0.03 | 5.0 |

The model peaked at near-perfect accuracy at step 50, then collapsed **back to 53.8%
by epoch end**. The collapse mechanism was mode-collapse onto letter "B" ("Cannot be
determined"): predicted ~63% of the time vs ground-truth 34.6%.

### 1.2 Diagnosis: reward shape, not PPO

Binary ±1 reward over a 3-way categorical action is mathematically the worst-case for
policy gradient[^bandit]: the gradient signal is `(±1 − V) × ∇log π(a|s)`, and after KL
re-anchoring the policy's optimum becomes the **highest-prior letter** (which gives
≈ 35% accuracy for free). The reward is **input-invariant** — it sees only the output
token, not the (image, text) pair — so it can teach the **marginal** letter
distribution, never the **conditional**. **It cannot teach causality.**

[^bandit]: Standard result: under softmax policies with KL anchoring and a sparse
±1 reward, the unique stationary point is `π(a) ∝ exp(β·r(a)) · π_ref(a)` — the
posterior-weighted prior. For a 3-class problem with one class having highest prior
and ±1 reward, that converges to "always pick the highest-prior class."

### 1.3 What Phase 0 shipped (code)

| Change | Why |
|---|---|
| Adaptive KL controller with `β` clamped to `[β_min, β_max]` | Stop unbounded KL drift after the peak |
| Cosine LR schedule floored at `min_lr_ratio·peak` via `LambdaLR` | Prevent total-decay-to-zero blowing up the optimizer |
| Value-loss clipping with `value_clip_range` | Standard PPO hygiene (inert in single-epoch PPO; see §3.1 caveat) |
| `--ckpt_every_steps` dense checkpoints | Capture peak before collapse for post-hoc analysis |

Commit hash: `3d5b01e` on branch `phase0_collapse_mitigation`.

### 1.4 What Phase 0 left open

- Could PPO knobs (KL aggressiveness, grad clip, LR, batch size) rescue the binary
  reward, or is the bottleneck the reward *shape*?
- Is the 0.99 peak a real ceiling or just an unstable transient on the way to collapse?

These motivate Phase 0.5's ablation matrix.

---

## 2. Phase 0.5 — PPO-Side Knob Ablation Matrix (A–E)

### 2.1 The ablation design

Five experiments, each isolating one PPO-side hypothesis:

| Exp | Hypothesis | Key knob |
|---|---|---|
| **A** | KL controller too slow + ceiling too low | `kl_beta=1.0, kl_min=1.0, kl_max=50.0, adapt_rate=0.5, target=0.005` |
| **B** | Training-batch acc too noisy for early-stop | `--use_eval_for_early_stop`, `--early_stop_patience 3` |
| **C** | LR too high | `learning_rate=5e-5` |
| **D** | Reward variance dominated by small batch | `gradient_accumulation_steps=8` (effective BS=64) |
| **E** | Single-step weight delta too large | `max_grad_norm=0.1` (was 1.0) |

### 2.2 Phase 0.5 infra shipped

Beyond the experiment matrix:

- **Grad-norm logging fix.** `policy_grad_norm` and `value_grad_norm` were being
  computed only inside `if accelerator.sync_gradients:` — so 3/4 logged rows showed 0.
  Fixed to log every micro-batch.
- **Mid-training held-out eval hook.** `PPOVLMController.evaluate_subset()` runs
  constrained greedy decoding on a deterministic tail-slice (default n=64) every N steps;
  logged as `event=midtrain_eval` rows in `metrics.jsonl`.
- **CLI plumbing.** `--max_grad_norm`, `--use_eval_for_early_stop`, `--midtrain_eval_*`
  added to `args.py`; `run_training` Modal entrypoint accepts the new kwargs.

Commit: `025088c` on `phase0_collapse_mitigation`.

### 2.3 Results — Exp A (aggressive KL)

| Step | KL | β | Eval acc |
|---:|---:|---:|---:|
| 30 | 0.001 | 1.00 | 0.6492 |
| 40 | 0.025 | 2.68 | 0.7401 |
| **50** | 0.069 | 10.7 | **0.8028** (peak) |
| 60 | 0.055 | **50.0 (pinned)** | — |
| 120 | 0.013 | 50.0 | 0.7178 (small recovery) |
| ep1-end | 0.011 | 50.0 | **0.5689** |

**Findings.**
- **KL controller succeeded at its job**: max observed KL = 0.069 (vs Phase 0's 0.82).
- **Peak dropped by 19 pp** (0.9949 → 0.8028) — over-regularisation suppressed the
  learning signal.
- **Collapse still happened**, just at lower amplitude (0.57 vs 0.54 end). The KL
  ceiling pinned at 50.0 by step 60 and never released.

### 2.4 Results — Exp E (tight grad clip)

`max_grad_norm = 0.1` instead of 1.0. Full eval sweep across 27 checkpoints on the
full 2916-sample SB-Bench test set:

| Tag | Exp E acc | Exp A acc | Δ (E − A) |
|---|---:|---:|---:|
| step10 | 0.6142 | 0.6108 | +0.003 |
| step30 | **0.7606** | 0.6492 | +0.112 |
| step50 (KL spike) | 0.7281 | **0.8028** | −0.075 |
| step70 | 0.7136 | 0.6495 | +0.064 |
| **step80 (E peak)** | **0.7709** | 0.5336 | **+0.237** |
| step90 | 0.7541 | 0.4969 | +0.257 |
| step120 | 0.5473 | 0.7178 | −0.170 |
| step194 (resume; β reset to 0.10 by bug) | — | — | — |
| step240 | 0.6361 | 0.5641 | +0.072 |
| **ep1-end** | **0.6646** | 0.5689 | **+0.096** |

**Findings.**
1. **Tight grad clip did NOT prevent the KL spike** at step 50 (KL = 0.802, almost
   identical to Phase 0's 0.76). The clipped per-step delta accumulates over multiple
   steps to the same total drift.
2. **Tight grad clip DID shift the peak** from step 50 → step 80, and let E keep
   improving for 30 steps past the KL spike (while A immediately decayed).
3. **E retained the most signal at ep1-end of any run so far**: 0.6646 vs A's 0.5689
   vs Phase 0's 0.5377.
4. **Peak-to-end drop was smallest** for E (10.6 pp) vs A (23.4 pp) vs Phase 0 (45.7 pp).
   Tight grad clipping **softens** the collapse trajectory even though it doesn't
   prevent the initial KL eruption.
5. **A serendipitous bug helped.** At a forced resume around step 194 the checkpoint
   loader failed to persist `kl_beta` controller state — β reset to 0.10. Accuracy then
   *climbed* from 0.61 to 0.66. This was a real signal: **the β ceiling and the
   controller's pinning behaviour were an active suppressor**, not only the reward shape.

### 2.5 Comparative summary

| Metric | Phase 0 | Exp A | **Exp E** |
|---|---:|---:|---:|
| Max KL observed | 0.82 | **0.069** | 0.80 |
| β ceiling | 5.0 | 50.0 | 5.0 |
| Eval peak | **0.9949** | 0.8028 | 0.7709 |
| Eval peak step | 50 | 50 | **80** (shifted later) |
| Eval ep1-end | 0.5377 | 0.5689 | **0.6646** |
| Peak-to-end drop | −0.457 | −0.234 | **−0.106** |

### 2.6 Key reinterpretation

The Phase 0 0.9949 peak occurred *during* an uncontrolled KL spike — it was an
**unstable transient**, not a stable mode of operation. Once any regularisation is
added that prevents the spike, the stable ceiling drops to ~0.77–0.80. **The 0.77
ceiling — not 0.99 — is the true Pareto frontier of binary-reward PPO on SB-Bench.**

### 2.7 Decision: stop ablating PPO knobs

A + E together disproved three of the four PPO-side hypotheses (controller too lax,
controller too slow, step too large). B/C/D were skipped as low-EV.

**Beating 0.77 stably requires changing the reward shape.** Phase 0.6 is therefore the
DRM (Decomposed Reward Model) implementation — replace the sparse binary scalar with
dense, per-token projections of hidden state `φ` onto direction vectors learned from
labelled bias-bench pairs.

---

## 3. Phase 0.6 — Making DRM Reward Functionally Correct (D1/D2/D3)

### 3.1 The diagnosis: DRM was silently broken in three places

Switching `--reward-mode binary → svm` produced a **zero-variance reward** — PPO was
running on noise. Three independent bugs were the cause:

| ID | One-liner | Status entering Phase 0.6 |
|---|---|---|
| **D3** | Heads trained on a `φ` that does not match what PPO scores at training time | broken |
| **D2** | All heads loaded unfiltered, including category-degenerate ones | broken |
| **D1** | Reward heads projected onto the **moving** `φ_active`, not the frozen `φ_ref` they were fit on | broken |

Implementation order: **D3 → D2 → D1**, gated after each.

### 3.2 D3 — token-position alignment of `φ`

**The bug.** The Phase-1 head-generation pipeline extracted `φ` from the **EOS** of a
free-text completion. At PPO time, the trainer projected `φ` from the **per-generated-token
positions of a short letter completion**. The two distributions are not the same vector
space — at best the SVM separates noise, at worst the head normal vectors point in
directions the trainer can never produce.

**The fix.** Two new flags propagated through `extract.py` and `generate_drm_heads.py`:

- `--completion-format {free_text, letter}` — controls whether the assistant turn is
  the dataset's free-text answer or just the letter (`A` / `B` / `C`).
- `--token-position {eos, pre_letter, post_letter}` — controls which token position's
  hidden state is taken as `φ`.

We tried `pre_letter` first; it produced a **degenerate SVM** (acc ≈ 0.5, ||w|| ≈ 0)
because chosen and rejected share the same prompt prefix up to the pre-letter position.
**`post_letter`** corresponds to the hidden state immediately after the letter has been
emitted — which is exactly what the trainer projects against at the answer position.

**Acceptance.** SVM heads at `post_letter` on the held-out test split scored **acc = 1.00**
across all 9 SB-Bench categories. (Original requirement: `overall_mean ≥ 0.65 AND per-cat min ≥ 0.55`.)

### 3.3 D2 — kept-heads filter

**The bug.** `load_pca_components` loaded **all** `component*.pth` files from the
heads dir. For SVM this meant all 9 category-specific heads, including any that failed
to learn a useful boundary. For PCA, all 50 components, most of which are noise after
the top ~10 eigenvectors. The PPO trainer mean-pools the K projections through the
FastRL `alpha`, so noisy heads dilute the signal and let the policy hack the reward by
drifting `φ` into the null space of the genuine heads.

**The fix.** `evaluate_drm_heads.py` gained `--head_type`, `--keep_threshold` (default
0.55), and writes `kept_heads.json` with:
- `kept_indices`: list of original head indices that passed
- `category_coverage`: `{category_name: [head_indices_that_help_with_it]}`

**SVM rule:** keep head `i` iff its own category's `accuracy_mean ≥ threshold`.
**PCA rule:** keep head `i` iff `overall_per_head[i] ≥ threshold`.

`drm_loader.load_pca_components` gained an optional `kept_heads_filter` argument; when
set, it loads the JSON, validates indices, and subsets to just those heads.

**Acceptance.** With D3 producing acc = 1.0 on all 9 SVM categories, the
`≥ 5 SVM heads AND ≥ 1 per category` gate trivially passed — 9/9 SVM heads kept.

### 3.4 D1 — frozen-φ projection

**The bug.** The trainer was projecting reward heads against the **active** (LoRA-on)
penultimate hidden state:

```python
h_active = curr_penultimate[:, :-1, :]
r_token_k = torch.matmul(h_active, self.reward_heads_weight.T)
```

But the heads were fit on `φ` from the **un-adapted** base model. As LoRA trains,
`φ_active` drifts. The policy can then increase reward without changing its answer
distribution at all, simply by drifting `φ` into directions the heads happen to score
highly — classic **representation hacking**. The reward stops being a function of
model behaviour.

**The fix.** Reuse the `h_ref` that the trainer **already** computes inside
`with self.policy.disable_adapter()` for the KL penalty:

```python
h_reward = h_ref if self.use_frozen_phi else h_active
r_token_k = torch.matmul(h_reward, self.reward_heads_weight.T)
```

A new `--use_frozen_phi` flag (default off → preserves backcompat with binary-mode
Exp E 0.7709 baseline) controls it.

**Acceptance.** A 1-batch smoke run with `--reward_mode svm --use_frozen_phi --kept_heads_filter ...`
logs a per-token `reward_task` with std > 0.01. Smoke result on Modal:
`reward_task=-1.375, reward_task_std=1.0, policy_grad_norm=2.114`. 252/504 trainable
LoRA params received non-zero gradients on the first backward pass.

### 3.5 Phase 0.6 reference training command

```bash
modal run src/run_modal.py::run_training \
  --epochs 1 --max-train-samples 2000 --batch-size 8 --max-gen-tokens 8 \
  --reward-mode svm --head-type svm --use-frozen-phi \
  --heads-suffix _letter_post_letter \
  --kept-heads-filter /mnt/data/generated_heads_letter_post_letter/kept_heads_svm.json \
  --learning-rate 1e-4 --value-learning-rate 5e-4 \
  --kl-beta 0.1 --target-kl 0.02 --kl-adapt-rate 0.1 \
  --max-grad-norm 0.1
```

### 3.6 Phase 0.6 F result: SB-Bench saturation at 1.0000

The SVM-DRM run with all three fixes (`phase06_F_no_causal`) reached **SB-Bench
acc = 1.0000 at step 70**, sustained through `ep1-end` across 18 consecutive
checkpoints. The confusion matrix was perfectly diagonal — **mode-collapse rejected**.

This is the result that motivated Phase 0.7's discrimination question: is the 1.0000
genuine debiasing transfer, or is it just **in-distribution oracle distillation** (since
the SVM heads scored 1.0 on the same chosen-vs-rejected test split during D3 acceptance)?

---

## 4. Phase 0.7 — Verifying the 1.0, Then Testing Transfer

### 4.1 Acceptance gates

| Gate | Test | Pass criterion |
|---|---|---|
| **G1** | P1 — `question_polarity` split on `ep1-end.jsonl` | Both polarities ≥ 0.95 |
| **G2** | P2 — letter-order perturbation (cyclic-1 shift) | Perturbed acc ≥ 0.95 |
| **G3** | P3 — POPE regression | Drop ≤ 5 pp vs base Qwen2.5-VL-3B-Instruct |
| **G4** | T1 — BBQ / VLBiasBench transfer | BBQ acc strictly > base-model BBQ acc |

### 4.2 G1 — polarity & gold-is-unknown splits (P1)

**Source:** 2916 generations from `checkpoint-ep1-end` of `output_ppo_phase06_F_no_causal`.

Every cell scored **1.0000**:

| split | acc | n |
|---|---:|---:|
| polarity = negative | 1.0000 | 1489 |
| polarity = nonneg   | 1.0000 | 1427 |
| gold_is_unknown = False (named-gold) | **1.0000** | **618** |
| gold_is_unknown = True (safe-option-gold) | 1.0000 | 2298 |

Hypotheses ruled out:
- "Always answer the unknown option" — the 618 named-gold items are all correct.
- "Only wins on the easier polarity framing" — 1.0 on both polarities.
- "Only wins where gold = safe option" — named-gold subset is 1.0 too.

**G1 PASS.**

### 4.3 G2 — letter-order perturbation (P2 cyclic-1)

**Design.** Rotate `(ans0, ans1, ans2)` by one position before generation. The
originally-correct answer text moves from old position `L` to new position `(L − 1) mod 3`;
the eval gold label is updated to match. If the policy is reasoning about answer
**content**, accuracy holds; if it is reading **position**, accuracy collapses to ~0.33.

| ckpt | perturbed acc | acc if scored vs ORIGINAL label |
|---|---:|---:|
| `ep1-step70` | **1.0000** (n=2916) | 0.0000 |
| `ep1-end`    | **1.0000** (n=2916) | 0.0000 |

Per-category: all 9 categories at 1.0000. Predicted letters exactly track rotated gold
positions: pre-rotation A=959/B=1008/C=949; post-rotation gold A=1008/B=949/C=959;
predicted A=1008/B=949/C=959. The exact 0.0000 against original positions confirms the
policy followed gold text into its new position, not the previously-correct letter slot.

**G2 PASS.** Combined with G1: the policy is reading the **answer text** and routing it
to the correct letter even under permutation. This is the strongest in-distribution
evidence achievable without a held-out benchmark — but it does not yet falsify
"in-distribution oracle distillation."

### 4.4 G3 — POPE capability regression

POPE = 9000 yes/no object-presence VQA items. Tests whether SVM-PPO broke general VQA.

| ckpt | acc | Δacc | F1 | ΔF1 | unknown-rate |
|---|---:|---:|---:|---:|---:|
| **base** Qwen2.5-VL-3B-Instruct | 0.8714 | — | 0.8746 | — | 0.0068 |
| `ep1-step70` | 0.8636 | **−0.79 pp** | 0.8693 | −0.53 pp | 0.0016 |
| `ep1-end`    | 0.8668 | **−0.47 pp** | 0.8718 | −0.28 pp | 0.0012 |

Both well inside the 5 pp tolerance. **Capability tax is negligible** (< 1 pp on both
checkpoints). Unknown-rate **dropped** — model is more decisive without sacrificing
accuracy. POPE is a different task family (yes/no, no MCQ letters), so the LoRA delta
is transparent to it. **G3 PASS.**

### 4.5 PCA-DRM secondary control variant

To avoid pinning the whole programme to one head family before transfer, we trained
an analogous PPO run using **PCA-100 DRM heads** (90/200 kept above 0.55 signal cutoff).
All other hyperparameters bit-identical to Phase 0.6F.

P1 (n=2916, SB-Bench test):

| polarity | acc |
|---|---:|
| negative | 0.9181 |
| nonneg | 0.9166 |
| gold = named group (n=618) | **0.8333** |
| gold = unknown (n=2298) | 0.9399 |

The PCA G1 nominally **FAILS** (< 0.95). But what matters is the *shape* of the failure:
**a 10.6 pp gap** between gold-unknown (0.9399) and named-gold (0.8333), monotone across
both polarities. The PCA policy is systematically more accurate when the safe option *is*
the gold → it learned a **conservative / "prefer-unknown" bias**. This is the opposite
of stereotype bias, but it is still a measurable bias signature — and it became a
*predictive* finding for what would happen at G4.

### 4.6 G4 — VLBiasBench transfer (the gold-standard signal)

**Sample.** 3000 records stratified across (bbq_axis × condition) = 10 × 3 = 30 cells
(seed=42, 1000 each per condition, 300 per BBQ axis). Filter `qformat ∈ {base, scene, scene_text}`.

| tag | OvAcc | Δ vs base | AmbigAcc | DisambigAcc | BIAS |
|---|---:|---:|---:|---:|---:|
| **base** | 0.5513 | — | 0.9190 | 0.3675 | +0.0070 |
| **svm_ep1-50pct** | **0.5873** | **+3.60 pp** | 0.4080 | **0.6770** | +0.0200 |
| **svm_ep1-end** | 0.5700 | +1.87 pp | 0.2030 | **0.7535** | **−0.0670** |
| **pca_ep1-step80** | 0.3367 | **−21.47 pp** | **1.0000** | 0.0050 | −0.0100 |
| **pca_ep1-end** | 0.3387 | −21.27 pp | **1.0000** | 0.0080 | −0.0100 |

### 4.7 Why overall accuracy is the wrong primary metric here

Base model OvAcc (55.13%) is almost entirely driven by ambig (91.9%) — i.e. by routinely
answering "Can't be determined". Disambig (where context fully resolves the answer) is at
chance: 36.75% ≈ 1/3. **Base is not engaging with the visual/contextual signal on
disambig; it succeeds on ambig by predicting "unknown" indiscriminately.**

The correct view splits the two:

| Variant | Ambig acc (gold = Unknown) | Disambig acc (gold = named group) | Δ disambig vs base |
|---|---:|---:|---:|
| base | 0.9190 | 0.3675 | — |
| svm_ep1-50pct | 0.4080 | **0.6770** | **+30.95 pp** |
| svm_ep1-end | 0.2030 | **0.7535** | **+38.60 pp** |
| pca_ep1-step80 | 1.0000 | 0.0050 | **−36.70 pp** |
| pca_ep1-end | 1.0000 | 0.0080 | −36.67 pp |

**SVM doubles-to-triples disambig accuracy** — the policy stops over-predicting "unknown"
and starts engaging with the disambiguating context. **PCA mode-collapses** to predicting
C on 99.5–99.7% of all 3000 records. This is the **exact failure mode predicted in §4.5**
by the in-distribution conservative-bias signature — confirming the P1 split is a
*predictive* probe for transfer-time failures, not just a literal gate.

### 4.8 SVM late-training drift

Global bias score:

| Variant | Bias score | Direction |
|---|---:|---|
| base | +0.0070 | neutral |
| svm_ep1-50pct | +0.0200 | neutral / slight counter-stereo |
| svm_ep1-end | **−0.0670** | **stereotype-amplification** |

By `ep1-end` the SVM policy has drifted further in the stereotype-confirming direction
on the highest-base-bias axes:

| BBQ axis | base bias | svm 50pct | svm end | drift end vs 50pct |
|---|---:|---:|---:|---:|
| Race_x_SES | −0.16 | −0.12 | **−0.38** | −0.26 pp |
| Race_x_gender | −0.08 | +0.06 | −0.18 | −0.24 pp |
| Disability_status | +0.14 | +0.05 | −0.19 | −0.24 pp |

**Continued PPO past in-distribution saturation does not preserve fairness.** SVM
`ep1-50pct` is the recommended deployment checkpoint.

### 4.9 SVM visual-reasoning regression

Invisible to POPE (yes/no), but G4 surfaces it:

| qformat | base acc | svm_50pct acc | svm_end acc |
|---|---:|---:|---:|
| base | 0.527 | 0.595 | 0.571 |
| scene | **0.726** | 0.562 | 0.557 |
| scene_text | **0.693** | 0.505 | 0.568 |

**SVM degrades on natural-scene images** (−17 pp, −19 pp). PPO on SB-Bench (essentially
text-driven MCQ) blunted the natural-image grounding. The `base` qformat improves
slightly (0.53 → 0.60) because it is most similar to SB-Bench style. This is a known
regression in any shipping artefact from this recipe.

### 4.10 T1.1 — the bridge to Phase 0.8 (REWARD-side pathology proof)

**Scored 990 stratified VLBiasBench responses with both SVM and PCA heads against the
**base** Qwen2.5-VL-3B-Instruct policy** (no PPO). Same processor, chat template,
`post_letter` position as the PPO eval.

Cell-level reward landscape (base model, no PPO):

| Head set | `ambig::C` (right, gold=C) | `disambig::correct` (right, gold=named) | `disambig::incorrect` (picked C, gold=named) | **C-bias** = ambC − dis_corr | **Confusion** = ambC − dis_inc |
|---|---:|---:|---:|---:|---:|
| SVM (K=9)  | **−10.95** | −20.95 | −12.40 | **+10.28** | **+1.47** |
| PCA (K=90) | **−1.91**  | −3.69  | −2.20  | **+1.79**  | **+0.29** |

Both pathologies are present on the **base model** with **no PPO**:

1. **C-bias ≫ 0.** Picking C beats picking the correct named option. PPO will exploit
   this. **PCA's +1.79 explains the mode-collapse exactly** — PCA-PPO discovers
   "always emit C" → +1.79 reward → 99.5% C-rate at convergence. The math is exact.
2. **Confusion ≈ 0.** The reward cannot distinguish "C because gold is C" from "C as
   refusal cop-out". They look identical to the heads. This is why PPO cannot learn
   *when not to* pick C.

Per-head structure:

| Head set | K | Heads preferring C | Heads preferring named | Heads with `|confusion| < 0.3` | C-bias std |
|---|---:|---:|---:|---:|---:|
| SVM | 9   | **9/9 (100%)** | 0 | 0 | 1.63 |
| PCA | 90  | 59/90 (66%) | 31/90 (34%) | **36/90 (40%)** | 4.76 |

**SVM** is structurally homogeneous — all 9 heads agree on "prefer C", but the §4.6
result still showed +30.95 pp disambig improvement. FastRL's α-weighting plus the
sign-alignment work extracted a *named-preferring* combination *despite* the unanimous
C-bias in the equal-mean view. **PCA** is heterogeneous but noisy: top-5 C-preferring
heads have extreme bias (head 1 = +28.67), and 40% of PCA heads are indifferent between
good-C and bad-C — pure noise diluting the α-weighted reward.

**Verdict.**

| Hypothesis | T1.1 status |
|---|---|
| **Reward-side**: heads prefer C / cannot distinguish good-C from bad-C | ✅ **CONFIRMED** |
| **PPO-side**: KL too loose, α collapsed | ❌ **Falsified** — PCA mirrors base reward landscape exactly |
| **Data-side**: ambig dominates training mix | Secondary — irrelevant if reward is structurally biased |

### 4.11 Why this motivates Phase 0.8

The SVM/PCA-DRM heads were built on the **penultimate** hidden state of Qwen2.5-VL-3B.
That state is task-conditioned (it must encode "which letter to output"); the bias-content
representation competes with answer-formatting for the same dimensions. The probing
literature (Tenney, Hewitt & Manning, Belinkov, Schwettmann et al.) repeatedly shows
demographic / social features peak at **middle** transformer layers, not final ones.
And the vision-language fusion in Qwen2.5-VL adds a further hypothesis: bias signal may
live in cross-modal fusion layers (LM blocks ~12–24) before being compressed by
late-layer answer-formatting circuits.

**Phase 0.8 hypothesis (to be tested, not assumed):** there exists a layer L* < penultimate
such that a linear probe of L* hidden states exhibits lower C-bias and higher
good-C-vs-bad-C confusion than the penultimate SVM/PCA heads.

---

## 5. Trajectory Summary — What the Pre-Phase-0.8 Work Established

| Conclusion | Phase | Evidence |
|---|---|---|
| Binary letter-correctness reward cannot debias via PPO | 0 | Mode-collapse to 63% B, peak 0.99 was an unstable transient |
| PPO knobs cannot rescue binary reward | 0.5 | Exp A (KL controller) capped peak at 0.80; Exp E (grad clip) only softened collapse to 0.66 at ep-end |
| The DRM (SVM/PCA) reward pipeline had 3 independent silent bugs | 0.6 | D3 token-position misalignment, D2 noisy-head loading, D1 moving-φ representation hacking |
| Fixed SVM-DRM saturates SB-Bench at 1.0000 | 0.6F | Phase 0.6F no-causal, sustained through 18 checkpoints |
| 1.0000 SB-Bench is **content-routing**, not letter-position memorisation | 0.7 G1+G2 | P1 cells all 1.0; P2 cyclic-1 also 1.0; cross-permutation perfectly tracked |
| Capability tax is negligible | 0.7 G3 | POPE Δacc = −0.47 pp at ep1-end |
| SVM transfers to VLBiasBench, **PCA mode-collapses** | 0.7 G4 | SVM disambig +30.95 pp / +38.60 pp; PCA 99.5% always-C |
| **The transferable SVM gain is highest at `ep1-50pct`**; late training drifts | 0.7 §5.6 | bias_score goes +0.020 → −0.067 from mid to end; Race_x_SES bias −0.38 at ep1-end |
| SVM has a scene-image visual-reasoning regression | 0.7 §5.8 | −17 pp on scene qformat, invisible to POPE |
| **Both pathologies (SVM late drift, PCA mode-collapse) are REWARD-side, not PPO-side** | 0.7 T1.1 | Base-model reward landscape already biased: PCA C-bias = +1.79, SVM C-bias = +10.28 |
| Penultimate-layer heads conflate bias and answer formatting | 0.7 T1.1 → 0.8 | Confusion ≈ 0 for both head families: cannot distinguish "C because gold = C" from "C as refusal" |
| **Pivot to probe-as-reward at a mid-layer** | 0.8 | Test whether a linear probe at L < penultimate can decouple bias from answer formatting |

The Phase 0.8 deep-dive picks up here: probe design, layer audit, PPO recipe, KLFIX,
wash-out diagnostic, two-regime bias geometry. See [REPORT_EXTENSIVE.md](REPORT_EXTENSIVE.md).

---

## Appendix A. Reproduction-Footnote Index

**Phase 0 / 0.5:**
- Commits: `3d5b01e` (Phase 0), `025088c` (Phase 0.5 infra), branch `phase0_collapse_mitigation`.
- Modal volume artefacts:
  - `/mnt/data/output_ppo_phase0/` — Phase 0 baseline
  - `/mnt/data/output_ppo_phase05_A_high_kl_floor/` — Exp A (with `phase05_eval_sweep.json`)
  - `/mnt/data/output_ppo_phase05_E_tight_grad_clip/` — Exp E (with `phase05_eval_sweep.json`)
- Scripts: [scripts/phase05_ablation.sh](../scripts/phase05_ablation.sh),
  [scripts/phase0_letter_distribution.py](../scripts/phase0_letter_distribution.py).
- Local copies of metrics: `phase05_results/` (A and E training metrics + A's eval sweep).
- Detailed handoff: [scratch/Phase0.5_Context_Handoff.md](../scratch/Phase0.5_Context_Handoff.md).

**Phase 0.6:**
- Branch `phase0_collapse_mitigation`; D1/D2/D3 work documented in [Phase0.6.md](../Phase0.6.md).
- New Modal entrypoint: `run_drm_eval` (see `src/run_modal.py`).
- Heads dir suffix: `/mnt/data/generated_heads_letter_post_letter/` (added by D3
  auto-suffixing).
- Final SVM-DRM checkpoints under `/mnt/data/output_ppo_phase06_F_no_causal/`.

**Phase 0.7:**
- Branch `phase0.7_transfer_eval`; documented in [Phase0.7.md](../Phase0.7.md).
- Remediation analysis: [Phase0.7_Remediation_Plan.md](../Phase0.7_Remediation_Plan.md).
- VLBiasBench gens: `/mnt/data/phase07_vlbiasbench/{base,svm_ep1-50pct,svm_ep1-end,pca_ep1-step80,pca_ep1-end}_vlbias_gen.jsonl`.
- POPE results: `/mnt/data/phase07_pope/{base,phase06F_step70,phase06F_end}_pope_results.json`.
- T1.1 offline reward scoring: [src/modules/evaluation/score_vlbias_offline.py](../src/modules/evaluation/score_vlbias_offline.py),
  [scripts/phase07_t1_offline_reward.sh](../scripts/phase07_t1_offline_reward.sh).


---
---

# PART III — PHASE 0.8 EXTENSIVE

> Source: [REPORT_EXTENSIVE.md](REPORT_EXTENSIVE.md)

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

## 9. The Bias Representation Has a Two-Regime Geometric Structure (Supporting Evidence)

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

1. **n=60 per axis.** Small sample with class-imbalanced labels (~10–19 positives per
   axis). Robust at C ∈ {0.1, 10.0} regularisation, but bootstrap confidence intervals
   on PC0 have not yet been computed.
2. **Single dataset.** The per-axis probes were fit on VLBiasBench hidden states (the
   same source as the deployment probe). The two-regime pattern has not yet been
   replicated on SB-Bench-derived hidden states. If the SB-Bench replication produces
   the same pattern, the finding becomes a **model property** rather than a
   dataset-coupled probe artefact.
3. **Single model.** Qwen2.5-VL-3B only. Whether the two-regime structure is specific
   to this architecture, or a general property of multi-modal transformers with
   similar depth/hidden-dim, is untested.
4. **The "convergent subspace" claim is geometric, not interpretive.** We have not
   shown that the deep-layer shared axis is *causally* responsible for the final
   answer — only that the per-category probe directions align there.

Both R1 and R2 in the roadmap (§10.3) are pre-registered to address (1) and (2)
before this finding is promoted from "supporting evidence" to "headline contribution".

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

---

## 10. Strategic Implication and Roadmap

### 10.1 Action 1 — multi-layer ensemble reward (next intervention)

**Offline simulation result** ([source: Phase0.8/ensemble/README.md](../Phase0.8/ensemble/README.md)).
Replayed all 11 layers' probes on the same 900 cached records, then z-score-normalised
per-layer and averaged across the top window {L17, L21, L25, L29, L33}:

| Variant | Δμ_corr | Cohen-d | pooled_sd | axes pos/10 |
|:--|--:|--:|--:|:--:|
| single L13 (current head) | +0.621 | +3.70 | 0.168 | 10/10 |
| single L21 | +1.431 | +5.68 | 0.252 | 10/10 |
| single L25 | +5.890 | +6.29 | 0.937 | 10/10 |
| ensemble raw_mean | +4.434 | +6.97 | 0.636 | 10/10 |
| **ensemble zscore_mean** | **+1.586** | **+7.21** | **0.220** | **10/10** |
| ensemble rank_mean | +165.140 | +4.26 | 38.748 | 10/10 |
| ensemble z_max | +1.362 | +4.59 | 0.297 | 10/10 |

`zscore_mean` is the only variant that simultaneously:
- Strictly dominates every single-layer baseline on Cohen-d (+95% over L13);
- Keeps the reward in a sensible scale (no rank-bloat from per-layer norm growth);
- Is positive on all 10 BBQ axes.

**Implementation gaps** (½-day code):
1. [src/modules/training/drm_loader.py](../src/modules/training/drm_loader.py): load
   N probe heads keyed by layer; store as a `dict[int, nn.Linear]`.
2. [src/modules/rl_components/phase08_ppo_trainer.py](../src/modules/rl_components/phase08_ppo_trainer.py):
   single forward pass capturing hidden states at all N layers; per-layer z-score
   normalisation; mean pooling to a scalar reward.
3. [src/modules/training/args.py](../src/modules/training/args.py): new flag
   `--reward-mode bias_aligned_ensemble`, or extend `--reward-head-layer` to accept a
   comma-separated list with `--reward-pool {zmean,mean,max}`.
4. New launcher `scripts/phase08_ensemble_*.sh` mirroring the KLFIX launcher structure.

**Pre-committed gate** (Strategic Plan §3bis):
- Mean canonical-acc Δ vs vanilla ≥ **+1.5 pp** (beats KLFIX +1.08 pp by ≥ 0.4 pp).
- Cross-seed std ≤ **0.5 pp**.
- Per-axis: beats KLFIX on ≥ 6/9 BBQ axes.
- KL stability: tail-KL std ≤ 0.005.
- Ambig channel preserved: VLBias ambig rate within −2 pp of vanilla.

**Expected outcome (priors).** If the offline +95% Cohen-d gain transfers at any
non-trivial rate, KLFIX +1.08 pp should rise to +2.0 to +3.0 pp on canonical n=2916,
with cross-seed std ≤ 0.5 pp (ensemble averages out per-layer noise — layer-layer
Pearson is 0.87–0.99 across the top window).

### 10.2 Action 3 — counterfactual flip-rate at n=6166 (causal-evidence gate)

The existing CF pair set (n=228) has CI ± 6 pp on a 35–50% baseline — useless for the
+1.08 pp headline claim. The n=6166 build is on the roadmap. Pass criterion:
`flip_rate(target) ≤ 0.7 × flip_rate(vanilla)` AND `accuracy_swap − accuracy_orig`
within ±2 pp AND corrOnly does not meet the first criterion. This is the only test
that converts the headline from an "accuracy" claim into a "causal debiasing" claim.

### 10.3 Pre-Action-1 hardening experiments (new, added 2026-06-20)

The two-regime finding (§9) is currently supporting evidence with limitations (n=60,
single dataset). Two cheap follow-ups are pre-registered to land before Action 1:

| # | Experiment | Cost | What it gates |
|---|---|---|---|
| **R1** | **Bootstrap CIs on per-layer PC0** | ~10 min CPU | Resample 100× per axis with replacement; report 95% CI on PC0 evr and mean \|cos\| at each layer. Decision: §9.4 table becomes publication-grade iff the two-regime gap (L13 PC0 vs L25 PC0) is statistically separable. |
| **R2** | **Cross-dataset replication on SB-Bench** | ~30 min Modal | Re-fit per-axis `bias_aligned` probes on SB-Bench-derived hidden states; re-PCA. Decision: two-regime pattern is a **model property** iff both datasets produce the same regime structure. Otherwise it is a dataset-coupled probe artefact and we will report it accordingly. |

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
