# Phase 0.9 Critical Review — Methodology audit before Phase 0.9.5 ablations

> **Date**: 2026-06-24 · **Branch**: `phase0.9_multilayer_ensemble_reward` · **Head at writing**: `cd68767`
> **Companions**: [Phase0.9_Strategic_Plan.md](Phase0.9_Strategic_Plan.md) (forward-looking plan), [Phase0.9/FINAL_REPORT.md](Phase0.9/FINAL_REPORT.md) (R1–R3 verdict), [Phase0.8/Multi_layer_head_analysis.md](Phase0.8/Multi_layer_head_analysis.md) (per-layer Δμ_corr, Cohen-d, ambig signature)
>
> **What this is**: pre-Phase-0.9.5 critical review of the R3 multi-layer ensemble result. R3 is *one point* in a 4-dimensional design space (window composition × pool function × probe sample size × LoRA placement). Before committing to R4 (causal-evidence gate at n=6166), we ablate the design axes that R3 left unexplored. This document is the analytical justification for the eight ablations listed in the strategic plan.
>
> **What this is not**: a re-litigation of the R1/R2 findings, both of which closed cleanly (publication-grade two-regime convergence; model-property cross-dataset replication on |cos|).

---

## 1. Why was L13 excluded from the ensemble, and is the exclusion justified?

**Genesis of the exclusion.** The `{L17, L21, L25, L29, L33}` window came from `Phase0.8/ensemble/README.md` Test A, which optimised **Cohen-d on the offline n=900 cache** — pure within-sample signal sharpness. By that metric L13 (Cohen-d +3.70) was strictly dominated by L21 (+5.68) and L25 (+6.29), so excluding it was "free" — i.e., it improved the Test A ranking without any downside that Test A could measure.

**The downside Test A could not see.** [Phase0.8/Multi_layer_head_analysis.md](Phase0.8/Multi_layer_head_analysis.md) §4½.14 documents a property the offline-Cohen-d ranking ignored — the ambig signature flips sign past L17:

| layer | `Δμ_corr` | `μ(ambig::incorrect)` | reward scale vs L13 | ambig signature |
|---:|---:|---:|---:|---|
| **L13** | +0.621 | **+0.184** | 1.00× | **intact ✅** |
| L17 | +0.724 | +0.027 | 1.00× | borderline |
| L21 | +1.430 | −0.030 | 1.49× | flipping |
| L25 | +5.890 | −0.288 | 3.24× | strongly negative |
| L29 | +6.450 | −0.260 | 3.50× | strongly negative |
| L33 | +7.675 | −0.311 | 4.10× | strongly negative |

`ambig::incorrect μ > 0` is the canonical BBQ ambig-bias signature ("when the model fails on an ambiguous prompt, it picks the stereotype-aligned option"). Probes at L21+ score these records as "anti-bias" because the records are out-of-distribution for the probe's training (`bias_aligned=None` on ambig records, see §4½.13 source).

**Connection to the R3 measured failure.** The R3 ensemble eroded VLBias `ambig_acc` by **−2.39 pp** vs vanilla; the KLFIX-L13 baseline erodes by only −1.0 pp. The mechanism is now mechanically obvious in hindsight: **the shipped R3 ensemble contains zero layers with intact ambig signature**. L13 was the ambig-preserving probe in the family and we removed it.

**Predicted effect of adding L13 back to the ensemble** (6 layers `{L13, L17, L21, L25, L29, L33}`):

- Mean ensemble z mass shifts toward 0 on ambig.correct records (L13's z is high-positive on ambig.correct cells where the deeper layers are strongly negative → pool-mean is closer to 0 → after negation, reward stops aggressively pushing the policy away from emitting C on ambig prompts).
- Expected gain: recover 1.0–1.4 pp of VLBias `ambig_acc` without losing the OOD `bias_score` benefit.
- Expected risk: marginal dilution of the deep-layer signal (one more "small magnitude" entry in the per-layer z pool).

**Two ablations follow directly:**
- **A1**: `{L13, L17, L21, L25, L29, L33}` (add L13, zmean pool unchanged) — direct compensation test.
- **A2**: `{L13, L17, L21}` (disentangled-regime-only) — falsification: does the deep-layer convergent regime contribute OOD, or is the entire R3 OOD benefit attributable to the disentangled regime?

---

## 2. Is `zscore_mean` the best pool function, or is there something more robust?

**The logic of `zmean` as it stands.** Per-layer raw projections have wildly different scales (L17 σ=0.89 vs L33 σ=4.36, a 5× spread). Without per-layer z-norm, a raw mean is dominated by the deepest layer. After per-layer z-norm, all layers contribute on a 1-σ scale, then equal-weight average to a scalar. `Phase0.8/ensemble/README.md` Test A picked zmean over five alternatives because it was the only one that simultaneously (a) strictly dominated every single-layer baseline on Cohen-d, (b) kept reward scale sensible (no rank-bloat), and (c) was positive on all 10 BBQ axes.

**Three assumptions zmean makes that may not hold.**

1. **Equal information value across layers.** Your compensation argument from §1 explicitly breaks this: L13's strength (ambig preservation) is not in the same dimension as L25's strength (sharp disambig discrimination). Equal weight under-uses L13's unique contribution.
2. **Calibration distribution = training distribution.** μ/σ were computed on the cached HS of the **vanilla model's greedy outputs** (n=751). At PPO time the policy samples at temperature with active LoRA — the HS distribution drifts. We see this directly in the smoke metrics: training-time L17 z ≈ −1.14 vs calibration mean of 0.0. The z-norm scaler is fixed, so distribution shift bakes in.
3. **Per-record independence.** Not really; layer-layer per-record Pearson is 0.87–0.99 in the top window, so the 5 z's are highly correlated. Equal-weight mean of correlated signals reduces noise by only √(1+(N−1)ρ)/N ≈ 0.94 (vs 0.45 for uncorrelated), so statistical benefit of averaging is small.

**Alternative pool functions and when each should win:**

| Pool | Formula | Differentiator | Should win when… |
|---|---|---|---|
| `zmean` (current) | `mean(z_L)` | equal-weight, scale-invariant | Layers carry similar information |
| **`zmedian`** | `median(z_L)` | robust to one-layer outliers | One layer has a different OOD profile (our case: L13 vs L25–33 on ambig) |
| **`agreement_mean`** | `mean(z_L) · [count(sign(z_L)==sign(mean)) ≥ K]` | zero reward unless K layers agree on sign | Confident-detection matters more than magnitude |
| `trimmed_mean` | `mean(sorted(z_L)[1:-1])` | drop top + bottom per record | Sample-level outlier rejection |
| `varpen_mean` | `mean(z_L) − λ · std(z_L)` | penalise inter-layer disagreement | Disagreement signals "not confidently bias" |
| `cohend_weighted` | `Σ w_L z_L, w_L ∝ Cohen-d` | up-weight more-discriminative layers | If discriminability is the right axis (likely worsens ambig erosion — *not* recommended) |

**Recommended ablations:**

- **B1**: `zmedian` with the winning window from A1/A2. Median of 6 z's is much less swayed by extreme deep-layer values, so the ambig.correct cells (L25+ strongly negative, L13 strongly positive) won't get pushed by the deep-layer majority — the median sits at the middle-layer "borderline" value. This is the cleanest theoretical match to "compensate one layer's blind side with another layer's information."
- **B2**: `agreement_mean` with K=4 of 6 (or K=2 of 3 for A2). Only rewards bias when ≥ K layers agree on the sign. High-bias records → strong negative reward. Ambig.correct records (where layers disagree because L13 says "fine, model abstained" and L25+ say "anti-bias-aligned by my reckoning") → near-zero reward → policy not pushed. Confidence-gated signal.

**The current `zmean` is a reasonable default but is not doing the robustness work the compensation hypothesis envisions.**

---

## 3. Single-seed-first as a standing rule for ablations

A trivial methodology win. The R3.3 smoke pattern (`SEEDS="1" bash ...`) is already infrastructure-supported.

**Adopt as Phase 0.9.5 standard.** Decision rule: if s1 SB-Bench canonical Δ vs vanilla ≥ +1.0 pp AND OOD bias_score reduction ≥ +0.001 → fire s2/s3/s4. Otherwise stop the ablation and reason about why.

**Cost asymmetry**: $1.50 for s1 vs $6 for 4 seeds. We save $4.50 every time an ablation doesn't pan out, with negligible information cost (single seed is noisier but usually directionally informative for a binary "is this worth running 4 seeds" decision).

**Hard rule going forward**: no 4-seed runs without a green s1.

---

## 4. Phase 0.9 deeper scope — what we measured vs what we left in the design space

The R3 result we shipped is **a single point in a 4-dimensional design space**:

| Axis | Current setting | Reason to ablate |
|---|---|---|
| Window | `{L17, L21, L25, L29, L33}` | Missing L13 → ambig signature flipped → −1.4 pp ambig erosion (the only OOD downside) |
| Pool | `zmean` | Equal-weight average of 0.87+ correlated signals; doesn't exploit the compensation hypothesis |
| Probe n | 900 (D=2048 underdetermined) | Untested convergence; cross-dataset magnitude drop in R2 may be partly L2-regularisation artefact |
| LoRA placement | Distributed (default) | Wash-out diagnostic shows the reward is barely moving representations; targeted LoRA at bias-forming layers may amplify the actual mechanism |

Plus two orthogonal experiments outside the four axes:
- **Inference-time erasure baseline** (§6 below): a free, missing, orthogonal data point.
- **First-significant-layer framing** (§8 below): converges to A1+B2 (`{L13, L17, L21, L25, L29, L33}` + agreement-mean pool).

The forward-looking sequencing of these eight items is in [Phase0.9_Strategic_Plan.md](Phase0.9_Strategic_Plan.md) §3. This document only argues *why* each ablation is worth running.

---

## 5. Targeted LoRA on the bias-forming layers — well-motivated alternative

**Current state.** LoRA targets `q_proj, k_proj, v_proj, o_proj` (and possibly MLP) at every transformer block (PEFT default for `LoraConfig(task_type=CAUSAL_LM)`). Gradient flow is distributed across all 36 blocks of Qwen2.5-VL-3B.

**Why distributed LoRA may be sub-optimal.** REPORT_EXTENSIVE §8 (the KLFIX wash-out diagnostic) shows the per-layer activation shift is uniformly tiny (max |z| ≤ 0.075σ across all 11 probed layers). The single-layer reward isn't moving representations anywhere in particular — it's selecting between completions. Distributed LoRA may be over-parameterised for the actual change needed.

**Hypothesis.** If we constrain LoRA to the regime-boundary window (e.g., L17–L25), the policy has fewer degrees of freedom but is forced to make the change where the bias representation is actually forming. The bias signal grows sharpest at L17→L21→L25 (Multi_layer_head_analysis §4½.14). Targeting LoRA there could directly modify the bias-forming computation rather than the post-hoc completion selection.

**Counter-risks.**

- Constrained LoRA may sacrifice the policy's ability to fix things elsewhere (e.g., the LM-head probability re-weighting that the post-letter token sits over).
- Fewer trainable params → faster training, but also less capacity to learn the multi-axis fairness manifold.
- Empirically untested on Qwen2.5-VL family; literature reports are mixed.

**Ablation E4** (single-arm, s1-first, $1.50 budget): ensemble reward (winning config from A/B) + LoRA targeted at attention/MLP **only in blocks 17–25** (9 blocks). All other hyperparameters identical to R3.4a. PEFT supports this via `target_modules=[...]` with explicit layer regex (e.g., `r"model\.layers\.(1[7-9]|2[0-5])\.(self_attn|mlp)\..*"`).

**Decision gate** (s1-only): Δ vs vanilla ≥ R3's +1.29 pp AND VLBias ambig_acc ≥ R3's 89.51% → green for 4-seed. Otherwise the constrained LoRA loses to distributed and we drop the line.

---

## 6. Probe-as-reward vs inference-time erasure — the missing baseline

**The framing.** Two approaches differ in *what they modify* and *what they cost*:

| Approach | Modifies | Cost | Reversibility |
|---|---|---|---|
| **Probe-as-reward + PPO (current)** | Policy weights (LoRA adapter) | $6+ per run | Permanent (saved adapter) |
| **Inference-time direction erasure** | Forward-pass activations only | $0 training, cheap inference | Trivially reversible |

The literature term for the second is **linear concept erasure** or **INLP** (Iterative Null-space Projection). Applied to a single layer L with bias direction `v_bias`:

```
For each forward pass at layer L:
    h_L ← h_L − (h_L · v̂_bias) v̂_bias        # v̂ = unit-normed
```

This removes the component of every hidden state along the bias direction. Does nothing about other layers, doesn't affect training.

**Why we haven't tried it.** Probe-as-reward came out of Phase 0.6/0.7 where activation-steering attempts (PCA, SVM) had collapse issues (Multi_layer_head_analysis §4½.11: the `pca_ep1-end` 100% C-emission failure). But that was activation *steering* (`h ← h + λv`), which is the wrong direction. Activation *erasure* (`h ← h − (h·v̂)v̂`) is a different operation and is untested in our pipeline.

**Re-stating the user's concern correctly.** "The whole idea was to isolate the subspace within the representations and eliminate this bias subspace" — inference-time erasure IS exactly subspace elimination applied directly. Reward shaping is the indirect version (the policy slowly learns to avoid biased states); inference-time is the explicit, immediate version. Both are doing the same thing to the same subspace; they differ in *when* and *how*.

**Why this is a critical missing baseline.** If single-layer L13 erasure matches PPO+KLFIX on SB-Bench Δ vs vanilla AND OOD bias_score reduction, the entire $50+ PPO investment becomes the wrong primary. If PPO meaningfully beats it, that's evidence the policy is doing something beyond direction-removal — and worth the cost.

**Ablation E1** (zero-training baseline, $0.50): inject a `pre_forward` hook at L13 on Qwen2.5-VL-3B that does `h ← h − (h·v̂_bias)v̂_bias`. Score on canonical SB-Bench n=2916 + VLBias n=2000 from the base model with the modified forward pass. Strictly cheaper than R3.3 smoke. **This is the highest-EV first move of Phase 0.9.5.**

Optional extension if E1 is encouraging: multi-layer erasure at `{L13, L17, L21, L25}` (still no training; same hook applied at multiple layers).

---

## 7. n = 900 vs D = 2048 — is the probe underfit?

A legitimate methodological concern. With n=900 and D=2048, the system is under-determined. The probe finds a solution that's "best" on the training data under L2 regularisation but may not be the *best* bias direction recoverable with more data.

**Quantitative reasoning.**

- **Strict underdetermined regime.** n < D means a (D−n) = 1148-dimensional null-space of `X^T X`. Unregularised LR has infinitely many zero-training-error solutions in this regime.
- **The L2 regularisation (C=1.0) picks the minimum-‖w‖² solution from this null space.** This is the *only* reason the probe gives a stable answer.
- **Effective dimension.** Looking at the §4½.10 holdout accuracies (0.866 → 0.881 → 0.940 across L9/L11/L13), the probe is finding real signal — 5-fold CV gives unbiased estimates. The signal is recoverable in the regularised solution.
- **What we can't verify with n < D.** Whether the recovered direction is the *best* bias direction or just a regularisation-biased projection. With more data, the unregularised solution would be unique and could differ from the regularised one.

**Diagnostic test (cheap).** Refit at L13 with `n ∈ {600, 900, 1500, 3000, ...}` (whatever's available) and compute `cos(w_n, w_n_max)` — the convergence rate of the probe direction. If `cos(w_900, w_3000) > 0.95`, n=900 is essentially as good as more data; the regularisation is already converging. If `cos(w_900, w_3000) < 0.90`, more data is materially changing the direction and our probes are underfit.

**Practical note.** The VLBiasBench gen file is much larger than n=900 — the original probe set came from a `max_per_cell=30` stratified subsample of ~3000+ records. We can refit at `max_per_cell=100` or no cap for the diagnostic. Estimated cost: ~30 min A100 to re-extract HS for the larger subset + 10 min CPU to refit probes.

**Risk if we don't address this.** The probe direction we're rewarding could be partly an artefact of L2 regularisation. The R2 cross-dataset replication is reassuring (the probe transfers, suggesting real signal), but the **magnitude difference** between VLBias and SB-Bench in R2 (~3× weaker on SB-Bench) is consistent with the probe being partly fit-set-specific.

**Ablation E2** ($1 total): convergence test at L13 with `n ∈ {600, 900, 1500, 3000}` (or whatever is available). If `cos(w_900, w_max) > 0.95`, lock the existing probes and proceed. If < 0.90, retrain at higher n and rebuild the ensemble bundle before any of the A/B ablations.

---

## 8. "Eliminate bias where it gains significance" — first-significant-layer principle

The user's eighth concern points at a **first-significant-layer principle**: target intervention at the earliest point where bias signal is statistically real, plus all downstream layers as "no-progress gates."

From Multi_layer_head_analysis §4½.14:

| Layer | Cohen-d (signal sharpness) | Interpretation |
|---:|---:|---|
| L1  | +0.275 | template lexical features, no real signal |
| L5  | +0.498 | borderline |
| **L9**  | +0.676 | first crosses 0.5; signal first detectable |
| L11 | +0.753 | |
| **L13** | +0.752 | crosses 0.75 plateau |
| L17 | +0.825 | |
| L21 | +1.114 | crosses 1.0 |
| L25 | +1.709 | jump to convergent regime |
| L29–L35 | +1.7 (saturated) | |

Two readings of "first significant":

- **Conservative (Cohen-d ≥ 0.5)** → L9 or L11.
- **Aggressive (Cohen-d ≥ 0.75)** → L13 or L17.

**Convergence with §1.** This argument independently lands on the same conclusion as the L13-inclusion ablation: the ensemble should include layers from BOTH "first-significant" AND "convergent" regimes. Specifically:

- `{L9 or L13}` (first-significant; ambig signature intact)
- `{L17 or L21}` (sharp standardised signal; reward scale still sane)
- `{L25 or L29}` (peak convergent signal; reward-scale risk; ambig-OOD flip)

**The "no progress" operationalisation.** You want the reward at L13 to PUSH the policy to keep the L13 representation un-biased, AND the reward at L25 to PUSH it to keep the L25 representation un-biased. If the ensemble pool uses any "all-must-agree" function (like `agreement_mean` from §2), the policy can only get reward when it has driven bias to zero at *every* layer in the window. This is much stronger than the current "average bias signal" objective.

**This converges exactly to ablation A1 + B2** (`{L13, L17, L21, L25, L29, L33}` + agreement-mean pool with K=4/6). The first-significant-layer principle and the L13-compensation principle agree on the same experiment.

---

## 9. Synthesised verdict and ablation priority order

| # | Ablation | What it tests | Cost (s1) | Cost (4-seed if green) | Priority |
|---|---|---|---|---|---|
| E1 | Inference-time L13 erasure | Whether PPO is the right primary at all | $0.50 | n/a (one-shot) | **HIGHEST EV** |
| E2 | Probe convergence test n∈{600,900,1500,3000} | Whether n=900 probes are underfit | $1 | n/a (one-shot) | **HIGHEST EV** |
| E3 (A1+B1 combined) | `{L13,L17,L21,L25,L29,L33}` + zmedian | L13 compensation + robust pool | $1.50 | $6 | High |
| E4 | Targeted LoRA L17–25 + winning E3 | Whether constrained LoRA helps | $1.50 | $6 | Medium (conditional) |
| E5 (A2+B1) | `{L13,L17,L21}` disentangled-only + zmedian | Does deep regime contribute OOD? | $1.50 | $6 | Medium |
| E6 (A1+B2) | `{L13..L33}` + agreement_mean K=4/6 | Confidence-gated reward | $1.50 | $6 | Medium |
| E7 | Probe retrain n>>900 if E2 demands | Direction-stability of probe | $2 | n/a (one-shot, conditional on E2) | Conditional |
| R4 | CF n=6166 with winning config | Causal-evidence gate | $3 + ½-day build | conditional | After 0.9.5 |

**Worst-case total spend for Phase 0.9.5**: ~$45 (every ablation fan-outs to 4 seeds). **Expected spend**: ~$15–25 (most ablations will trigger only s1).

R4 budget is then ~$10 (build + 1 Modal sweep), bringing total Phase 0.9.5 + R4 spend to under $60.

This is far less than the cost of running R4 with a wrong primary configuration and getting an inconclusive causal-evidence result.

---

## 10. Open questions and reading suggestions

- The exact PEFT regex for layer-targeted LoRA on Qwen2.5-VL-3B: needs to be derived from the actual module-name layout (`model.layers.N.self_attn.q_proj` etc.). Single command can list module names: `grep -E "model\.layers\.[0-9]+\." <state_dict_keys_dump> | head`.
- Whether `agreement_mean` is differentiable: it's piecewise constant in the indicator, but the indicator is binary and the reward signal multiplied by it is differentiable in `mean(z_L)` everywhere the indicator is 1. Reward gradients only flow through the mean, not through the indicator (it's a reward-side operation, not a policy-side operation, so non-differentiability is fine).
- Whether E1's inference-time erasure should use the L13 probe direction or the ensemble-pooled direction (e.g., the first principal axis of W = stack of 5 unit-normed probes). Reasonable to do both; trivial cost.

These are notes-to-self for the dev phase; they don't change the ablation priority order.

---

**This review's verdict**: R3 is a respectable first-pass measurement that revealed a clear trade-off (OOD bias-reduction vs ambig erosion). Phase 0.9.5 ablations are designed to (a) test whether that trade-off can be neutralised by including L13, (b) check whether the L13 probe direction was even well-fit to begin with, and (c) supply the missing zero-training inference-time baseline. R4 is best run with whichever configuration wins 0.9.5; running it now would lock in the R3 point and force us to defend a sub-optimal setting.