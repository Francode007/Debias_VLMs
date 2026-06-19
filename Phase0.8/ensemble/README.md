# Phase 0.8 — Multi-layer ensemble + probe seed-sensitivity (offline)

**Date**: 2026-06-19 · **Branch**: `phase0.8_strategic_plan_day1`
**Script**: [scripts/phase08_ensemble_offline_tests.py](../../scripts/phase08_ensemble_offline_tests.py)
**Artifacts**: [multilayer_ensemble_test.json](multilayer_ensemble_test.json), [probe_seed_sensitivity.json](probe_seed_sensitivity.json)

---

## Motivation

The 4-seed value-warmup PPO sweep (s1, s2, s3, s4) was decisively negative
at the canonical SB-Bench eval (mean Δacc = **−0.90 pp**, 95% CI
[−2.61, +0.81] pp). The reward-causality diagnostic showed reward is causal
*within-run* (ρ = +0.95) but uncorrelated *cross-seed* (Pearson ≈ +0.08).
This sets up two upstream questions before any further GPU spend:

1. **Is the L13 single-layer signal too narrow?** Can multi-layer ensembling
   of the top window {L17, L21, L25, L29, L33} produce a wider, more
   per-axis-uniform reward without introducing scale pathologies?
2. **Is the probe head itself a source of seed noise?** All 8 PPO runs used
   the same saved L13 head, but if the head is materially seed-sensitive
   then any retraining could amplify the lottery.

Both tests run offline on the cached VLBias hidden states
(`Phase0.8/a3_results/local_cache/base_probe_hs.npz`, n=900, 37 layers,
D=2048) — no GPU needed, no PPO retraining.

---

## Test A — multi-layer ensemble feasibility

### Setup
- For each layer L ∈ {1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35}: per-record
  reward `r_L[i] = hs[i, L, :] · (coef_L / ||coef_L||)` (unit-normed head,
  matches PPO convention).
- Ensemble window: top-5 = {L17, L21, L25, L29, L33}.
- Pooling variants: `raw_mean`, `zscore_mean` (per-layer μ=0, σ=1 then mean),
  `rank_mean`, `z_max`.
- Metric: Δμ_corr = μ(neg::correct) − μ(non_neg::correct) and Cohen-d on the
  same cells. Per-axis Cohen-d over 10 BBQ axes.

### Single-layer baseline (reproduces §4½.14)

| Layer | Δμ_corr | Cohen-d | pooled_sd |
|:--:|--:|--:|--:|
| L1 | +0.260 | +1.36 | 0.191 |
| L5 | +0.365 | +2.54 | 0.144 |
| L9 | +0.505 | +3.44 | 0.147 |
| L11 | +0.589 | +3.72 | 0.158 |
| **L13 (current head)** | **+0.621** | **+3.70** | **0.168** |
| L17 | +0.724 | +4.06 | 0.178 |
| L21 | +1.431 | +5.68 | 0.252 |
| L25 | +5.890 | +6.29 | 0.937 |
| L29 | +6.452 | +6.53 | 0.988 |
| L33 | +7.671 | +6.39 | 1.201 |
| L35 | +8.410 | +6.96 | 1.208 |

Reward magnitude inflates ~14× from L13 → L35 in raw units — confirms a
raw-mean would be dominated by deeper layers.

### Variants

| Variant | Δμ_corr | Cohen-d | pooled_sd | axes pos / 10 |
|:--|--:|--:|--:|:--:|
| single L13 | +0.621 | +3.70 | 0.168 | 10/10 |
| single L21 | +1.431 | +5.68 | 0.252 | 10/10 |
| single L25 | +5.890 | +6.29 | 0.937 | 10/10 |
| ensemble raw_mean | +4.434 | +6.97 | 0.636 | 10/10 |
| **ensemble zscore_mean** | **+1.586** | **+7.21** | **0.220** | **10/10** |
| ensemble rank_mean | +165.140 | +4.26 | 38.748 | 10/10 |
| ensemble z_max | +1.362 | +4.59 | 0.297 | 10/10 |

`zscore_mean` is the only variant that:
- Strictly dominates every single-layer baseline on **Cohen-d**.
- Keeps the reward in a sensible scale (no rank-bloat).
- Is positive on **all 10 BBQ axes**.

### Per-axis Cohen-d — L13 vs L21 vs zscore_mean

| Axis | L13 | L21 | zscore_mean |
|:--|--:|--:|--:|
| Age | +3.44 | +4.82 | +6.02 |
| Disability_status | +4.34 | +7.14 | **+10.07** |
| Gender_identity | +5.42 | +7.84 | +9.59 |
| Nationality | +3.88 | +6.33 | +6.62 |
| Physical_appearance | +2.35 | +4.77 | **+7.74** |
| Race_ethnicity | +3.30 | +4.70 | +6.54 |
| Race_x_SES | +2.67 | +5.07 | +7.58 |
| Race_x_gender | +3.60 | +5.06 | +6.12 |
| Religion | +4.93 | +5.75 | +6.09 |
| SES | +3.99 | +7.06 | **+9.49** |
| **mean** | **+3.79** | **+5.85** | **+7.59** |
| **std** | 0.90 | 1.10 | 1.51 |

### Layer-layer per-record Pearson (top window)

|   | L13 | L17 | L21 | L25 | L29 | L33 | L35 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| L13 | +1.00 | +0.98 | +0.95 | +0.88 | +0.88 | +0.88 | +0.87 |
| L17 | | +1.00 | +0.95 | +0.88 | +0.89 | +0.89 | +0.87 |
| L21 | | | +1.00 | +0.93 | +0.93 | +0.93 | +0.92 |
| L25 | | | | +1.00 | +0.98 | +0.97 | +0.96 |

### Verdict

**Multi-layer ensemble is worth promoting — but as variance reduction +
scale normalisation, not as new-information fusion.** Top-window layers are
0.87–0.99 correlated per-record, so the ensemble is mostly averaging out
per-layer noise; the apparent Cohen-d gain (3.70 → 7.21) is real but
inflated by the train-set evaluation (we trained these probes on the same
records). Per-axis uniformity is the more robust win: worst-axis Cohen-d
goes from +2.35 (L13/Physical) → +6.02 (zmean/Age).

Recommended config: `generated_heads_probe_L17-33_zmean_base_biasA/`,
using the 5 probes [L17, L21, L25, L29, L33] with `zscore_mean` pooling
implemented at reward-computation time.

---

## Test B — probe-fit seed sensitivity (L13 bias_aligned)

### Setup
- 80% bootstrap × 8 seeds {0, 1, 2, 7, 13, 42, 99, 2026} of the n=600
  L13 bias_aligned training pool.
- `LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')`
  (lbfgs, same as the saved probe).
- Reference: fit on the FULL pool → cos vs saved head = **+0.99992** ✓
  (saved coef is deterministic given the train set).

### Per-seed metrics

| Seed | cos(coef, ref) | holdout_acc | Δμ_corr | Cohen-d | \|\|coef\|\| |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.923 | 0.925 | +0.632 | +3.14 | 6.77 |
| 1 | +0.917 | 0.866 | +0.632 | +3.05 | 6.71 |
| 2 | +0.946 | 0.910 | +0.612 | +3.27 | 6.73 |
| 7 | +0.952 | 0.910 | +0.624 | +3.20 | 6.72 |
| 13 | +0.940 | 0.896 | +0.633 | +3.11 | 6.62 |
| 42 | +0.920 | 0.896 | +0.643 | +2.90 | 6.46 |
| 99 | +0.933 | 0.896 | +0.632 | +3.23 | 6.66 |
| 2026 | +0.944 | 0.925 | +0.616 | +3.14 | 6.69 |
| **mean** | **+0.934** | **0.903** | **+0.628** | **+3.13** | **6.67** |
| **std** | 0.013 | 0.020 | 0.010 (1.5%) | 0.11 (3.5%) | 0.10 |

### Cross-seed cosine matrix

Off-diagonal: **mean = +0.875**, std = 0.021, min = +0.81, max = +0.92.

### Verdict

**The probe direction is mildly seed-sensitive (cos≈0.87 between bootstrap
refits) but the downstream-relevant signal is stable to within ~3%.**
- Δμ_corr varies by 0.01 (1.5%) across seeds.
- Cohen-d varies by 0.11 (3.5%) across seeds.
- All 8 PPO runs used the *same* saved L13 head (deterministic lbfgs fit),
  so probe-fit randomness contributed **exactly zero** PPO variance.

**Probe is not the source of the seed lottery.** The PPO inter-seed variance
must come from one of:
1. The PPO loss-landscape lottery itself.
2. KL anchoring drift (s1 tail KL = 0.044 is a 50× outlier vs others).
3. The narrow reward range (r_task ∈ [−0.124, −0.110] across all 8 runs).

---

## Next investigation

Before promoting the `zmean-top5` ensemble, investigate **(2)**: pull
per-step KL trajectories from the saved s1 vs s2/s3/s4 vwarmup checkpoints,
identify when/where s1 diverges, and characterise whether the 50× tail-KL
spike correlates with any particular batch or reward distribution. The
ensemble promotion is queued behind that diagnostic.
