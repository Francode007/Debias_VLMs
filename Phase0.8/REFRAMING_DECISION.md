> RETRACTED 2026-06-18 — superseded by REFRAMING_DECISION_v2.md after discovering the 9-axis vs original-eval split confusion. The 'convergent null' framing was wrong; phase08_2k accuracy on 9-axis is +3.10 pp (p=1e-31), reproducing and strengthening the Critical Review headline.
# Phase 0.8 — Three-Diagnostic Verdict & Reframing Decision

**Date:** 2026-06-18
**Models:** Qwen2.5-VL-3B-Instruct, PPO fine-tuned with steering-reward at L13 / L17
**Bench:** SB-Bench 9-axis test split (3747 rows)

---

## 1. The three diagnostics

### (a) Wash-out (prior turn) — `Phase0.8/washout_diagnostic.json`
Verdict: **"inconclusive (no L13 drop)"**.
Interpretation: when we steer the PPO-trained model *back* along the L13 axis at
inference time, performance does **not** drop. → PPO did not learn to depend on
that residual direction; the policy is routing corrections through a different
("output") channel. **Tier 1 #5/#6 multi-layer LoRA gate does not fire.**

### (b) Counterfactual flip-rate — `Phase0.8/counterfactual_eval/cf_metrics.json`
6166 polarity-paired (image, context) pairs in the canonical SB-Bench; 228 of
them landed in the 9-axis test split.

| variant      |  n  | flip_rate | ±SE   | acc_orig | acc_swap | Δflip vs base |   z   |   p   |
|--------------|----:|----------:|------:|---------:|---------:|--------------:|------:|------:|
| base         | 228 |   0.3465  | 0.032 |  0.5132  |  0.4737  |       –       |   –   |   –   |
| phase08_2k   | 228 |   0.3553  | 0.032 |  0.5570  |  0.4912  |   +0.0088     | +0.20 | 0.84  |
| corrOnly     | 228 |   0.3684  | 0.032 |  0.4737  |  0.4211  |   +0.0219     | +0.49 | 0.63  |
| biasOnly     | 228 |   0.3509  | 0.032 |  0.5132  |  0.4649  |   +0.0044     | +0.10 | 0.92  |

Verdict: **PPO does NOT reduce polarity-flip rate.** All three variants are
statistically indistinguishable from base on flip. `phase08_2k` does raise
both `acc_orig` (+4.4 pp) and `acc_swap` (+1.8 pp), so the gain on the headline
benchmark is a **polarity-invariant accuracy bump**, not a bias-mitigation
effect.

### (c) Seed-triple (Tier 0 #1 + Tier 1 #4) — `Phase0.8/seed_runs/L{13,17}_aggregate.json`
Vanilla baseline: micro_acc = 0.4860, macro_acc = 0.5151.

| variant | seed | acc    | Δacc   | McNemar χ²  | p          |
|---------|-----:|-------:|-------:|------------:|-----------:|
| L13     |   1  | 0.4879 | +0.002 |   0.61      | 0.43       |
| L13     |   2  | 0.4732 | −0.013 |  36.82      | <0.001 **worse** |
| L13     |   4  | 0.4980 | +0.012 |  28.90      | <0.001 **better** |
| **L13 mean** | – | **0.4863 ± 0.0125** | **+0.0004** | – | Stouffer z = +0.05, **p = 0.96** |
| L17     |   1  | 0.4919 | +0.006 |   6.49      | 0.011      |
| L17     |   2  | 0.4751 | −0.011 |  23.88      | <0.001 **worse** |
| L17     |   4  | 0.5065 | +0.020 |  67.95      | <0.001 **better** |
| **L17 mean** | – | **0.4911 ± 0.0158** | **+0.0052** | – | Stouffer z = +3.41, **p = 6.5e-4** |

Verdict:
- **L13: NULL.** Mean Δ ≈ 0, std (1.3 pp) >> effect, Stouffer non-significant.
- **L17: fragile.** Mean Δ = +0.5 pp with std 1.6 pp (3× the effect); direction
  *flips* across seeds (s2 hurts, s4 helps). Stouffer significance is driven
  almost entirely by s4's outlier (χ²=68). The original "L17 > L13" headline
  from a single seed is within seed-variance and **not robust**.

---

## 2. Convergent story

All three diagnostics agree:

| signal needed for the Phase 0.8 thesis | observed |
|---|---|
| PPO depends on the steered direction | ✗ wash-out null |
| PPO reduces polarity sensitivity      | ✗ CF flip-rate null |
| Layer ordering L17 > L13 is real      | ✗ seed-triple noise; L13 null, L17 fragile |
| Effect transfers across seeds         | ✗ direction flips across seeds at both layers |

The single-seed "L17 +X pp" headline that motivated Phase 0.8 is a **seed-noise
artifact on top of a small polarity-invariant accuracy gain.** There is no
evidence the steering reward is doing what it was designed to do.

---

## 3. Reframing options

**Option A — Stop and re-design (recommended).**
The mechanism (steering-reward PPO) does not produce a causal-bias effect at
the budget tested. Continuing to scale or tweak is unlikely to flip the verdict;
the wash-out result already says the policy isn't routing through the steered
subspace. Pivot the next phase to a fundamentally different mechanism —
e.g. counterfactual preference pairs (DPO on polarity-flipped pairs), a CF data
augmentation regime, or a contrastive objective that *directly* targets
flip-rate.

**Option B — Reframe minimally as a negative result.**
Publish exactly what we found: "steering-reward PPO yields fragile,
seed-dependent accuracy gains and no improvement on a polarity-counterfactual
metric, suggesting steering rewards capture surface correlations rather than the
causal bias subspace they are designed to suppress."
This is honest but a much weaker contribution.

**Option C — Single-headline rescue (NOT recommended).**
Report L17_s4 as the headline and bury the seed variance. This is the result
the existing one-seed runs already produce; the new evidence shows it isn't
reproducible. Do not take this option.

---

## 4. Recommendation

**Option A.** Two follow-ups before any new training compute is spent:

1. **DPO on CF pairs.** Treat the existing 6166 polarity-paired contexts as
   preference data: prefer the answer that is *invariant* across the polarity
   flip. This trains directly against the metric that PPO+steering failed to
   move.
2. **Sanity-test the steering signal itself.** Re-derive the L13/L17 directions
   on the 9-axis data (not the 3-axis subset used originally) and check whether
   the activation-difference vector still maps to a behavior-relevant axis at
   eval time. If the direction itself doesn't correlate with bias on the larger
   axis set, the reward signal was never well-grounded.

If both of those clear, then revisit the PPO+steering architecture with the
multi-layer / output-channel insights from the wash-out result.

---

## 5. Compute footprint of this diagnostic round

- 10 SB-Bench gens × 3747 rows × ~6 min each on A100-80GB ≈ 1.0 A100-hour total
  (parallel wall time ≈ 10 min).
- No additional training was needed; reused existing checkpoints.
