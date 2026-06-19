# Phase 0.8 — Wash-out Diagnostic Verdict (KLFIX-s1)

**Status:** Action 2 closed.
**Inputs:** `Phase0.8/washout_diagnostic_klfix.json`, `Phase0.8/washout_scores/phase08_2k_klfix_s1__probe_L*__offline_reward.json`
**Method:** Strategic Plan §3 — per-layer probe reward Δμ_corr = mean(reward | correct) − mean(reward | incorrect) on `phase08_2k_klfix_s1` (KLFIX-stabilised PPO, seed 1) versus base, across L ∈ {1,5,9,11,13,17,21,25,29,33,35}, normalised to vanilla σ.

## Verdict
**Pattern 2 (wash-out)** — same family as production phase08_2k, biasOnly, and corrOnly. KLFIX-s1 is the **flattest** variant of the four (max |z| = 0.019σ at L17). No layer crosses the §3 thresholds for Pattern 1 or 3.

| variant                | pattern | max \|z\| | argmax L |
|------------------------|---------|----------|----------|
| phase08_2k (prod)      | Pattern 2 | 0.032 | 11 |
| phase08_2k_corrOnly    | Pattern 2 | 0.075 | 25 |
| phase08_2k_biasOnly    | Pattern 2 | 0.021 | 25 |
| **phase08_2k_klfix_s1**| **Pattern 2** | **0.019** | **17** |

All z-scores are |z| < 0.10σ at every probed layer — well below §3's Pattern 1 cut-off (|z| ≥ 0.30σ at L13).

## Per-layer (klfix-s1 − base) / σ_van
```
L 1  +0.008    L13 +0.016    L25 -0.009
L 5  +0.015    L17 +0.019    L29 -0.013
L 9  +0.012    L21 +0.019    L33 -0.006
L11  +0.016                  L35 -0.000
```
Pattern: marginal positive drift at L1–L21 (≤+0.019σ), marginal negative drift at L25–L33 (down to −0.013σ), zero at L35. Direction is consistent with a **very slight** weakening of deep-layer bias-axis projection, but the magnitudes are noise-floor.

## Interpretation
1. **KLFIX successfully stabilises PPO** (canonical n=2916: +1.08pp ± 0.35pp, seed-replicable) but **does NOT propagate the L13 reward signal into the representation**. Same bottleneck as prod.
2. **Cause is signal sparsity, not training instability.** The reward head at L13 is a single 2048-d projection; gradients reaching the residual stream are too weak to shift the deep-layer bias-axis distribution measurably. KLFIX's value-head + clipping + KL-adapt machinery removes the *noise floor* but doesn't add *signal*.
3. **The +1.08pp accuracy gain is real** — it comes from policy-level letter-selection shifts and minor logit recalibration, not from representational debiasing. Consistent with the bias_score asymmetry (|bias| reduced 43%, Gender_identity still +0.40 residual).

## Implication for next step
Validates the §4 Action 1 thesis in `Phase0.8/NEXT_STEPS.md`: **the bottleneck is reward-signal coverage, not stability.** Multi-layer z-score-mean ensemble reward across L ∈ {9, 11, 13, 17, 21} is the right next experiment. KLFIX should be kept as the training backbone (it is the lowest-variance variant) and stacked with the ensemble reward.

## Pre-committed gate for proceeding to Action 1
Now satisfied:
- ✅ KLFIX-s1 wash-out classified
- ✅ Pattern remains 2 (predicted; matches §3 hypothesis)
- ✅ No Pattern-1 or Pattern-3 signal emerged that would change strategy

**Recommendation:** proceed to Action 1 (multi-layer ensemble reward) under KLFIX backbone.
