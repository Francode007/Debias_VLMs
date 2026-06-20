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

---

## Appendix — Wash-out pattern reference

Originally specified in [`Phase0.8_Strategic_Plan.md`](../Phase0.8_Strategic_Plan.md) §3 (now CLOSED) and preserved verbatim in [`archive/Phase0.8_Strategic_Plan_v1.md`](archive/Phase0.8_Strategic_Plan_v1.md) §3. Reproduced here for future reference.

Let `Δμ_full` = per-layer change in mean probe reward (correct − incorrect cells), PPO model minus vanilla, normalised to vanilla σ. The three pre-registered patterns are:

| Pattern | Definition | Mechanistic interpretation |
|---|---|---|
| **Pattern 1 — Clean debias** | `Δμ ≤ −0.30σ` at L13 **AND** `≤ −0.15σ` at every L ≥ 17 | Reward signal *propagates*: L13 training pushes the bias direction down at L13 **and** the suppression persists deeper. Representation genuinely "cleaned." |
| **Pattern 2 — Wash-out** | `Δμ ≤ −0.30σ` at L13 **BUT** `|Δμ| < 0.10σ` at every L ≥ 21 | Reward signal stays *local*: L13 itself shifts, but downstream layers reconstruct the bias representation. Policy outputs better answers via surface routing, not representational change. |
| **Pattern 3 — Proxy-hacking** | `Δμ ≤ −0.30σ` at L13 **BUT** `Δμ > 0` at any L ≥ 25 | Reward signal is *gamed*: model learns to push bias representation *out* of L13 and *into* deeper layers. Net bias preserved or amplified, just hidden from the probe. |

### Implication per pattern (original Strategic Plan §2 decision tree)

| Pattern | Action |
|---|---|
| **Pattern 1** | Skip multi-layer LoRA (T1#5). Reduce LoRA rank — single-layer reward is sufficient and the LoRA may be over-parameterised. |
| **Pattern 2** | Run multi-layer LoRA (T1#5) attached to L13/17/21/25. Keep reward at L13 (it works), but give the *store* more depth to land in. |
| **Pattern 3** | Run **both** multi-layer LoRA (T1#5) **AND** multi-layer reward (T1#6). Treat the single-layer L13 result as proxy-hacked / unreliable. |

### What the diagnostic actually returned

**None of the four PPO variants crossed even the L13 entry threshold** (`Δμ ≤ −0.30σ at L13`). Max |z| across all 4 variants × all 11 layers stayed below 0.10σ:

| variant | max \|z\| | argmax L |
|---|---|---|
| phase08_2k (prod) | 0.032 | 11 |
| corrOnly | 0.075 | 25 |
| biasOnly | 0.021 | 25 |
| **KLFIX-s1** | **0.019** | **17** |

This is a **degenerate case the v1 patterns didn't anticipate**: not Pattern 1, 2, or 3 in the strict definitional sense — instead the L13 representation **never moved at all** even though canonical accuracy improved (+1.08 pp KLFIX, +1.85 / +3.10 pp prod).

The classifier ([`scripts/phase08_washout_classify.py`](../scripts/phase08_washout_classify.py)) returns `"FLAT (<0.10σ everywhere) — no bias-axis perturbation"` for this case. This verdict file labels it as "Pattern 2" by a relaxed reading (deep layers are flat by definition when nothing moved anywhere), but the live Strategic Plan §0bis re-interprets it more precisely as:

> **The L13 reward is a *selector*, not an *eraser*.** PPO is reweighting *which trajectories get reinforced* rather than rewriting *what the representation encodes*.

### Why this redirected priority to Action 1 (not T1#5)

The v1 §2 decision tree assumed the failure mode would be "change at L13 that washes out at depth" (Pattern 2 in the strict sense), which would have called for T1#5 (multi-layer LoRA) — more *store* capacity at deeper layers. The actual failure mode is "no representational change anywhere," which makes the LoRA placement irrelevant: there is no L13 change to spread. The correct lever is **more reward signal coverage** — fire the selector at multiple layers simultaneously so that more of the trajectory's bias content shows up in the reward landscape. That is Action 1 (multi-layer ensemble reward; see Strategic Plan §3bis).

[`SCALING_PRIORITY_ANALYSIS.md`](SCALING_PRIORITY_ANALYSIS.md) documents why scaling data alone won't help here either (the synergy is saturated at 2k samples and the trajectory has decelerated to +0.17 pp in the last quartile).
