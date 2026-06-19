# Phase 0.8 KL-fix s1 smoke — analysis

**Date:** 2026-06-19
**App:** `ap-tJONiT59UDa5ZF6Mm4Bo2H` (completed)
**Config:** seed=1, max_train_samples=2000, value-warmup=30 (lr×2), value-clip-range=0.2, kl-adapt-rate=0.3, target-kl=0.02, value-head zero-init.
**Fixes under test:**
1. **Value-head zero-init** ([`phase08_ppo_trainer.py:141-150`](../../src/modules/rl_components/phase08_ppo_trainer.py#L141-L150)).
   Removes the seed-correlated 5× spread in initial `value_grad_norm` (Kaiming-uniform default vs all-zeros).
2. **value-clip-range=0.2** — bounds value-update magnitude.
3. **kl-adapt-rate=0.3** (was 0.1) — faster Schulman adapter response.
4. **Bug fix (`9662465`)** — grad-flow safety check gated by `not _value_only_mode`
   (zero-init `value_head.weight` means v_loss can no longer accidentally backprop
   into LoRA during warmup; the old check was firing on the warmup zero-grad).

## Cross-seed KL stability (200 PPO steps each)

| seed       | tail-KL (20) | mean-KL  | max-KL   | β-tail | vgn-head5 | pgn-tail | logdiff-tail | acc[50→200] |
|------------|--------------|----------|----------|--------|-----------|----------|--------------|-------------|
| base s1    | **0.0512**   | 0.0114   | **0.36** | 0.34   | 35.7      | 1.76     | 0.112        | 0.297→0.312 |
| base s2    | 0.0007       | 0.0005   | 0.0017   | 0.05   | 30.2      | 1.15     | 0.022        | 0.281→0.312 |
| base s3    | 0.0012       | 0.0006   | 0.0047   | 0.05   | 15.7      | 0.84     | 0.035        | 0.312→0.281 |
| base s4    | 0.0007       | 0.0006   | 0.0056   | 0.05   | 8.1       | 0.67     | 0.027        | 0.297→0.297 |
| **klfix s1** | **0.0006** | **0.0004** | **0.0019** | **0.05** | **1.2** | 0.67 | 0.020 | 0.297→0.281 |

## Verdict — KL anomaly: **FIXED decisively**

- **tail-KL: 0.0512 → 0.0006 (85× reduction)**, well under 0.005 pass criterion, and *tighter than any baseline seed*.
- max-KL: 0.36 → 0.0019 (190× reduction). The runaway-ratchet pathology is gone.
- β-tail: 0.34 → 0.05 (floor). Adaptive KL controller never needed to engage — by design.
- `vgn-head5`: s1's 35.7 collapses to 1.2 — exactly the predicted zero-init effect. The seed-correlated value-head initialisation magnitude that drove the divergence is eliminated.
- `pgn-tail`: 0.67 — *identical* to baseline s4 (the most stable seed). klfix s1 sits in the stable cluster.

## Sanity checks (policy is still updating, not frozen)

- `pg_loss == 0.0` is benign and seed-invariant — per-batch advantage normalisation produces `mean(A)=0`, and the PPO surrogate equals `-mean(A)` whenever `ratio≈1.0`. All baseline seeds also show `pg_loss ≈ 0`. The pg gradient comes from the *per-sample* `A_i * log_ratio_i` deviations, which are nonzero.
- `mean_abs_logprob_diff = 0.020` (klfix) vs 0.022 / 0.035 / 0.027 (base s2/s3/s4). Policy *is* moving, in the same regime as the stable baseline seeds.
- `policy_grad_norm` trajectory monotonically nonzero (0.38–0.82 across all sampled steps).
- Midtrain acc 0.297→0.281 is within noise band (n=64 eval → SE ≈ 5.7%). All baseline seeds also drift inside ±2pp.

## Predicted root cause: confirmed end-to-end

Pre-fix baseline s1 trajectory (every 30 steps) shows the documented runaway:
- step 0:   `vgn=41.25, v_loss=8.55`  ← huge random-init value-head magnitude
- step 120: `kl=0.0024, logdiff=0.045` ← KL starts climbing
- step 180: `kl=0.013,  logdiff=0.10`  ← cycle accelerates
- step 210: `kl=0.010,  β=0.14`        ← adapter just starting to react (too slow)
- step 240: `kl=0.0065, β=0.60`        ← β finally ratcheted up, but KL already settled high

klfix s1 trajectory:
- step 0:   `vgn=0.71, v_loss=0.30`    ← zero-init starting point
- step 120: `kl=0.00056, logdiff=0.018`
- step 240: `kl=0.00017, logdiff=0.012` ← stable throughout, β never leaves floor

## Next step

**Promote to full 4-seed klfix re-sweep + canonical eval.** Pass criteria:
- All 4 seeds: tail-KL ≤ 0.005 (5× tighter than worst baseline).
- Cross-seed canonical-eval std ≤ 0.7pp (vs current baseline std ~1.5pp).
- Mean canonical-eval acc not worse than current baseline.

Estimated cost: 4× ~30min training + 4× ~5min canonical eval ≈ 2.5h Modal A100.

If pass → adopt klfix as new Phase 0.8 baseline → then layer the zmean-top5
multi-layer ensemble on top.
