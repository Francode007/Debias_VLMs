# Phase 0.8 KL-fix 4-seed sweep — analysis

**Date:** 2026-06-19
**Apps:** `ap-PdwbnNMTXSlCTInJwCneUo` (s1), `ap-188m9bXlNbtPr0JVS9YHii` (s2),
`ap-sKUYobPZhvckWLpUTitU0C` (s3), `ap-bbjGZ1jBVBPNmqlVLRRnTE` (s4) — all completed.
**Config:** 4 seeds × {value-warmup=30 (lr×2), value-clip=0.2, kl-adapt=0.3,
value-head zero-init}, max_train_samples=2000, L13 reward probe.

## Cross-seed KL stability (full sweep)

| metric | s1 | s2 | s3 | s4 | mean | std | baseline mean | baseline std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tail-KL(20) | 0.00057 | 0.00095 | 0.00071 | 0.00061 | **0.00071** | **0.00017** | 0.01345 | 0.02524 |
| mean-KL | 0.00039 | 0.00054 | 0.00043 | 0.00038 | 0.00044 | 0.00007 | — | — |
| max-KL | 0.00163 | 0.00239 | 0.00149 | 0.00164 | 0.00179 | 0.00041 | — | — |
| β-tail | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.00000 | 0.115 | 0.143 |
| vgn-head5 | 1.22 | 0.55 | 0.94 | 1.53 | 1.06 | 0.42 | 22.4 | 11.9 |
| pgn-tail | 0.668 | 0.753 | 0.648 | 0.639 | **0.677** | **0.052** | 1.105 | 0.479 |
| logdiff-tail | 0.019 | 0.023 | 0.022 | 0.019 | 0.021 | 0.002 | 0.046 | 0.040 |
| midtrain acc@200 (n=64) | 0.281 | 0.281 | 0.281 | 0.281 | 0.281 | 0.000 | 0.301 | 0.014 |

## Pass criteria — all ✅

- **tail-KL ≤ 0.005 for all 4 seeds**: worst seed = 0.00095 (5.3× under limit).
- **KL spread max/min**: 1.69× (was 73× pre-fix → seed lottery on KL eliminated).
- **β-tail std**: 0.000 (KL controller never engages, by design).
- **vgn-head5 std**: 0.42 (was 11.9 → 28× tighter; zero-init mechanism confirmed at scale).
- **pgn-tail std**: 0.052 (was 0.479 → 9× tighter).

## Caveat: midtrain acc landed at exactly 18/64 for all 4 seeds

This is suspicious cross-seed coincidence. Possible explanations:
1. **Discretisation artefact** (most likely): n=64 has resolution 1/64 = 1.56pp.
   With small policy moves (logdiff ~0.02 ≈ 2% per-token logprob change), the
   greedy predictions on the fixed eval prompts converge to nearly-identical
   answer letters across seeds. The bias-aligned reward pulls all seeds toward
   the same `r_bias` surface so they all flip the same prompts.
2. **Policy converging to a degenerate distribution** (need to check via
   `midtrain_eval_pred_A/B/C`): if all 4 seeds produce nearly identical A/B/C
   probability mass at step 200, this is just KL-fix doing its job — tight
   policy = consistent eval predictions.
3. **Eval is too small to discriminate** (1.56pp resolution, SE ≈ 5.7pp).

The midtrain mean drop (0.301 → 0.281, −2pp) is within 1 SE. The proper
verdict requires the canonical n=2916 eval, which has SE ≈ 0.9pp.

## Next step: canonical n=2916 eval

```bash
bash scripts/phase08_klfix_canonical_gens.sh
# wait ~30 min, then:
for S in 1 2 3 4; do
  modal volume get debias-vlm-persistent-storage \
      /phase08_seeds/canonical_L13_s${S}_klfix_sbbench_gen.jsonl \
      Phase0.8/canonical/ --force
done
python scripts/phase08_seed_aggregate.py \
    --vanilla Phase0.8/canonical/canonical_base_sbbench_gen.jsonl \
    --variant L13_canonical_klfix_4seeds \
    --seeds Phase0.8/canonical/canonical_L13_s{1,2,3,4}_klfix_sbbench_gen.jsonl \
    --output-json Phase0.8/canonical/L13_canonical_klfix_4seeds_aggregate.json
```

Final acceptance gates:
- **Cross-seed acc std ≤ 0.7pp** (vs baseline std ≈ 1.5pp from earlier sweep).
- **Mean acc not worse than baseline** (baseline vwarmup mean was the +0.21pp tail).
- **Per-axis fairness metrics** consistent or improved.

If pass → klfix becomes the new Phase 0.8 baseline, ensemble work resumes on top.
