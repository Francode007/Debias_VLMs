# Phase 0.9 Strategic Plan

> **Live document.** Updated as experiments land. **Last updated**: 2026-06-24.
> **Branch**: `phase0.9_multilayer_ensemble_reward`
> **HEAD at writing**: `cd68767`
>
> **Companions**
> - [PROJECT_STATE.md](PROJECT_STATE.md) — live navigator (read this first in a new conversation)
> - [Phase0.9_Critical_Review.md](Phase0.9_Critical_Review.md) — analytical justification for every ablation below
> - [Phase0.9/FINAL_REPORT.md](Phase0.9/FINAL_REPORT.md) — R1–R3 closure report (the result this plan builds on)
> - [Phase0.9/STRATEGIC_PLAN_SUMMARY.md](Phase0.9/STRATEGIC_PLAN_SUMMARY.md) — live experiment log; update after every run
> - [Phase0.8/Multi_layer_head_analysis.md](Phase0.8/Multi_layer_head_analysis.md) — per-layer Δμ_corr, Cohen-d, ambig signature (single source for layer-choice arguments)

---

## §0. Status snapshot

| | |
|---|---|
| **Phase 0.9 R1–R3** | ✅ closed. Verdict: ensemble reward partially vindicated (better OOD bias-reduction, no ID gain, costs 1.4 pp ambig erosion). See [Phase0.9/FINAL_REPORT.md](Phase0.9/FINAL_REPORT.md). |
| **Phase 0.9.5 (this plan)** | 🔄 opening. Eight pre-committed ablations to map the design space around R3 before committing to R4. |
| **R4 (CF n=6166)** | ⏸ deferred to post-0.9.5 (run with the winning config). |
| **Workshop submission** | Phase 0.9 R1–R3 already integrated into [Workshop_Submission/REPORT_EXTENSIVE.md](Workshop_Submission/REPORT_EXTENSIVE.md) §9 + §10.1. Phase 0.9.5 results will go in a new §10.x update. |
| **Spend budget** | $50 cap for all of Phase 0.9.5; expected ~$15–25. R4 budget separate (~$10). |

---

## §1. Live ablation table

Three priority tiers (P0 highest). Costs are s1-only; expand to 4 seeds only if s1 passes the green-light rule (§2).

| ID | Title | Tests | Cost s1 | Cost 4-seed | Priority | Status |
|---|---|---|---|---|---|---|
| **E1** | Inference-time L13 erasure baseline | Whether PPO is the right primary at all | $0.50 | n/a (one-shot) | **P0** | not-started |
| **E2** | Probe-direction convergence diagnostic | Whether n=900 probes are underfit | $1.00 | n/a (one-shot) | **P0** | not-started |
| E3 | Add L13 to ensemble + zmedian pool `{L13,L17,L21,L25,L29,L33}` | L13 compensation hypothesis + robust pool | $1.50 | $6.00 | P1 | not-started |
| E4 | Targeted LoRA L17–25 + winning ensemble | Whether constrained LoRA outperforms distributed | $1.50 | $6.00 | P1 (conditional) | not-started |
| E5 | Disentangled-regime-only ensemble `{L13,L17,L21}` + zmedian | Does the deep convergent regime contribute OOD? | $1.50 | $6.00 | P2 | not-started |
| E6 | Agreement-pool `{L13,...,L33}` + agreement_mean K=4/6 | Confidence-gated reward (first-significant-layer principle) | $1.50 | $6.00 | P2 | not-started |
| E7 | Re-fit probes at n > 900, rebuild bundle | Only triggered if E2 says cos(w_900, w_max) < 0.90 | $2.00 | n/a (one-shot) | conditional | not-started |
| **R4** | CF flip-rate n=6166 with winning config | Causal-evidence gate | n/a | $3 + ½-day build | post-0.9.5 | deferred |

Detailed experiment specs are in §3.

---

## §2. Sequencing + decision rules

### §2.1 Standing rule — s1-first

For every ablation that produces a PPO sweep:

1. Fire `SEEDS="1" bash scripts/...` only.
2. Wait for s1 to finish + pull metrics.
3. **Green-light gate**: SB-Bench canonical Δ vs vanilla ≥ +1.0 pp AND VLBias |bias_score| reduction vs vanilla ≥ +0.001.
4. If green → fire `SEEDS="2 3 4"`. Pull all gens. Aggregate.
5. If red → log the s1 result in [Phase0.9/STRATEGIC_PLAN_SUMMARY.md](Phase0.9/STRATEGIC_PLAN_SUMMARY.md), explain why, drop the ablation.

This is the **only** way to fan out to 4 seeds. No exceptions.

### §2.2 Decision tree across ablations

```
START
  │
  ├── E1: Inference-time L13 erasure baseline (free)
  │     │
  │     ├── If E1 beats PPO+KLFIX on either ID or OOD →
  │     │      MAJOR FINDING. Stop the PPO ablation chain; pivot to inference-
  │     │      time evaluation. Update strategic plan; write a separate report.
  │     │
  │     └── If E1 < PPO+KLFIX on both →
  │            PPO is doing real work. Continue.
  │
  ├── E2: Probe-direction convergence test (free)
  │     │
  │     ├── If cos(w_900, w_max) ≥ 0.95 →
  │     │      Probes are converged. Continue with existing bundle.
  │     │
  │     └── If cos(w_900, w_max) < 0.90 →
  │            Probes underfit. Run E7 first to rebuild bundle.
  │            All downstream ablations use the new bundle.
  │
  ├── E3: Add L13 + zmedian (P1)
  │     │
  │     ├── s1 green → fan to 4 seeds → if 4-seed clears §4 closure gates,
  │     │      WINNING CONFIG. Run R4. Stop.
  │     │
  │     └── s1 red →
  │            Log + drop. Move to E4 / E5 / E6 (in that order).
  │
  ├── E4: Targeted LoRA L17–25 + best ensemble so far (P1)
  │     │
  │     ├── s1 green → fan; compare to E3.
  │     └── s1 red → log + drop.
  │
  ├── E5: Disentangled-regime-only `{L13,L17,L21}` (P2)
  │     │
  │     ├── s1 green → fan; informative even if not best (tells us deep regime is unnecessary).
  │     └── s1 red → log + drop.
  │
  ├── E6: Agreement-pool K=4/6 (P2)
  │     │
  │     ├── s1 green → fan; compare to E3/E4/E5.
  │     └── s1 red → log + drop.
  │
  └── ANY WINNING CONFIG → R4 with winning config; then Phase 0.9.5 closes.
       NO WINNING CONFIG → publish R3 as the final result with the ablation
                           grid as the negative-result evidence; skip R4.
```

### §2.3 Parallelism

- E1 and E2 are independent and zero-Modal-quota; run them concurrently.
- E3 can start as soon as E2 lands (or sooner if you accept the existing bundle pending E2).
- E4 is conditional on E3 (uses E3's winning config); cannot start before E3 4-seed lands.
- E5 and E6 are independent of E3/E4 (different ensemble configs); can be queued without waiting.

---

## §3. Experiment specs

### §3.1 E1 — Inference-time L13 erasure baseline

**Hypothesis.** If the L13 `bias_aligned` direction is the right "bias subspace," projecting it out at inference time should match (or beat) the entire PPO+KLFIX pipeline — a $0.50 baseline that may invalidate $50+ of PPO work.

**Method.**
1. Load vanilla Qwen2.5-VL-3B-Instruct.
2. Inject a forward-hook at `model.layers[13]` that does:
   ```python
   h_post = h_pre - (h_pre @ v_bias.unsqueeze(-1)) * v_bias  # v_bias unit-normed (1, D)
   ```
3. Score canonical SB-Bench n=2916 + VLBias n=2000.

**Two variants**:
- E1a: erase L13's `bias_aligned` direction only.
- E1b: erase at `{L13, L17, L21, L25}` simultaneously (multi-layer erasure).

**Files to touch**: a new ~80-line script `scripts/phase09_inference_erasure.py` that wraps the base model with a hook and re-uses `generate_sb_bench_answers.py` / `generate_vlbiasbench_answers.py` as inner generators. No code changes to the trainer or PEFT path.

**Pass criterion** (informational): is E1 vs vanilla > R3 vs vanilla on either ID or OOD? If yes, the PPO investment is in question.

**Expected cost**: $0.50 (no GPU training, just two gen runs at A100).

**Outputs**: `Phase0.9/erasure/E1a_sbbench_gen.jsonl`, `E1a_vlbias_gen.jsonl`, `E1b_*` likewise.

### §3.2 E2 — Probe-direction convergence diagnostic

**Hypothesis.** With n=900 training records and D=2048 features, the probe is under-determined; the L2 regularisation picks the minimum-‖w‖² direction from a 1148-dim null-space. If `cos(w_900, w_3000) ≥ 0.95`, the regularisation has converged and we can trust the existing bundle. If `cos(w_900, w_3000) < 0.90`, we need to retrain at higher n.

**Method.**
1. Pull the larger VLBiasBench gen JSONL (the source the existing n=900 cache was subsampled from — it has 3000+ records).
2. Re-extract HS at L13 (and optionally L17, L21, L25, L29, L33) at full size + multiple subsample sizes.
3. Fit `bias_aligned` LR (C=1.0, 5-fold CV) at each n ∈ {600, 900, 1500, 3000, max}.
4. Compute pairwise cosines `cos(w_n_i, w_n_j)` and report convergence rate.

**Files to touch**: a new ~120-line script `scripts/phase09_probe_convergence.py` that reads the cache + computes sub-sampled fits. Maybe one Modal job to extract the bigger HS cache if the existing one doesn't have enough records.

**Pass criterion**:
- `cos(w_900, w_max) ≥ 0.95` at L13 → existing probes are stable; lock the bundle.
- `cos(w_900, w_max) < 0.90` at L13 → trigger E7.
- Intermediate (0.90 ≤ cos < 0.95) → soft trigger; consult and decide.

**Expected cost**: $1 (one A100 hour to extract a larger HS cache if needed; otherwise CPU only).

**Outputs**: `Phase0.9/probe_convergence/diagnostic.json`.

### §3.3 E3 — `{L13, L17, L21, L25, L29, L33}` + zmedian pool

**Hypothesis.** Adding L13 to the ensemble restores the ambig-preservation signal (R3's only OOD downside). Switching pool to median makes the aggregate robust to the deep-layer outlier values on ambig records.

**Method.**
1. Build a 6-layer bundle including L13: pull `base_L13_probe_weights_biasA.npz` (already on volume), re-run `scripts/phase09_build_ensemble_bundle.py` with `WINDOW = [13, 17, 21, 25, 29, 33]`.
2. Push to `/generated_heads_probe_L13-33_zmedian_base_biasA/` on the Modal volume.
3. Add `zmedian` to the pool options in `phase08_ppo_trainer.py` (currently supports `zmean / mean / max`; need to add `median` via `torch.median(per_layer_z, dim=1)`).
4. Run s1 with the new bundle + `--ensemble-pool zmedian`.
5. Apply §2.1 green-light rule.

**Files to touch**:
- `scripts/phase09_build_ensemble_bundle.py`: parameterise the `WINDOW` constant via CLI arg (currently hard-coded).
- `src/modules/training/drm_loader.py` `ENSEMBLE_SUPPORTED_POOLS`: add `"zmedian"`.
- `src/modules/rl_components/phase08_ppo_trainer.py`: in the pool dispatch, add `elif self.ensemble_pool == "zmedian": pooled = per_layer_z.median(dim=1).values`.
- New launcher `scripts/phase09_e3_zmedian_seeds.sh` (mirrors `phase09_ensemble_klfix_seeds.sh`).

**Pass criterion s1**: SB-Bench Δ ≥ +1.0 pp AND VLBias |bias_score| reduction ≥ +0.001.

**4-seed closure criterion** (§4 below).

**Expected cost**: $1.50 (s1); $6 if 4-seed runs.

### §3.4 E4 — Targeted LoRA L17–25 + winning ensemble

**Hypothesis.** Constraining LoRA to the bias-forming layers (L17–L25) forces the policy to modify the bias-forming computation rather than the post-hoc completion selection.

**Method.**
1. Use the winning config from E3 (window + pool).
2. Add a new CLI arg `--lora-layer-range "17-25"` that constrains PEFT `target_modules` to `r"model\.layers\.(1[7-9]|2[0-5])\.(self_attn|mlp)\..*"`.
3. Run s1.

**Files to touch**:
- `src/modules/training/args.py`: add `--lora-layer-range` flag.
- `src/modules/training/setup.py::build_policy_model`: parse the range, build the regex, pass to `LoraConfig(target_modules=...)`.
- New launcher `scripts/phase09_e4_targeted_lora_seeds.sh`.

**Pass criterion s1**: Δ vs vanilla ≥ E3's s1 Δ AND VLBias ambig_acc ≥ E3's s1 ambig_acc.

**Expected cost**: $1.50 (s1); $6 if 4-seed runs.

**Risk note**: PEFT layer-regex syntax needs to be verified against the actual module-name layout. Quick check: `python -c "from transformers import AutoModelForImageTextToText; m = AutoModelForImageTextToText.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct'); print([n for n,_ in m.named_modules()][:30])"`.

### §3.5 E5 — Disentangled-regime-only `{L13, L17, L21}` + zmedian

**Hypothesis.** Does the deep convergent regime (L25+) actually contribute OOD, or is the entire R3 OOD benefit attributable to the disentangled regime?

**Method.** Same pipeline as E3 with a 3-layer bundle `{L13, L17, L21}` and `zmedian` pool.

**Files to touch**: identical to E3, just different `WINDOW` arg + launcher.

**Pass criterion s1**: §2.1 rule.

**Informational value even if E3 wins**: a green E5 means the deep regime is *unnecessary*; a red E5 means the deep regime is doing real work. Either way the ablation grid sharpens.

**Expected cost**: $1.50 (s1); $6 if 4-seed runs.

### §3.6 E6 — Agreement-pool `{L13,...,L33}` + agreement_mean K=4/6

**Hypothesis.** "All-must-agree" reward shaping operationalises the first-significant-layer principle: the policy only gets reward when bias is driven to zero at *every* layer in the window.

**Method.** Add `agreement_mean` pool function to the trainer; use the 6-layer window from E3 with K=4 of 6 agreement threshold.

**Files to touch**:
- `src/modules/training/drm_loader.py` `ENSEMBLE_SUPPORTED_POOLS`: add `"agreement_mean"`.
- `src/modules/rl_components/phase08_ppo_trainer.py`: add the pool function and a new arg `--ensemble-agreement-k` (default = N//2 + 1).
- New CLI flag in `args.py`.
- New launcher.

**Pass criterion s1**: §2.1 rule.

**Expected cost**: $1.50 (s1); $6 if 4-seed runs.

### §3.7 E7 — Probe re-fit at n > 900 (conditional on E2)

**Trigger**: E2 returns `cos(w_900, w_max) < 0.90` at L13.

**Method.**
1. Extract HS at the larger VLBiasBench subsample (e.g., `max_per_cell=100` → n≈3000).
2. Re-fit `bias_aligned` probes at L13, L17, L21, L25, L29, L33.
3. Rebuild ensemble bundle with the new probes (μ/σ on the larger calibration set too).
4. Restart whichever ablation E3/E4/E5/E6 is most recent with the new bundle.

**Files to touch**: re-run existing pipeline with adjusted `max_per_cell`. No code changes.

**Expected cost**: $2 (one A100 hour for HS extraction + CPU for probe fit + bundle build).

### §3.8 R4 — CF flip-rate n=6166 (post-0.9.5)

Spec deferred to a separate R4 plan once 0.9.5 closes.

---

## §4. Closure gates for Phase 0.9.5

**One winning config** must satisfy all four to close Phase 0.9.5:

| Gate | Threshold | Notes |
|---|---|---|
| ID SB-Bench canonical Δ vs vanilla | ≥ +1.3 pp (n=4 seeds) | Match R3 +1.29 pp lower bound; not regressing from current Phase 0.9 R3 |
| OOD VLBias \|bias_score\| reduction vs vanilla | ≥ +0.0015 (n=3–4 seeds) | Match R3 +0.0021 lower bound |
| OOD VLBias ambig_acc within … of vanilla | ≥ −1.0 pp | Recover 1.4 pp of what R3 lost; the headline improvement target of Phase 0.9.5 |
| R4 CF flip_rate(target) | ≤ 0.8 × flip_rate(vanilla) | Causal-evidence gate; runs post-0.9.5 with winning config |

If **no** config satisfies all four → publish R3 as the final result with the 0.9.5 ablation grid as the negative-result frontier evidence. Skip R4.

If **at least one** config satisfies all four → that config goes through R4; the result is the new Phase 0.9.5 close-out + workshop update.

---

## §5. Timeline & checkpoints

| Slot | Activity | Cumulative spend cap |
|---|---|---|
| Day 1 morning | E1 + E2 dev + run (zero Modal training) | $1.50 |
| Day 1 afternoon | E1 + E2 verdicts; if E2 trips E7 → start E7 | $4 |
| Day 2 | E3 s1; verdict; if green → E3 4-seed + E4 s1 | $11.50 |
| Day 3 | E3 4-seed gens (SB-Bench + VLBias) + aggregate; E5 + E6 s1 | $20 |
| Day 4 | Fan-outs of E4/E5/E6 if any went green | $35 |
| Day 5 | Phase 0.9.5 close-out: write `Phase0.9.5_FINAL_REPORT.md`; pick winner; update PROJECT_STATE.md | $35 |
| Day 6+ | R4 build + sweep with winning config | $45 |

These are rough; actual times depend on Modal queue availability. The budget cap is the hard guardrail.

---

## §6. What we are explicitly NOT doing (Phase 0.9.5)

- **Re-training the L13 baseline.** Already 4-seed-validated (KLFIX +1.49 pp, +1.37 pp like-for-like). All comparisons are against this fixed reference.
- **Phase 0.10 (7B model scale-up).** Out of scope until Phase 0.9.5 + R4 close.
- **DPO/IPO replacement of PPO.** Orthogonal axis; not in Phase 0.9.5.
- **Re-doing R1 / R2.** Both closed; verdicts in [Phase0.9/FINAL_REPORT.md](Phase0.9/FINAL_REPORT.md).
- **Cross-architecture transfer (LLaVA, Idefics, etc.).** Single-model finding remains; cross-architecture is Phase 0.10+.
- **Adding more probe tasks (`is_C`, `correct`).** Already characterised in [Phase0.8/Multi_layer_head_analysis.md](Phase0.8/Multi_layer_head_analysis.md) §4½.10; `bias_aligned` is the established primary reward.

---

## §7. Risks & contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| Modal infra crashes (cf. R3 s3 DataLoader-worker crash) | medium | Adopt single-eval-worker patch if a second seed crashes at the same step; bound retries to 2 per seed. |
| E1 (inference-time erasure) beats PPO → narrative pivot | medium-low | If it happens, frame R3 as "PPO recovers most of the inference-time benefit while remaining policy-permanent"; emphasise reversibility/deployability rather than absolute accuracy. |
| E2 trips E7 → all bundles need rebuilding | low | E7 is $2 and ½-day; absorbable. The diagnostic IS worth doing before fan-outs. |
| No config clears the closure gates | medium | Publish R3 + the ablation frontier as the final negative-result paper; this is itself a publishable contribution. |
| Targeted LoRA (E4) needs PEFT regex that doesn't work on Qwen2.5-VL | low | Inspect `model.named_modules()` first; if the regex can't be made to fit, fall back to PEFT's `target_modules` list-form. |

---

## §8. Pointers

| Topic | File |
|---|---|
| Why each ablation matters | [Phase0.9_Critical_Review.md](Phase0.9_Critical_Review.md) §1–§8 |
| R1–R3 baseline + verdict | [Phase0.9/FINAL_REPORT.md](Phase0.9/FINAL_REPORT.md) |
| Live experiment log | [Phase0.9/STRATEGIC_PLAN_SUMMARY.md](Phase0.9/STRATEGIC_PLAN_SUMMARY.md) |
| Per-layer Δμ_corr, Cohen-d, ambig signature | [Phase0.8/Multi_layer_head_analysis.md](Phase0.8/Multi_layer_head_analysis.md) §4½.14 |
| KLFIX recipe + L13 baseline aggregator | `REPORT_EXTENSIVE.md` §6 + [scripts/phase08_seed_aggregate.py](scripts/phase08_seed_aggregate.py) |
| Existing ensemble bundle + builder | [scripts/phase09_build_ensemble_bundle.py](scripts/phase09_build_ensemble_bundle.py) + [Phase0.9/ensemble_bundle/](Phase0.9/ensemble_bundle/) |
| Existing trainer ensemble path | [src/modules/rl_components/phase08_ppo_trainer.py](src/modules/rl_components/phase08_ppo_trainer.py) `is_ensemble_mode` branch |
| Ensemble launchers (template for new ones) | [scripts/phase09_ensemble_klfix_seeds.sh](scripts/phase09_ensemble_klfix_seeds.sh) + `_canonical_gens.sh` + `_vlbias_gens.sh` |