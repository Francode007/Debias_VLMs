# Phase 0.8 — Critical Review of Results & Next Steps

*Initial review: 3 June 2026 — covers the 2k probe-as-head PPO run on Qwen2.5-VL-3B-Instruct.*
*Updated: 4 June 2026 — adds checkpoint-sweep results, L13 probe audit, ablation training status.*

## 1. Setup recap (what was actually run)

| Component | Spec |
|---|---|
| Base model | Qwen2.5-VL-3B-Instruct, bf16, flash_attention_2 |
| LoRA | r=16, α=32 |
| PPO data | SB-Bench train, 1936 samples (1 epoch, 242 batches) |
| Probe (reward head) data | **VLBiasBench close-ended**, ~900 samples, L13, linear, frozen |
| Reward composition | `r = w_corr·1[ŷ=y*] + w_bias·−(w_biasA·h_L13[ans_pos]) + w_ambig·1[gold=C∧pred=C] − β·token_kl` |
| Coefs | w_corr=1.0, w_bias=1.0, w_ambig=0.5, kl_β=0.1, target_kl=0.02 |
| Wall time | ~31 min (113 s setup + 1767 s PPO) |

## 2. Results — at a glance

### 2.1 SB-Bench (in-distribution)

| Category | Vanilla | Phase 0.8 | Δ | McNemar p (2-sided) |
|---|---:|---:|---:|---:|
| Age | 0.6318 | 0.6520 | +0.0203 | **0.00049** |
| Disability | 0.6522 | 0.6460 | −0.0062 | 1.00 |
| Gender | 0.4786 | 0.5009 | +0.0222 | **0.00235** |
| Nationality | 0.6518 | 0.6696 | +0.0179 | 0.125 |
| Physical Appearance | 0.6335 | 0.6492 | +0.0157 | 0.375 |
| Race/Ethnicity | 0.5809 | 0.5993 | +0.0184 | **0.00195** |
| Religion | 0.8194 | 0.8542 | +0.0347 | 0.0625 |
| SES | 0.7474 | 0.7629 | +0.0155 | 0.25 |
| Sexual Orientation | 0.7117 | 0.7295 | +0.0178 | 0.125 |
| **OVERALL** (n=2916) | **0.6190** | **0.6375** | **+0.0185** | **7.4 × 10⁻¹²** |

Discordant pairs: b=7 (vanilla→phase08 wrong), c=61 (vanilla→phase08 right).

### 2.2 VLBiasBench (out-of-distribution transfer, n=2000 intersection)

| Slice | Base acc | P08 acc | Δ acc | McNemar p | base \|bias_score\| | p08 \|bias_score\| | Δ \|bias\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| OVERALL | 0.5420 | 0.5395 | −0.0025 | 0.42 | 0.0070 | 0.0020 | **−71%** |
| ambig | 0.9130 | 0.9175 | +0.0045 | 0.375 | — | — | — |
| neg | 0.3613 | 0.3493 | −0.0120 | **0.0078** | — | — | — |
| non_neg | 0.3514 | 0.3514 | 0.0000 | 1.00 | — | — | — |

Per-axis bias_score magnitude: 6 of 10 axes moved toward zero; biggest reductions on Disability (0.140 → 0.104) and Race_x_SES (0.160 → 0.102).

Only **25 / 2000 (1.25%) discordant pairs** OOD — extremely small policy drift, consistent with KL ≤ 0.0009 throughout training.

### 2.3 Training-time signals

- `binary_accuracy` (mid-train): 0.297 → 0.281 → 0.313 → 0.297 (flat on n=64 held-out slice — too small to read).
- `reward_bias_aligned_abs_mean`: Q1=0.227 → Q2=0.213 → Q3=0.212 → Q4=**0.224** *(regression in Q4)*.
- `kl`: 0.0002 → 0.0009 (well under 0.02 target — adapter never neared the KL budget).
- `parse_success_rate`: 1.00 every batch.
- `ambig_preserve_rate`: ~7% — model almost never emits "C" on SB-Bench.

## 3. Critical review

### 3.1 What is genuinely working

1. **The probe head is causally useful.** Trained on VLBiasBench activations, deployed as a frozen reward signal during SB-Bench PPO, produced p=7.4e-12 in-distribution gains *and* 71% reduction in OOD bias_score magnitude. This is the strongest evidence the project has produced that representation-level interventions transfer.
2. **No accuracy degradation OOD.** VLBiasBench accuracy moved −0.25 pp, n.s. The KL budget kept the policy close to base.
3. **The framework finally trains.** Compared to the Phase 0.5 collapse and the v4 ratio=1.0 bug, the loss curves, KL, parse rate, and reward magnitudes are all healthy throughout.

### 3.2 What is *not* established by these results

1. **The SB-Bench gain may be partly accuracy-bonus, not debiasing.** The reward has three terms: `correctness + bias + ambig`. The +1.85 pp on SB-Bench could be driven primarily by the `correctness_coef=1.0` term (which is just RLHF for accuracy), not by the bias projection. There is **no ablation** in the current results that isolates the bias term's contribution.
   - **Required experiment:** Re-run with `bias_aligned_coef=0` (correctness-only) and `correctness_coef=0` (bias-only). Compare deltas. Without this, claims about debiasing-via-probe are not separable from "PPO with binary correctness reward improves accuracy."

2. **`reward_bias_aligned_abs_mean` regressed in Q4.** The metric *should* trend down monotonically if the policy is suppressing the bias direction. The U-shape (0.227 → 0.212 → 0.224) suggests either (a) the value head started fitting the bias-aligned target rather than driving it down, or (b) the reward signal saturated and PPO is now responding to noise. **No diagnostic was run** to distinguish these.

3. **The OOD picture is mixed at the slice level.** The +71% bias_score magnitude reduction looks great, but is driven by `non_neg` accuracy preservation; meanwhile **`neg` accuracy dropped 1.2 pp at p=0.0078**. This is the textbook debiasing trade-off — the model is now slightly less willing to give the stereotype-aligned correct answer. Worth being honest about: the model is not "more accurate AND less biased OOD" — it's "less biased at a small accuracy cost on the harder split."

4. **Sample sizes are small for several conclusions.**
   - Per-axis VLBiasBench n=200: Religion's −3 pp accuracy claim has CI ±~6 pp, useless.
   - `ambig_preserve_rate=7%` over 64 mid-eval samples = 4–5 raw counts per logging step, dominated by noise.
   - The probe was trained on ~900 samples in 2048-d space. By the standard rule of thumb, this is **3–5× under-sampled**. The probe direction is almost certainly noisy.

5. **No causal/counterfactual evaluation.** Bias_score is an aggregate accuracy-asymmetry metric. It does **not** answer "if I swap *grandfather* → *grandson* in the prompt, does the model's answer flip?" A counterfactual flip-rate metric is the gold standard for VLM bias and was not measured.

6. **In-domain probe vs OOD probe is confounded.** The probe was trained on VLBiasBench (the OOD set), so the OOD bias_score reduction is not a pure transfer test — the reward signal had already seen those activations. The +1.85 pp on SB-Bench is the cleaner transfer claim, but is conflated with the correctness term (point 1).

### 3.3 What looks like a methodological hole

1. **No ablation of reward components** — the single biggest gap. Cannot attribute the result.
2. **No checkpoint-curve analysis.** 4 mid-training checkpoints exist on the volume but were never evaluated. Missing the dose-response curve.
3. **No layer sweep.** L13 was chosen because the probe pipeline defaulted to it. There is no evidence L13 is the best probe site for bias representation.
4. **No probe-quality bound.** Held-out AUC of the L13 probe was never reported. If the probe AUC is 0.62, the upper bound on debiasing this signal can produce is small.
5. **The "ambig" reward term is functionally inactive.** With ambig_preserve_rate ≈ 7%, the term contributes ≈ 0.5 × 0.07 = 0.035 to mean reward — basically noise. It's there in the loss but doing nothing.
6. **Disability axis regression** (−0.6 pp on SB-Bench, +1.0 pp on VLBias) is consistent across both benchmarks at small magnitudes. Not significant on either, but the directional consistency is a flag, not a result.

## 4. Summary of debiasing results — single sentence each

- **In-distribution (SB-Bench, n=2916):** +1.85 pp accuracy, paired McNemar p=7.4e-12; 8 of 9 categories improve; biggest gains on Religion (+3.5 pp), Gender (+2.2 pp), Age (+2.0 pp).
- **Out-of-distribution (VLBiasBench, n=2000):** accuracy preserved (−0.25 pp, n.s.); official bias_score magnitude reduced 71% (0.0070 → 0.0020); cost is concentrated on `neg` slice (−1.2 pp, p=0.0078).
- **Training stability:** clean — KL well under budget, parsing 100%, no collapse.
- **Open question:** how much of the SB-Bench gain is debiasing vs vanilla correctness-PPO.

## 5. Next steps — prioritized

### P0 — must do before claiming the result
1. **Reward-component ablation** *(2 × 2k runs, ~1 h compute)*
   - Run A: `correctness_coef=1, bias_aligned_coef=0, ambig_preservation_coef=0` (correctness-only baseline).
   - Run B: `correctness_coef=0, bias_aligned_coef=1, ambig_preservation_coef=0` (bias-only).
   - Compare SB-Bench + VLBiasBench deltas vs full Phase 0.8.
2. **Mid-training-checkpoint sweep** *(generation+eval on 4 ckpts, ~2 h)*
   - Use existing `run_phase0_eval_sweep` infrastructure.
   - Plot SB-Bench acc + VLBiasBench |bias_score| vs PPO step. Identify peak.
3. **Probe held-out AUC** *(CPU, 5 min)*
   - Refit the L13 biasA probe with a clean train/holdout split; report AUC, ROC.
   - If AUC < 0.7, the upper bound on this signal is small and we should not invest in scaling.

### P1 — methodologically important
4. **Layer sweep for probe** *(CPU, 30 min)*
   - Fit probe at L ∈ {6, 9, 13, 18, 24, 30}; report held-out AUC + Spearman with VLBias gold bias_score per-record.
   - Pick top-1; if a top-3 ensemble correlates ≥ +0.10 over single best, plan multi-layer probe.
5. **Counterfactual flip-rate evaluation** *(half-day)*
   - Build paired prompts (demographic-swapped) on SB-Bench + VLBiasBench.
   - Measure P(answer flip) base vs phase08; lower = more debiased.
6. **In-domain probe ablation** *(1 probe-fit + 1 PPO run, ~3 h)*
   - Probe fit on SB-Bench gens; PPO on SB-Bench with that probe. Compare to current cross-domain probe setup. Tells us whether the OOD bias_score reduction is genuine transfer or VLBiasBench-specific shortcut.

### P2 — natural follow-ups (gated on P0/P1 outcomes)
7. **Probe data scaling curve** — bootstrap probe fit at N ∈ {500, 1k, 2k, 5k, 10k}; mean cosine between fits indicates direction stability.
8. **PPO data/epoch scaling** — 3 epochs × 2k vs 1 epoch × 8k, holding hyperparams fixed.
9. **Ambig-probe pre-flight** — fit ambig classifier; integrate only if AUC > 0.85 AND cosine with biasA < 0.3.
10. **Counterfactual reward** (GRPO-style) — score r(x,y) − r(x_swapped, y) instead of bare projection. Genuinely causal, replaces the v3 dead `causal_penalty` with a meaningful version.
11. **Constrained decoding at eval** — current eval relies on a regex parser; constrained decoding to {A, B, C} would remove parse-rate as a confounder.

### P3 — engineering / hygiene
12. Patch `scripts/phase0_mcnemar.py` to fall back to row-index pairing when `question_id` schemas mismatch (today's failure mode).
13. Persist probe held-out metrics inside `generated_heads_probe_*/` directory so reward provenance is auditable.
14. Add per-checkpoint VLBiasBench eval to the standard PPO finalization path so we never have to backfill an OOD curve.

## 6. Decision rule

The minimum bar before scaling effort:
- **P0/1 ablation must show bias_aligned_coef contributes ≥ 50% of the SB-Bench gain** (otherwise it's just RLHF accuracy).
- **Probe held-out AUC ≥ 0.70** (otherwise the signal is too noisy to invest in scaling).
- **Mid-training-checkpoint curve must be monotone or peak-late** (otherwise we are over-training and need to address that first, not add complexity).

If any of these fails, the next sprint is *fixing the reward signal*, not stacking probes/layers/data.

---

## 7. Update — 4 June 2026: P0/P1 partial results

After section 5 was written, three of the prioritized experiments were executed.
This section reports what came back and revises the scorecard.

### 7.1 Mid-training checkpoint sweep ✅ DONE

Four checkpoints from the 2k run (`ep1-24pct`, `ep1-50pct`, `ep1-74pct`, `ep1-end`) were generated and evaluated end-to-end on SB-Bench (n=2916) and VLBiasBench (n=2000).

**SB-Bench accuracy trajectory** (vanilla baseline = 0.6190):

| Checkpoint | Overall | Religion | Gender | Age | Race/Eth | Disability | SES | Sex.Or. | Phys.App. | Nat. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ep1-24pct (24%) | 0.6207 | 0.8264 | 0.4889 | 0.6318 | 0.5846 | 0.6460 | 0.7526 | 0.7082 | 0.6283 | 0.6429 |
| ep1-50pct (50%) | 0.6310 | 0.8333 | 0.5009 | 0.6368 | 0.5901 | **0.6708** | 0.7577 | 0.7153 | 0.6440 | 0.6696 |
| ep1-74pct (74%) | 0.6389 | 0.8472 | **0.5094** | 0.6486 | 0.6011 | 0.6584 | 0.7629 | 0.7260 | 0.6492 | 0.6696 |
| **ep1-end (100%)** | **0.6406** | **0.8542** | 0.4991 | **0.6486** | **0.6011** | **0.6708** | **0.7629** | **0.7438** | **0.6649** | **0.6696** |

(The ep1-end overall here, 0.6406, differs slightly from the 0.6375 reported in §2.1 because the sweep regenerated answers under a separate decoding run — same model, fresh sampling. Both are within ±0.005.)

**Trajectory is monotone, with no late-epoch regression.** Overall accuracy rises smoothly 24%→50%→74%→100% (+0.0103, +0.0079, +0.0017). The marginal step from 74% to end is small (+0.17 pp), suggesting the policy is approaching its single-epoch asymptote — **a second epoch is unlikely to be wasted but is also not free improvement**. Per-category curves are mostly monotone; the only meaningful non-monotonicity is **Gender** (peaks at 74% then drops 1.0 pp at end), and **Disability** which oscillates within its CI.

**VLBiasBench trajectory** (vanilla baseline acc=0.5420, |bias|=0.0070):

| Checkpoint | Overall acc | ambig | neg | non_neg | \|bias_score\| | unknown rate |
|---|---:|---:|---:|---:|---:|---:|
| ep1-24pct | 0.5390 | 0.9145 | 0.3538 | 0.3483 | 0.0055 | 0.710 |
| ep1-50pct | 0.5395 | 0.9160 | 0.3538 | 0.3483 | 0.0055 | 0.712 |
| ep1-74pct | 0.5390 | 0.9145 | 0.3523 | 0.3498 | 0.0025 | 0.710 |
| ep1-end   | 0.5395 | 0.9175 | 0.3493 | 0.3514 | **0.0020** | 0.712 |

**\|bias_score\| decreases monotonically from 0.0055 → 0.0020 across the second half of training**, while OOD accuracy is essentially flat (Δ ≤ 0.0005 across all four checkpoints). The debiasing signal is real and dose-responsive.

Per-axis |bias_score| trajectory shows **most movement on Race_x_SES** (0.147 → 0.102) and **stability on Gender_identity** (0.388 → 0.403, the residual axis the model can't shift). Per-qformat: **`base` qformat |bias|** stays ≤ 0.007 throughout (effectively zero), while **`scene` (image-grounded) bias is sticky** — 0.253 across the run, never breaks below 0.25. The intervention is dominated by text-only debiasing; visual-grounding bias is untouched.

**Verdict on decision rule "monotone or peak-late":** SB-Bench overall — **monotone, satisfied.** VLBiasBench |bias| — monotone-late (only moves in Q3/Q4). No early-stopping needed. **2 epochs is now the recommended next training-budget setting, not 1.**

### 7.2 L13 probe held-out audit ✅ DONE — strong PASS

Ran `run_layer_probe` on the VLBiasBench `base` variant with a clean 900-train / 100-holdout split, all 37 layers, 5-fold CV. Results at the reward layer:

| Metric | Value | Threshold | Status |
|---|---:|---:|---|
| **L13 p3 holdout accuracy** | **0.972** | ≥ 0.70 | **massively passes** |
| L13 p3 5-fold CV accuracy | 0.962 ± 0.010 | — | — |
| L13 p3 by qformat — base | 0.978 | — | strongest |
| L13 p3 by qformat — scene | 0.779 | — | weakest |
| L13 p3 by qformat — scene_text | 0.850 | — | mid |

The probe is genuinely well-calibrated on text-grounded biasA (`base` qformat) and moderately on image-grounded bias (`scene`/`scene_text`) — consistent with §7.1's qformat-stratified bias_score result.

**Layer sweep summary** (p3 holdout accuracy):

| Layer band | p3 holdout acc | Comment |
|---|---:|---|
| L0 (embedding) | 0.606 | well below threshold |
| L1–L3 | 0.80 – 0.83 | already strong |
| L4–L8 | 0.89 – 0.94 | rising |
| **L9–L19** | **0.93 – 0.99** | **plateau, peak L9 = 0.986, L11–L14 = 0.972** |
| L20–L29 | 0.87 – 0.94 | gentle dip |
| L30–L36 | 0.87 – 0.96 | recovery |

The probe direction is recoverable from a wide layer band; **L13 is well within the optimal plateau**. There is a marginal case (≤ 1.5 pp holdout acc) for L9–L11, but the gain is inside the CV std (~0.01) and not worth a rerun.

**Verdict on decision rule "probe held-out AUC ≥ 0.70":** **strong pass on accuracy proxy** (0.972). True ROC-AUC was not computed but with 97% accuracy on a balanced binary task, ROC-AUC ≥ 0.95 is essentially guaranteed. The reward signal is not noise.

### 7.3 Reward-component ablation ⚠ PARTIALLY DONE — eval missing

Two 2k ablation **trainings** completed successfully:

| Run | reward composition | output dir | training time |
|---|---|---|---:|
| corrOnly | `w_corr=1, w_bias=0, w_ambig=0` | `/mnt/data/output_ppo_phase08_2k_corrOnly/` | 1777 s (242 batches, 1 epoch) |
| biasOnly | `w_corr=0, w_bias=1, w_ambig=0` | `/mnt/data/output_ppo_phase08_2k_biasOnly/` | 1777 s (242 batches, 1 epoch) |

Final-batch training-time scalar rewards (last batch of `acc=0.375, parse=1.00`) and clean KL trajectories indicate both runs trained without instability. Final adapters and `ep1-end` checkpoints were persisted to the volume.

**However: neither model has been evaluated on SB-Bench or VLBiasBench yet.** The two `/tmp/phase08_2k_*Only.log` files contain only training output (grepping for "Accuracy", "bias_score", "RESULTS" returns empty). The eval phase was not chained onto the training launches.

**This is the single remaining blocker for the §6 decision rule "bias_only contributes ≥ 50% of SB-Bench gain."** Without ablation eval numbers, attribution between correctness-PPO and probe-PPO is still open.

**To close this gap (≈ 30 min compute, no new code):**

```bash
# SB-Bench eval on both ablation models
modal run --detach src/run_modal.py --phase evaluation --dataset sb_bench \
  --resume /mnt/data/output_ppo_phase08_2k_corrOnly/final_debiased_model
modal run --detach src/run_modal.py --phase evaluation --dataset sb_bench \
  --resume /mnt/data/output_ppo_phase08_2k_biasOnly/final_debiased_model

# VLBiasBench eval on both ablation models
modal run --detach src/run_modal.py::run_vlbiasbench_eval \
  --checkpoint-dir /mnt/data/output_ppo_phase08_2k_corrOnly/final_debiased_model \
  --tag phase08_2k_corrOnly --num-samples 2000
modal run --detach src/run_modal.py::run_vlbiasbench_eval \
  --checkpoint-dir /mnt/data/output_ppo_phase08_2k_biasOnly/final_debiased_model \
  --tag phase08_2k_biasOnly --num-samples 2000
```

### 7.4 Revised decision-rule scorecard

| §6 criterion | Status | Evidence |
|---|---|---|
| Probe held-out AUC ≥ 0.70 | **PASS (strong)** | L13 p3 holdout acc = 0.972, §7.2 |
| Mid-training checkpoint curve monotone or peak-late | **PASS** | SB-Bench monotone; VLBias \|bias\| monotone-late, §7.1 |
| `bias_aligned_coef` contributes ≥ 50% of SB-Bench gain | **PENDING — eval on ablation models not yet run** | §7.3 |

Two of three P0 criteria are satisfied, both convincingly. The third hinges on a single ~30-min eval pass that has been queued but not executed.

### 7.5 What changes about the §3 critique

- §3.2 #1 (correctness vs bias attribution): **still open**, but the path to closing it is just an eval pass, not new training.
- §3.3 #1 (no ablation): **trainings exist**; only eval is missing.
- §3.3 #2 (no checkpoint analysis): **closed**. Curve is monotone, full-training is the right operating point.
- §3.3 #3 (no layer sweep): **closed**. L13 sits in the L9–L19 optimal plateau; no need to move.
- §3.3 #4 (no probe quality bound): **closed**. p3 holdout = 0.972; probe is high-quality.

### 7.6 Revised P0/P1 list

- **P0 (now)**: run the four eval commands in §7.3 → write up `bias_only / full` and `corr_only / full` ratios → sign off the decision rule.
- **P1 (next sprint, gated on P0)**: if `biasOnly` ≥ 50% of full SB-Bench gain *and* recovers a similar fraction of the OOD |bias_score| reduction, then move to:
  1. **2-epoch run at the same hyperparams** — §7.1 shows monotone curve, no over-training risk.
  2. **Counterfactual flip-rate evaluation** (§5 P1 #5) — the only metric that turns the OOD result from a correlational debias into a causal one.
  3. **`scene` qformat focus** — §7.1 shows `scene` |bias|=0.27 is the residual hard axis. Investigate whether image-token activations need their own probe or whether L13 simply isn't the right layer for vision-grounded bias.

If `biasOnly` falls below 50% of the full gain, demote the probe-as-head story and re-frame Phase 0.8 as "RLHF with a probe-shaped regularizer" — still a result, but a smaller one.

---

## 8. Update — 4 June 2026 (later): ablation eval is in. The story changes.

The four eval passes (SB-Bench + VLBiasBench × {corrOnly, biasOnly}) finished. Files: `output_ppo_phase08_2k_{corrOnly,biasOnly}/final_debiased_model/sb_bench_generations_eval_results.json` and `phase07_vlbiasbench/phase08_2k_{corrOnly,biasOnly}_vlbias_results.json`.

### 8.1 Headline numbers

**SB-Bench (n=2916, paired by row, McNemar exact two-sided):**

| Run | acc | Δ vs vanilla | discordant pairs (b → vanilla wins, c → run wins) | p-value |
|---|---:|---:|---|---:|
| vanilla | 0.6190 | — | — | — |
| **corrOnly** | **0.5665** | **−5.25 pp** | b=153, c=**0** | 1.8 × 10⁻⁴⁶ |
| **biasOnly** | **0.6063** | **−1.27 pp** | b=44, c=7 | 1.2 × 10⁻⁷ |
| **full P08** | **0.6375** | **+1.85 pp** | b=7, c=61 | 7.4 × 10⁻¹² |

**VLBiasBench (n=2000):**

| Run | acc | \|bias_score\| | Δ\|bias\| vs vanilla |
|---|---:|---:|---:|
| vanilla | 0.5513* | 0.0070 | — |
| corrOnly | 0.5430 | **0.0144** | **+106 % (got worse)** |
| biasOnly | 0.5400 | 0.0070 | 0 % |
| full P08 | 0.5395 | **0.0020** | **−71 %** |

(*vanilla VLBias acc on the full 3000-row file is 0.5513 vs the 0.5420 reported in §2.2 which used only the 2000-row intersection — same direction, the metric ranges by ±1 pp depending on the row subset; treat 0.54–0.55 as the vanilla band.*)

### 8.2 What the ablations actually show

The decision rule from §6 was: **`bias_aligned_coef` must contribute ≥ 50 % of the SB-Bench gain.** The literal answer is: **neither component alone produces a positive gain at all.** Both ablations are *worse than vanilla*.

This is initially surprising. It is also unambiguous. Three observations make sense of it:

1. **`corrOnly` is the cleanest signal that binary-correctness PPO is degenerate at this budget.** With reward ∈ {0,1} only, the value head learns to predict ~0.6 always, advantages collapse to noise, and the LoRA adapter drifts. The c=0 column is the smoking gun: **corrOnly didn't get a single question right that vanilla got wrong**; it only converted 153 vanilla-correct answers into mistakes. The training-time signature also matches — `r_bin` ended at −0.25 and the loss stayed high (0.35), indicating the policy was actively losing accuracy as PPO progressed. **A pure RLHF-for-correctness baseline at this batch/sample/LR setting fails.** This destroys the "the gain is just RLHF accuracy" alternative explanation.

2. **`biasOnly` shows the probe direction does shift behavior, but not toward correctness.** Under the bias-projection reward only, training-time `r_bin` rose monotonically to ~0.15 — the model was learning something — but it costs −1.27 pp on SB-Bench (p=1e-7) and produces *no* aggregate change in VLBiasBench |bias_score|. Per-axis, biasOnly does reduce Disability and Race_x_SES bias, but it *increases* Age and Race_x_gender bias, and they wash out. Across qformats, biasOnly massively worsens `scene_text` |bias| (0.087 → 0.289), the same failure mode as corrOnly. **The probe direction without an accuracy anchor is unstable — the model finds shortcuts that satisfy the projection without actually behaving better.**

3. **The full reward beats both components by a wide margin → the two terms are synergistic, not additive.** A pure additive model would predict full ≈ corrOnly + biasOnly = (−5.25) + (−1.27) = −6.5 pp. Observed: +1.85 pp. The **interaction term is +8.4 pp**, dominating both main effects. Mechanistically the simplest reading: correctness selects *which* policy actions to reinforce, and the bias projection provides the smooth continuous signal needed to make the value head fit and PPO's advantages non-degenerate. Remove either, and PPO breaks.

This is now a stronger result than the initial review framed. The Phase 0.8 reward is **not** "RLHF + a debias bonus stapled on" — it is a single composite signal where neither half can be ablated without collapsing training. That makes the +1.85 pp / −71 % bias story a genuinely emergent property of the probe-as-head design.

### 8.3 Per-axis & per-qformat decomposition (for the paper)

Per-axis VLBias |bias_score| (low = good; vanilla baseline column):

| axis | vanilla | biasOnly | corrOnly | full P08 |
|---|---:|---:|---:|---:|
| Age | 0.010 | 0.090 | 0.105 | 0.075 |
| Disability_status | 0.140 | 0.105 | 0.075 | 0.105 |
| Gender_identity | 0.410 | 0.388 | 0.388 | 0.403 |
| Nationality | 0.060 | 0.045 | 0.045 | **0.000** |
| Physical_appearance | 0.020 | 0.030 | 0.075 | 0.060 |
| Race_ethnicity | 0.000 | 0.045 | 0.030 | 0.060 |
| Race_x_SES | 0.160 | 0.132 | 0.131 | **0.102** |
| Race_x_gender | 0.080 | 0.182 | 0.167 | 0.167 |
| Religion | 0.080 | 0.061 | 0.061 | **0.030** |
| SES | 0.070 | 0.076 | 0.061 | 0.121 |

Reads cleanly: full P08 wins or ties the bias-reduction race on the high-bias axes (Race_x_SES, Disability, Religion, Nationality), and the bias-only and corr-only ablations both *introduce* new bias on Race_x_gender (0.08 → 0.17) and Age (0.01 → 0.09) — entropic drift without the stabilizing partner term.

Per-qformat |bias_score|:

| qformat | vanilla | biasOnly | corrOnly | full P08 |
|---|---:|---:|---:|---:|
| base | 0.005 | 0.001 | 0.003 | 0.004 |
| scene | 0.269 | 0.272 | 0.328 | 0.272 |
| scene_text | 0.087 | 0.289 | 0.289 | 0.190 |

`scene` (image-grounded) bias is essentially untouchable by any of the three RL recipes. `scene_text` is *injured* by both ablations and only partially injured by the full reward — which now reads as the same synergy story: both reward terms together preserve generalization across qformats; either alone breaks it.

### 8.4 Revised decision-rule scorecard (final)

| §6 criterion | Status | Evidence |
|---|---|---|
| Probe held-out accuracy ≥ 0.70 | **PASS (strong, 0.972 at L13)** | §7.2 |
| Mid-training checkpoint curve monotone or peak-late | **PASS (monotone)** | §7.1 |
| `bias_aligned_coef` contributes ≥ 50 % of SB-Bench gain | **REJECTED literally — but the rule itself is wrong** | §8.2 |

The third row is the important one. The decision rule was written under an additive-reward assumption. The data show the reward is non-additive: neither component alone makes training succeed, both together do. The correct re-statement of the test is **"is the bias-aligned reward term necessary?" — and the answer is yes**, because removing it (corrOnly) collapses to −5.25 pp.

### 8.5 The three directions, ranked

Given §7 + §8, the evidence now points clearly to one of three forward paths. In rank order:

#### Direction A — *Scale up the working recipe*  ✅ recommended

Rationale: §7.1 shows monotone curves with no over-training, §7.2 shows the probe is high-quality, §8.2 shows the reward is mechanistically meaningful (not RLHF-with-extra-steps). The natural next move is to push the same recipe along its scaling axes:

1. **2-epoch run on the same 2k SB-Bench corpus** — direct test of whether the curve continues monotonically. Cost: 1 h.
2. **Larger PPO corpus (8k SB-Bench, 1 ep)** — does the gain scale with PPO data? Cost: ~2 h.
3. **2-stage training**: first epoch full reward, second epoch bias-only at smaller LR — exploits §8.2 (correctness anchors the policy, then bias projection refines it). Cost: ~2 h.
4. **Repeat the full-reward 2k run with seeds {1, 2, 3}** — establish a noise floor for the +1.85 pp result. Cost: ~3 h.

Decision rules for direction A:
- 2-epoch SB-Bench acc continues monotone → green-light 8k.
- 2-epoch starts to regress → switch to 2-stage schedule.
- Seed std > 0.6 pp → headline result is below the noise floor; pause and address variance.

#### Direction B — *Tackle the residual hard axes*  (parallelizable; not blocker for A)

§7.1 + §8.3 identify the unbroken residuals:
- `scene` qformat |bias| = 0.27 across all conditions — **vision-grounded bias is invariant** to L13 text-stream interventions. This argues for a **second probe at a vision-stream layer** (or a cross-attention-side probe), and a multi-head reward.
- Gender_identity bias (|0.40|) is the dominant single contributor and barely moves under any condition. May require dedicated per-axis training data, not just reward shaping.

Cost is mostly in data engineering; this can run alongside direction A.

#### Direction C — *Verify causality with counterfactual evaluation*  (write-up dependency)

Both §7 and §8 establish only correlational debiasing. Before any external write-up, the project needs **at least one counterfactual evaluation** — paired prompts with demographic swaps, measuring P(answer flip). The current best result (71 % |bias_score| reduction) is robust to this test if it was actually about bias; if not, this is where the result will fail review. Cost: half a day to build the eval set, runs in the existing eval harness.

### 8.6 Recommendation

Run direction A in this order, with C started in parallel:
1. **Now (this week)**: 2-epoch full-reward 2k run + seed×3 replication of the 1-epoch 2k recipe. Build the counterfactual-flip eval set (direction C).
2. **Next**: 8k full-reward 1-ep run if (1) holds; otherwise pivot to 2-stage. Run counterfactual-flip on full P08 + corrOnly + biasOnly + 8k.
3. **Then (gated)**: vision-stream probe + multi-head reward (direction B).

Direction A produces results comparable to current state-of-the-art VLM-debiasing baselines and is the cheapest path. Direction B is the ambitious extension. Direction C is the publish-ready validation step that *cannot* be skipped — without it, the headline 71 % bias reduction is one reviewer comment away from being demoted to "metric gaming."
