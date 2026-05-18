# Debiasing Methodology Analysis & SB-Bench Experiment Log

## 1. Comparative Results: POPE Benchmark

| Metric | Vanilla Qwen2.5-VL-3B | Debiased (PPO, 1 epoch) | Δ |
|---|---|---|---|
| Accuracy | 87.18% | 87.19% | **+0.01%** |
| Precision | 85.43% | 85.36% | -0.07% |
| Recall | 89.64% | 89.78% | +0.14% |
| F1 | 87.49% | 87.51% | **+0.02%** |
| Yes Proportion | 51.94% | 52.07% | +0.13% |
| Unknown Rate | 0.69% | 0.69% | 0.00% |
| True Positives | 4034 | 4040 | +6 |
| True Negatives | 3812 | 3807 | -5 |

**Conclusion**: Statistically indistinguishable. The debiasing had near-zero effect on the POPE hallucination benchmark.

---

## 2. Root Cause Analysis — Why The Debiasing Failed

The SB-Bench evaluation confirms that the debiasing methodology failed to adapt the model even *in-domain*. The near-zero delta on both POPE and SB-Bench indicates a fundamental **optimization failure** during PPO training.

### Problem 1 — Reward Has No Output-Level Grounding (Most Critical for In-Domain Failure)
- `R_curr = FastRL(w_k · e_EOS)` measures the projection of the representation embedding onto bias directions.
- The model can trivially satisfy this continuous reward by shifting its internal representation slightly without ever changing the discrete `argmax` token output (the actual A/B/C answer).
- There is no scalar `+1 / -1` reward grounded in whether the final generated token is correct. The model is optimizing its hidden state, not its text output.

### Problem 2 — LoRA Capacity and Insufficient Training
- `kl_beta = 0.1` imposes a strict KL penalty preventing the policy from drifting from the reference model.
- With only `r=16` and 1 epoch, the LoRA adapter barely moves from its random initialization.
- The resulting model functions identically to the vanilla base model.

### Problem 3 — CAA Weight Collapses to Uniform Early in Training
- When the active policy is still nearly identical to the reference policy, the per-sample causal difference `||Δr||₂ ≈ 0`.
- The min-max normalization (`w_min ≈ w_max`) triggers the `allclose` guard, setting `w_hat = ones`.
- The causal weighting mechanism becomes a no-op exactly when it is needed most (early training).

### Problem 4 — Train/Test Domain Mismatch (Generalization Issue)
- Even if the SB-Bench in-domain training succeeded, generalizing to POPE would be difficult.
- **DRM heads** track social stereotyping (age, gender), while **POPE** tests visual object hallucination. These are geometrically orthogonal bias dimensions in representation space.

### Problem 5 — Low-Rank PCA Reward (50 components / 2048 dims)
- 50 components span only ~2.4% of the representation space. The model easily finds trivial directions to maximize reward.

---

## 3. SB-Bench Evaluation Experiment

> **Goal**: Evaluate the debiased model vs. vanilla baseline on the SB-Bench dataset itself — the same dataset used to train the DRM reward heads. If debiasing worked at all, we should see improvement here (in-domain performance).

### Setup
- **Dataset**: `ucf-crcv/SB-Bench` — 9 bias categories, 3-way MCQ
- **Prompt format**: `{context} {question}\nA) ans0\nB) ans1\nC) ans2\nAnswer with only the letter (A, B, or C):`
- **Metric**: Accuracy = fraction of questions where model selected the non-stereotypical (label) answer
- **Output files**:
  - Debiased: `/mnt/data/output_ppo_debiased/checkpoint-ep1-end/sb_bench_generations.jsonl`
  - Vanilla: `/mnt/data/sb_bench_vanilla_baseline/sb_bench_generations.jsonl`

### Commands

**Step 1 — Generate debiased model answers on SB-Bench (A100 GPU, ~30–40 min):**
```bash
.venv/bin/modal run run_modal.py --phase evaluation --dataset sb_bench
```

**Step 2 — Generate vanilla baseline answers on SB-Bench (A100 GPU, ~30–40 min):**
```bash
.venv/bin/modal run run_modal.py --phase evaluation --dataset sb_bench --vanilla
```

> Both commands auto-run generation if no `--gen-file` is provided, then immediately evaluate.

**To re-run evaluation only on already-generated files:**
```bash
# Debiased:
.venv/bin/modal run run_modal.py --phase evaluation --dataset sb_bench \
  --gen-file /mnt/data/output_ppo_debiased/checkpoint-ep1-end/sb_bench_generations.jsonl

# Vanilla:
.venv/bin/modal run run_modal.py --phase evaluation --dataset sb_bench --vanilla \
  --gen-file /mnt/data/sb_bench_vanilla_baseline/sb_bench_generations.jsonl
```

---

## 4. Expected Findings & Interpretation Guide

| Scenario | Interpretation |
|---|---|
| Debiased >> Vanilla on SB-Bench | DRM reward heads worked in-domain — fine-tuning moved the model toward fair answers on the training distribution. Need to extend to POPE domain. |
| Debiased ≈ Vanilla on SB-Bench | DRM heads captured the embedding shift but the LoRA adapter has insufficient capacity or epochs to translate this to output-level changes. Need more training. |
| Debiased < Vanilla on SB-Bench | Negative transfer — PPO destabilized the model. KL penalty may be too weak, or reward heads are noisy. |

---

## 5. Refined Improvement Roadmap (Planned for `stage_v1_enhanced_debiasing`)

Given the confirmed in-domain optimization failure, the priority must shift from "improving the bias features" to **"fixing the RL optimization loop itself."**

| Priority | Improvement | Rationale |
|---|---|---|
| 🔴 P0 | **Direct Outcome-Grounded Reward** | Implement a `+1/-1` binary reward based on whether the model actually outputs the correct, fair answer token. Do not rely solely on representation-space DRM rewards. |
| 🔴 P0 | **Multi-epoch training (3-5 ep) + KL annealing** | Anneal KL penalty `0.3 → 0.1 → 0.03` and increase LoRA rank (`r=32` or `r=64`) to give the adapter enough capacity to learn. |
| 🟡 P1 | **Rank-based CAA weighting** | Replace fragile min-max normalization with ordinal ranks to ensure robust gradient scaling even when policy ≈ reference. |
| 🟡 P1 | **POPE-targeted DRM heads** | To generalize to hallucination tasks, the PCA contrastive pairs must be extracted from POPE (correct vs. hallucinated). |
| 🟢 P2 | **In-loop Evaluation Callbacks** | Track the actual accuracy on a subset of data every N steps, rather than waiting for the end of the epoch. |
| 🟢 P2 | **Merge adapter before final eval** | Eliminates LoRA overhead drift at inference time. |

---

## 6. Results Log

### SB-Bench Accuracy — Debiased vs. Vanilla

| Category | Vanilla | Debiased | Δ |
|---|---|---|---|
| Age | 64.19% | 64.23% | +0.04% |
| Disability | 63.96% | 63.71% | -0.25% |
| Gender | 51.03% | 50.48% | -0.55% |
| Nationality | 70.53% | 70.15% | -0.38% |
| Physical Appearance | 64.91% | 65.02% | +0.11% |
| Race/Ethnicity | 64.05% | 64.77% | +0.72% |
| Religion | 81.07% | 81.36% | +0.29% |
| SES | 84.85% | 85.28% | +0.43% |
| Sexual Orientation | 76.62% | 75.79% | -0.83% |
| **Overall** | **65.14%** | **65.10%** | **-0.04%** |

### Interpretation
The debiased model performed **identically** to the vanilla baseline, even on the *exact same dataset it was trained on* (SB-Bench). This confirms that the lack of improvement on POPE was not merely due to domain mismatch (though that is still a factor). The core issue is an **optimization failure**: the PPO algorithm completely failed to update the model to prefer non-stereotypical answers in its text generation.
