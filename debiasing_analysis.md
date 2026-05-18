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

## 2. Root Cause Analysis — Why POPE Showed No Improvement

### Problem 1 — Train/Test Domain Mismatch *(Most Critical)*
- **DRM heads** were trained on **SB-Bench** (social stereotyping: age, gender, race, etc.)
- **POPE** tests **object hallucination** — a completely different bias dimension
- The PCA directions learned from SB-Bench `(chosen − rejected)` embeddings are geometrically orthogonal to hallucination patterns
- The PPO reward signal during training is random noise from POPE's perspective

### Problem 2 — Reward Has No Output-Level Grounding
- `R_curr = FastRL(w_k · e_EOS)` measures projection of the EOS token onto bias directions
- The model can shift this representation internally while generating identical output tokens
- There is no `+1 / -1` signal tied to whether the generated answer is correct

### Problem 3 — Only 1 Epoch + Tight KL Constraint
- `kl_beta = 0.1` prevents the policy from drifting far from the base model
- After 1 epoch, the LoRA adapter barely moved from initialization
- The debiased checkpoint is functionally very close to the vanilla model

### Problem 4 — CAA Weight Collapses to Uniform Early in Training
- When policy ≈ reference: `||Δr||₂ ≈ 0` for most samples → `w_min ≈ w_max`
- The `allclose` guard fires → `w_hat = ones` → weighting is a no-op

### Problem 5 — Low-Rank PCA Reward (50 components / 2048 dims)
- 50 components span only ~2.4% of the representation space
- The model easily satisfies the reward without changing surface output

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

## 5. Proposed Improvements (Planned for `stage_v1_enhanced_debiasing`)

| Priority | Improvement | Rationale |
|---|---|---|
| 🔴 P0 | **Direct verifiable reward on POPE** | Give PPO a clean `+1/-1` signal tied to hallucination correctness |
| 🔴 P0 | **POPE-targeted DRM heads** | Extend PCA contrastive pair extraction to POPE (correct vs. hallucinated answers) |
| 🟡 P1 | **Multi-epoch training (3 ep) + KL annealing** | `0.3 → 0.1 → 0.03` to allow the adapter to grow |
| 🟡 P1 | **Rank-based CAA weighting** | Replace fragile min-max with ordinal ranks for robust gradient scaling |
| 🟢 P2 | **In-loop POPE eval callback** | Track yes-proportion during training as a live bias indicator |
| 🟢 P2 | **Merge adapter before final eval** | Eliminates LoRA overhead drift at inference time |

---

## 6. Results Log *(fill in after experiments)*

### SB-Bench Accuracy — Debiased vs. Vanilla

| Category | Vanilla | Debiased | Δ |
|---|---|---|---|
| Age | — | — | — |
| Disability | — | — | — |
| Gender | — | — | — |
| Nationality | — | — | — |
| Physical Appearance | — | — | — |
| Race/Ethnicity | — | — | — |
| Religion | — | — | — |
| SES | — | — | — |
| Sexual Orientation | — | — | — |
| **Overall** | **—** | **—** | **—** |

*Update this table after running the SB-Bench evaluation experiments above.*
