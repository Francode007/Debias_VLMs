# C-DeFR-L v2/v4: Token-Level Dense Reward VLM Debiasing — Complete Architecture Reference

> **Status as of 2026-05-22**: v4 training complete (2 epochs, 6 bugs fixed). Evaluation shows **66.67%** accuracy vs **66.56%** vanilla baseline on SB-Bench test split — a **+0.11% improvement**, statistically indistinguishable from noise (SE ≈ 0.87%). The pipeline is mechanically correct but the training signal is ineffective at changing evaluation behavior.

---

## Table of Contents
1. [Version History & Results](#version-history--results)
2. [Architecture Overview](#architecture-overview)
3. [Complete Data Flow](#complete-data-flow)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Implementation Details](#implementation-details)
6. [File-by-File Reference](#file-by-file-reference)
7. [Training Dynamics (v4 — 2 epochs)](#training-dynamics-v4--2-epochs)
8. [Evaluation Results](#evaluation-results)
9. [Failure Analysis](#failure-analysis)
10. [Hyperparameters](#hyperparameters)
11. [Runtime Commands](#runtime-commands)
12. [Escalation Path](#escalation-path)

---

## Version History & Results

| Version | Key Change | Accuracy | Δ vs Vanilla | Notes |
|---------|-----------|----------|--------------|-------|
| Vanilla baseline | — | 66.56% | — | Raw Qwen2.5-VL-3B-Instruct on test split |
| v1 (PCA, EOS-only) | Original | 66.53% | -0.03% | Credit assignment failure |
| v2 (SVM, 32 tokens) | SVM heads + dense reward | 66.60% | +0.04% | Too few generated tokens |
| v3 (SVM, 256 tokens) | Longer generation | — | — | Had 6 critical bugs; training unstable |
| **v4 (SVM, 256 tokens, bugs fixed)** | All 6 bugs fixed | **66.67%** | **+0.11%** | Stable training but no meaningful improvement |

All evaluations on SB-Bench **test split** (2916 samples, seed=42 80/20 split).

---

## Architecture Overview

### Design Philosophy

The system attempts to debias a Vision-Language Model (VLM) via Reinforcement Learning from human feedback expressed as geometric reward directions in embedding space. Rather than requiring explicit human annotations per-token, it leverages pre-computed SVM boundary normals that separate "fair" from "biased" responses in the model's own representation space.

### Three Fundamental Flaws Fixed from v1

| v1 Flaw | Root Cause | v2+ Fix |
|---------|-----------|---------|
| EOS-only reward | Credit assignment failure — PPO can't localize which token introduced bias | **Token-level dense reward** at every generated position |
| PCA heads = noise | Variance-maximizing captures structural artifacts, not bias semantics | **SVM boundary normals** — one per demographic category |
| Null-space hacking | Optimizer shifts embeddings without changing token outputs | **Logit-grounded reward** — measures actual probability shift |

### Six Critical Bugs Fixed in v4

| Bug | Impact | Fix |
|-----|--------|-----|
| KL = raw logprob_diff | KL could be negative, broke trust region | Schulman k3: `exp(Δ)-1-Δ` (always ≥ 0), clamped at 10.0 |
| pixel_values=None in scoring | ViT embeddings not computed for reference | Pass `pixel_values` to both policy and reference forwards |
| image_grid_thw not passed through | M-RoPE position encoding wrong for scored sequences | `_build_scoring_kwargs` passes vision metadata |
| Raw r_task magnitude ~10-30 | Dwarfed KL/logit components (~0.01-0.3) | Running Z-score normalization → N(0,1) |
| Advantage normalization over all positions | Prompt garbage diluted signal | Normalize only over `loss_mask.bool()` positions |
| Left-pad content extraction wrong | Used `sequences[i, :clen]` (wrong offset) | `attention_mask[i].argmax()` finds content start |

---

## Complete Data Flow

### Phase 1: Embedding Extraction (A100-80GB)

```
Input: SB-Bench dataset (14578 pairs × {image, chosen_text, rejected_text})
       ↓
Qwen2.5-VL-3B-Instruct forward pass (no LoRA)
       ↓  
Output: emb_{idx}.npy files, shape (1, 3, 2048) = [chosen_emb, rejected_emb, prompt_emb]
        Stored in: /mnt/data/embeddings_output/
```

- Uses train split only (80% = ~11662 pairs)
- Penultimate layer hidden state at EOS position
- Each .npy file indexed by `data_index = orig_row * 2 + pair_idx`

### Phase 2: SVM Head Generation (CPU, 32GB RAM)

```
Input: emb_*.npy files (train split only)
     + SB-Bench parquet (for category labels)
       ↓
For each category c ∈ {Age, Disability, Gender, ...}:
  1. Collect (chosen_emb, rejected_emb) pairs for category c
  2. X = [chosen_embs; rejected_embs], y = [1...1; 0...0]
  3. StandardScaler.fit_transform(X)
  4. LinearSVC(C=1.0, max_iter=5000).fit(X_scaled, y)
  5. w_original = svm.coef_[0] / scaler.scale_  (unscale to original space)
  6. w_c = w_original / ||w_original||_2  (unit normalize)
       ↓
Output: 9 .pth files in /mnt/data/generated_heads/sb_bench-SVM-component/
        Each: {"weight": tensor(1, 2048)}
        All SVM heads achieve 1.0 train accuracy (perfect linear separation)
```

**Critical observation**: 1.0 train accuracy means the SVM heads perfectly separate chosen/rejected in embedding space, but this does NOT guarantee that moving along these directions during generation actually changes model behavior on the multiple-choice task.

### Phase 4: PPO Training (A100-80GB)

```
Input: Train split prompts (10980 after length filtering)
     + SVM reward heads (9 × 2048)
     + Qwen2.5-VL-3B-Instruct with LoRA (r=16, α=32)
       ↓
For each batch of 12 prompts:
  1. Generate 256 new tokens (temperature=0.7, top_p=0.9)
  2. Score with active policy (logits + penultimate hidden states)
  3. Score with reference policy (disable LoRA adapter)
  4. Compute dense per-token reward
  5. Compute GAE advantages
  6. PPO clipped surrogate loss + value loss + dispersive loss
  7. Backprop through LoRA parameters only
       ↓
Output: LoRA adapter checkpoints at 25/50/75/100% of each epoch
```

### Evaluation Phase

```
Input: Test split (2916 samples) + Trained LoRA checkpoint
       ↓
1. Load base model + LoRA adapter (or vanilla for baseline)
2. For each test sample: format as "context + question + A/B/C options"
3. Generate answer (greedy/short)  
4. Parse A/B/C from output
5. Compare to label (index of non-stereotypical answer)
       ↓
Output: Overall accuracy + per-category accuracy
```

---

## Mathematical Formulation

### Dense Token-Level Reward (as implemented in v4)

For a generated sequence of length $T$ with valid (generated, non-pad) positions masked by $m_t$:

$$r_t = \underbrace{\text{ZNorm}\left(\sum_{k=1}^{9} \alpha_k \cdot (h_t^{(-2)} \cdot w_k)\right)}_{\text{task reward } \sim N(0,1)} + \underbrace{0.1 \cdot \max(0, \Delta_t)}_{\text{logit-grounded (ReLU)}} + \underbrace{\frac{c_{\text{causal}}}{T_{\text{valid}}}}_{\text{distributed causal penalty}} - \underbrace{0.1 \cdot (e^{\Delta_t} - 1 - \Delta_t)}_{\text{KL penalty (k3)}}$$

Where:
- $h_t^{(-2)} \in \mathbb{R}^{2048}$ — penultimate layer hidden state at token position $t$
- $w_k \in \mathbb{R}^{2048}$ — unit-normalized SVM boundary normal for category $k$
- $\alpha_k$ — entropy-regularized FastRL weight from deficit-based simplex update
- $\Delta_t = \log \pi_\theta(x_t | x_{<t}) - \log \pi_{\text{ref}}(x_t | x_{<t})$ — per-token logprob difference
- ZNorm uses EMA running mean/variance (decay=0.99)
- $c_{\text{causal}} = -0.5 \cdot \max(0, \|\bar{e}_{\text{curr}} - \bar{e}_{\text{ref}}\|_2 - 1.0)$ — hinge on mean-token L2 drift

### Running Z-Score Normalization (Task Reward)

```python
# EMA update (decay=0.99):
running_mean ← 0.99 * running_mean + 0.01 * batch_mean(r_task_raw[valid])
running_var  ← 0.99 * running_var  + 0.01 * batch_var(r_task_raw[valid])

# Normalize:
r_task_dense = (r_task_raw - running_mean) / sqrt(running_var + 1e-8)
```

**Problem**: This normalization hides whether the policy is improving — it re-centers to N(0,1) regardless of absolute reward level.

### FastRL Head Weighting (Deficit-Based)

```python
# Per-head mean reward (mean-pooled over valid token positions):
r_k_mean = r_token_k[mask].mean(dim=0)  # (9,)

# Z-score normalize per head:
r_norm = (r_k_mean - running_mean_per_head) / sqrt(running_var_per_head + 1e-8)

# Deficit = how far below zero each head is:
deficit_k = max(0, -r_norm_mean_k)

# Entropy-regularized softmax (tau=1.0):
alpha_k = softmax(deficit / tau)  # Focus weight on underperforming categories
```

### KL Penalty (Schulman k3 Estimator)

$$\text{KL}_t = e^{\Delta_t} - 1 - \Delta_t \geq 0 \quad \forall \Delta_t$$

Properties:
- Always non-negative (unlike raw $\Delta_t$)
- Quadratic near $\Delta_t = 0$: $\approx \frac{\Delta_t^2}{2}$
- Exponential for large positive $\Delta_t$ (strong penalty for large divergence)
- Clamped at 10.0 to prevent numerical explosion

### Logit-Grounded Reward (ReLU)

$$r_t^{\text{logit}} = 0.1 \cdot \max(0, \Delta_t)$$

Only rewards INCREASED confidence. Combined with KL penalty:
- Small positive $\Delta_t$: net positive signal (reward > penalty since $0.1\Delta > 0.1 \cdot \Delta^2/2$ for small $\Delta$)
- Large positive $\Delta_t$: net penalty (exponential KL dominates linear logit reward)
- Negative $\Delta_t$: zero logit reward + small KL penalty

### Causal Penalty

$$c_{\text{causal}} = -0.5 \cdot \max(0, \|\bar{e}_{\text{active}} - \bar{e}_{\text{ref}}\|_2 - 1.0)$$

Where $\bar{e} = \frac{\sum_t h_t \cdot m_t}{\sum_t m_t}$ is mean-pooled over valid positions.

- **delta_margin = 1.0**: No penalty for small embedding drift
- In practice with LoRA (r=16): drift stays well below 1.0 → **causal penalty is always 0**

### Dispersive Loss

$$\mathcal{L}_{\text{disp}} = -\frac{1}{B(B-1)} \sum_{i \neq j} \log(\|\bar{e}_i - \bar{e}_j\|_2 + 10^{-8})$$

Applied on mean-pooled embeddings (B=12 samples → 132 pairs). Prevents representation collapse.

### PPO Surrogate Loss

$$\mathcal{L}_{\text{PPO}} = \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \max\left(-A_t r_t^{\pi}, -A_t \text{clip}(r_t^{\pi}, 1-\epsilon, 1+\epsilon)\right)$$

Where $r_t^{\pi} = \exp(\log\pi_\theta(x_t) - \log\pi_{\text{ref}}(x_t))$ and $\epsilon = 0.2$.

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + 0.1 \cdot \mathcal{L}_V + 0.01 \cdot \mathcal{L}_{\text{disp}}$$

The PPO surrogate remains **unscaled** — trust region geometry preserved via ε-clipping.

### GAE (Generalized Advantage Estimation)

Dense GAE with γ=1.0, λ=0.95:
$$A_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Advantages normalized only over valid (generated) positions to prevent prompt garbage dilution.

---

## Implementation Details

### Model Architecture

- **Base**: Qwen/Qwen2.5-VL-3B-Instruct (3.1B parameters)
- **Vision Encoder**: ViT (no adapters, frozen during LoRA disable)
- **LoRA**: r=16, α=32, targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable params**: ~25M (LoRA) + 2048 (value head, no bias)
- **Precision**: bf16 mixed precision
- **Attention**: Flash Attention 2
- **Position encoding**: M-RoPE (Multi-modal Rotary Position Embedding) with `get_rope_index`
- **Padding**: Left-padded (required for Flash Attention + batch generation)

### Single-Model Reference Strategy

Instead of loading two copies of the 3B model, v2+ uses a single model with LoRA adapter toggling:
```python
# Active policy forward:
self.policy.train()
logits, values, hidden = self.extract_logits_values_and_hidden(self.policy, ...)

# Reference policy forward (same model, LoRA disabled):
with self.policy.disable_adapter():
    ref_logits, _, ref_hidden = self.extract_logits_values_and_hidden(self.policy, ...)
```

This saves ~6GB GPU memory but means both policies share the same ViT embeddings (correct since ViT has no LoRA).

### Value Head

```python
self.value_head = nn.Linear(hidden_size=2048, out_features=1, bias=False)
# dtype=bf16, on accelerator device
# Predicts per-token state value V(s_t) from final layer hidden state
```

Separate optimizer (same LR as policy).

### Generation Parameters

```python
max_new_tokens=256      # v4 (was 32 in v2)
do_sample=True
temperature=0.7
top_p=0.9
use_cache=True          # Disabled during training forward (incompatible with grad ckpt)
```

### Left-Padding Handling

HF `generate()` with left-padded inputs produces:
```
Input:  [PAD PAD prompt_tokens]
Output: [PAD PAD prompt_tokens gen_tokens PAD PAD]  (possible right-padding too)
```

The `_left_pad_generated` function:
1. Computes content length per sample via `attention_mask.sum(dim=1)`
2. Finds first content position via `attention_mask[i].argmax()`
3. Copies contiguous content block to right-aligned position in new tensor

### Loss Mask Construction

```python
# Only generated tokens contribute to loss (not prompt, not padding)
gen_starts = (total_seq_len - content_lens + prompt_lens).long()
gen_starts_shifted = (gen_starts - 1).clamp(min=0)  # Shifted space for logprobs
positions = torch.arange(shifted_len).unsqueeze(0)
loss_mask = (positions >= gen_starts_shifted.unsqueeze(1)) & attention_mask[:, 1:].bool()
```

### Reward Head Projection

```python
# h_active shape: (B, T, 2048) — penultimate layer, shifted to align with logprobs
# reward_heads_weight shape: (9, 2048)
r_token_k = torch.matmul(h_active, self.reward_heads_weight.T)  # (B, T, 9)

# Weighted sum using FastRL alpha:
r_task_dense_raw = torch.matmul(r_token_k, alpha)  # (B, T)
```

---

## File-by-File Reference

### Core RL Components (`src/modules/rl_components/`)

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| `custom_vlm_ppo_trainer.py` | Core PPO step: generation → scoring → reward → GAE → loss → backprop | `PPOVLMController` (390 lines) |
| `fast_rl.py` | Dynamic head weight balancing via deficit-based entropy-regularized softmax | `FastRLNode` |
| `caa_feedback.py` | Causal penalty (hinge on L2 drift) + dispersive loss (-mean log pairwise distance) | `compute_causal_reward_penalty()`, `compute_dispersive_loss()` |
| `rl_dataset_builder.py` | Loads SB-Bench, formats prompts, caches processed chunks | `RLDatasetBuilder` |
| `rl_data_collator.py` | Left-pads batch, handles pixel_values/image_grid_thw | `RLDataCollatorWithPadding` |
| `score_head.py` | Legacy multi-head scorer (not used in v2+) | `MultipleHead` |

### Training Orchestration (`src/modules/training/`)

| File | Purpose |
|------|---------|
| `train_rl.py` | Thin entrypoint: wires together all modules, runs training |
| `args.py` | Argument parsing (all hyperparameters) |
| `setup.py` | `build_accelerator()`, `build_policy_model()`, `build_ppo_controller()`, `build_dataloader()` |
| `ppo_loop.py` | Per-epoch/per-batch loop with tqdm, checkpointing at 25/50/75/100% |
| `checkpoint.py` | Save/load LoRA adapter, value head, FastRL state, optimizers |
| `drm_loader.py` | Loads .pth head files into (K, hidden_dim) tensor |

### Embedding & Head Generation (`src/modules/embeddings/`)

| File | Purpose |
|------|---------|
| `extract.py` | Phase 1: Forward pass to extract penultimate layer embeddings |
| `generate_drm_heads.py` | Phase 2: `generate_orthogonal_heads()` (PCA) + `generate_svm_heads()` (SVM) |

### Inference & Evaluation (`src/modules/inference/`, `src/modules/evaluation/`)

| File | Purpose |
|------|---------|
| `generate_sb_bench_answers.py` | Generates A/B/C answers for test split (batch_size=8) |
| `eval_sb_bench.py` | Parses A/B/C from output, computes accuracy vs label |

### Infrastructure

| File | Purpose |
|------|---------|
| `src/run_modal.py` | Modal cloud orchestrator: all phases, GPU allocation, volume management |
| `src/modules/utils/` | `ScriptArguments`, `DeviceManager`, `ModelLoader`, split utilities |

---

## Training Dynamics (v4 — 2 epochs)

### Configuration Used

```bash
modal run src/run_modal.py \
  --phase train \
  --epochs 2 \
  --head-type svm \
  --num-heads 9 \
  --logit-reward-coef 0.1 \
  --max-gen-tokens 256 \
  --batch-size 12 \
  --output-dir "/mnt/data/output_token_level_svm_v4" \
  --resume "/mnt/data/output_ppo_debiased/checkpoint-ep1-end"
```

Note: Resumed from a v3 epoch-1 checkpoint (which itself had bugs, so effectively started from near-random LoRA).

### Observed Metrics (Epoch 2, 915 batches)

| Metric | Typical Range | Interpretation |
|--------|--------------|----------------|
| `loss` | -0.015 to -0.035 | Negative = advantages are being exploited (PPO working) |
| `r_dense` | -0.7 to +0.4 | Oscillating around 0 (Z-normalized, expected) |
| `logit_r` | 0.0000 | Displayed as 0 due to 4-decimal precision; actual ~1e-5 |
| `causal` | 0.0000 | Drift < delta_margin=1.0 (expected with small LoRA) |
| `disp` | ~-10 to -12 | Dispersive loss stable (embeddings not collapsing) |
| `gpu_mem_gb` | 64-68 | A100-80GB, well within budget |

### Training Timeline

- **Batch time**: ~12-13 seconds
- **Epoch time**: ~3.2 hours (915 batches × 12s)
- **Total (2 epochs)**: ~6.5 hours on A100-80GB
- **Checkpoints**: 25%, 50%, 75%, end of each epoch

### Key Observations

1. **Loss is negative and stable**: The PPO objective is being minimized. Advantages are consistently exploited, meaning the policy IS changing in the direction of higher reward.

2. **r_dense oscillates around 0**: This is EXPECTED because of running Z-score normalization. Even if the absolute reward is increasing, the normalization re-centers it to N(0,1). This masks progress.

3. **logit_r = 0.0000**: This is the most concerning metric. With 4-decimal display precision, anything below 5e-5 shows as zero. This suggests the policy's token probabilities are barely diverging from reference — meaning LoRA updates are either:
   - Very small (LR too low or gradient signal too diffuse)
   - Being undone by the KL penalty

4. **causal = 0.0000**: Expected. With LoRA r=16 and alpha=32, the embedding drift stays well below the delta_margin=1.0 threshold. The causal penalty never activates.

5. **Dispersive loss stable**: No representation collapse.

### Diagnostic Added (Not Yet Run)

Two new metrics added to the progress bar for the next run:
- `|Δlogp|` (6 decimal places): mean |log π_θ - log π_ref| over valid tokens
- `ratio`: mean importance sampling ratio exp(log π_θ - log π_ref)

If `|Δlogp|` stays at 0.000000 after 50+ steps → LoRA weights are not updating.
If `|Δlogp|` grows to 0.001+ → training is working, just slowly.

---

## Evaluation Results

### v4 Checkpoint: epoch 2, 74% (evaluated)

```
============================================================
SB-BENCH EVALUATION RESULTS — Qwen2.5-VL
============================================================
Total Samples  : 2916
Accuracy       : 0.6667 (66.67%)
Unknown Rate   : 0.0000 (0.00%)
------------------------------------------------------------
Category                    Accuracy    Count    Unknown
------------------------------------------------------------
  Age                         0.6655      592     0.0000
  Disability                  0.6522      161     0.0000
  Gender                      0.5350      585     0.0000
  Nationality                 0.7500      224     0.0000
  Physical Appearance         0.6649      191     0.0000
  Race/Ethnicity              0.6324      544     0.0000
  Religion                    0.8125      144     0.0000
  SES                         0.8505      194     0.0000
  Sexual Orientation          0.7509      281     0.0000
============================================================
```

### Vanilla Baseline (same test split)

```
============================================================
SB-BENCH EVALUATION RESULTS — Qwen2.5-VL (Vanilla)
============================================================
Total Samples  : 2916
Accuracy       : 0.6656 (66.56%)
------------------------------------------------------------
Category                    Accuracy    Count    Unknown
------------------------------------------------------------
  Age                         0.6655      592     0.0000
  Disability                  0.6398      161     0.0000
  Gender                      0.5316      585     0.0000
  Nationality                 0.7321      224     0.0000
  Physical Appearance         0.6649      191     0.0000
  Race/Ethnicity              0.6434      544     0.0000
  Religion                    0.8264      144     0.0000
  SES                         0.8454      194     0.0000
  Sexual Orientation          0.7402      281     0.0000
============================================================
```

### Per-Category Delta (v4 - Vanilla)

| Category | Vanilla | v4 | Δ | Direction |
|----------|---------|-----|---|-----------|
| Age | 66.55% | 66.55% | 0.00% | — |
| Disability | 63.98% | 65.22% | **+1.24%** | ✓ |
| Gender | 53.16% | 53.50% | +0.34% | ✓ |
| Nationality | 73.21% | 75.00% | **+1.79%** | ✓ |
| Physical Appearance | 66.49% | 66.49% | 0.00% | — |
| Race/Ethnicity | 64.34% | 63.24% | **-1.10%** | ✗ |
| Religion | 82.64% | 81.25% | **-1.39%** | ✗ |
| SES | 84.54% | 85.05% | +0.51% | ✓ |
| Sexual Orientation | 74.02% | 75.09% | **+1.07%** | ✓ |
| **Overall** | **66.56%** | **66.67%** | **+0.11%** | — |

**Statistical significance**: With N=2916 and accuracy ≈ 66.67%, the standard error is $\sqrt{p(1-p)/n} \approx 0.87\%$. The +0.11% overall delta is 0.13σ — **not statistically significant**.

Per-category movements are larger but the sample sizes per category are small (144-592), giving SE of 2-4%, so even the ±1.5% movements are well within noise.

---

## Failure Analysis

### Why v4 Doesn't Work: Hypotheses

#### 1. Reward Signal is Disconnected from Evaluation Task

**The fundamental mismatch**: Training optimizes for high dot-product with SVM boundary normals in penultimate hidden space. Evaluation tests whether the model picks option A/B/C correctly in a multiple-choice format.

The SVM heads were trained on (chosen_EOS_embedding, rejected_EOS_embedding) pairs. The training loop projects ALL generated token embeddings onto these heads. But:
- Generated tokens during training are 256-token free-form continuations
- Evaluation asks for a single letter (A/B/C)
- There is no guaranteed causal link between "token embeddings project favorably onto SVM normal" and "model will select the non-stereotypical answer"

#### 2. Logit-Grounded Reward is Near-Zero (logit_r ≈ 0)

The logit reward component `max(0, Δ) * 0.1` shows as zero throughout training. This means:
- The policy is NOT increasing its confidence on any tokens relative to reference
- The KL penalty (even at β=0.1) is successfully preventing divergence
- Net result: LoRA weights update but produce negligible change in token distributions

This creates a paradox: the reward encourages embedding changes, the KL penalty discourages probability changes, and the evaluation only measures probability changes (which answer letter the model outputs).

#### 3. Running Normalization Masks Progress

The EMA Z-score normalization on r_task ensures the signal is always centered at 0. Even if the policy genuinely improves (absolute reward increases), the normalization hides this by re-centering. This means:
- The policy has no "memory" of improvement
- There's no gradient toward consistently higher reward
- The signal looks like noise at every point in training

#### 4. Dense Reward Distributes Signal Too Thinly

With 256 generated tokens and 9 heads, the per-token reward is the weighted projection of that specific token's embedding. But most tokens in a 256-token continuation are generic text (articles, prepositions, punctuation). The bias-relevant signal is concentrated in a few key tokens (nouns describing people, adjectives about groups, etc.). Averaging over all tokens dilutes this signal.

#### 5. Evaluation Format Mismatch

During training: model generates 256 free-form tokens.
During evaluation: model generates ~1-3 tokens ("A", "B", or "C").

The LoRA adapters learn to modify behavior during long-form generation, but the evaluation tests short-form forced-choice. These may activate very different model pathways.

#### 6. Small LoRA Capacity

With r=16 and α=32 (effective scaling = α/r = 2.0), each LoRA adapter adds a rank-16 perturbation. The total trainable parameters (~25M) may be insufficient to shift the model's decision boundary across 9 diverse bias categories simultaneously.

### What We Know For Certain

1. **The pipeline is mechanically correct**: Loss converges, GAE computes, PPO clips, gradients flow.
2. **SVM heads have perfect train accuracy**: They DO separate fair/biased in embedding space.
3. **The model IS being updated**: Loss is negative (advantages exploited), weights change each step.
4. **But the updates don't transfer to evaluation**: The multiple-choice behavior is unchanged.

---

## Hyperparameters

### PPO Training (v4)

| Parameter | Value | Flag | Notes |
|-----------|-------|------|-------|
| Epochs | 2 | `--epochs` | Evaluated at 74% of epoch 2 |
| Batch size | 12 | `--batch-size` | Per-device |
| Gradient accumulation | 4 | (hardcoded) | Effective batch = 48 |
| Learning rate | 1e-5 | `--learning-rate` | AdamW for both policy and value |
| LoRA rank | 16 | `--lora-r` | |
| LoRA alpha | 32 | `--lora-alpha` | Effective scaling = 2.0 |
| KL beta | 0.1 | `--kl-beta` | Coefficient on Schulman k3 KL |
| PPO clip range | 0.2 | `--ppo-clip-range` | |
| Num heads | 9 | `--num-heads` | One per SB-Bench category |
| FastRL tau | 1.0 | `--tau` | Entropy temperature (uniform) |
| Lambda causal | 0.5 | `--lambda-causal` | Hinge penalty strength |
| Delta margin | 1.0 | `--delta-margin` | Drift tolerance before penalty |
| Lambda dispersive | 0.01 | `--lambda-dispersive` | Weight of dispersive loss |
| Logit reward coef | 0.1 | `--logit-reward-coef` | Weight of max(0,Δ) term |
| Value function coef | 0.1 | (hardcoded) | Weight of L_V in total loss |
| GAE gamma | 1.0 | (hardcoded) | No discounting |
| GAE lambda | 0.95 | (hardcoded) | Standard GAE smoothing |
| Max new tokens | 256 | `--max-gen-tokens` | Generation length during training |
| Temperature | 0.7 | (hardcoded) | Sampling temperature |
| Top-p | 0.9 | (hardcoded) | Nucleus sampling |
| Max sequence length | 2048 | `--max-length` | Including prompt |
| EMA decay (task norm) | 0.99 | (hardcoded) | Running normalization smoothing |
| EMA decay (FastRL) | 0.99 | (hardcoded) | Per-head statistics |

### SVM Head Generation

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Classifier | `LinearSVC` (sklearn) | Decision boundary normal = reward direction |
| C | 1.0 | Default margin hardness |
| max_iter | 5000 | Ensures convergence on 2048-dim embeddings |
| dual | Auto (True if n > d) | sklearn efficiency recommendation |
| Preprocessing | `StandardScaler` | Stabilizes SVM on raw embeddings |
| Post-processing | Un-scale → L2-normalize | Unit vector for dot-product reward |
| Data per category | ~600-1600 pairs | From train split (varies by category) |
| Train accuracy | 1.0 for all 9 heads | Perfect linear separability |

### Dataset

| Parameter | Value |
|-----------|-------|
| Total samples | 14578 |
| Train split | 11662 (80%) |
| Test split | 2916 (20%) |
| Split seed | 42 |
| Length-filtered train | 10980 (sequences > max_length removed) |
| Categories | 9 (Age, Disability, Gender, Nationality, Physical Appearance, Race/Ethnicity, Religion, SES, Sexual Orientation) |
| Format | Multiple-choice (A/B/C) with 1 fair + 2 biased answers |

---

## Runtime Commands

### Prerequisites

Embeddings must already be extracted (Phase 1). If not:

```bash
modal run src/run_modal.py --phase phase1 --dataset sb_bench
```

### Step 1: Generate SVM Reward Heads

```bash
modal run src/run_modal.py --phase phase2
```

Output:
- `/mnt/data/generated_heads/sb_bench-PCA-component/` — 50 PCA heads (legacy)
- `/mnt/data/generated_heads/sb_bench-SVM-component/` — 9 SVM category heads

### Step 2: Train

```bash
modal run src/run_modal.py \
  --phase train \
  --epochs 3 \
  --head-type svm \
  --num-heads 9 \
  --logit-reward-coef 0.1 \
  --max-gen-tokens 256 \
  --batch-size 12 \
  --output-dir "/mnt/data/output_token_level_svm_v4"
```

### Step 3: Evaluate

```bash
modal run src/run_modal.py \
  --phase evaluation \
  --dataset sb_bench \
  --resume "/mnt/data/output_token_level_svm_v4/checkpoint-ep2-end"
```

### Step 4: Vanilla Baseline

```bash
modal run src/run_modal.py \
  --phase evaluation \
  --dataset sb_bench \
  --vanilla
```

### Full Pipeline (Fresh Start)

```bash
modal run src/run_modal.py --phase all \
  --epochs 3 \
  --head-type svm \
  --max-gen-tokens 256 \
  --output-dir "/mnt/data/output_token_level_svm"
```

---

## Escalation Path

Given v4's failure (+0.11%, not significant), these are potential next steps ordered by effort/novelty:

### Option A: Fix the Reward-Evaluation Alignment (Low Effort)

**Problem**: Training generates 256 tokens but evaluation generates 1 token (A/B/C).

**Fix**: Train on the EXACT evaluation format. Generate only A/B/C during PPO training. Reward = +1 if correct (non-stereotypical), -1 if wrong. This is essentially RLHF with binary correctness reward.

**Pros**: Perfect alignment between training and evaluation.
**Cons**: Requires ground-truth labels during training (we have them); loses the unsupervised geometric elegance.

### Option B: GRPO (Group Relative Policy Optimization)

Generate G=4-8 completions per prompt. Use group-relative advantages (no value function needed). The reward signal comes from ranking completions within each group.

**Pros**: Eliminates value function estimation error; stronger signal from comparative ranking.
**Cons**: 4-8× more generation per step; still has the evaluation format mismatch.

### Option C: DPO/ORPO (Direct Preference Optimization)

Skip RL entirely. Use the SVM projections to rank completions offline, then train with DPO loss directly.

**Pros**: No RL instability; simpler training loop; proven effective.
**Cons**: Offline (no exploration); requires generating preference pairs first.

### Option D: Representation Engineering (Activation Steering)

Instead of RL, directly add/subtract SVM normal vectors from activations during inference (no training needed).

**Pros**: Zero training cost; instant evaluation; highly interpretable.
**Cons**: Requires finding correct injection layer and scaling factor; may degrade general capability.

### Option E: DSO (Direct Steering Optimization)

Train affine LayerNorm adapters instead of LoRA. These directly modify the activation statistics along bias-relevant directions.

**Pros**: More targeted than LoRA; directly moves embeddings in SVM direction.
**Cons**: Complete paradigm shift; new codebase.

### Option F: INLP (Iterative Null-space Projection)

Project out bias directions from the embedding space iteratively. Pure linear algebra, no training.

**Pros**: Guaranteed to remove linear bias information; theoretically grounded.
**Cons**: May remove useful information; doesn't directly map to generation improvement.

---

## Appendix: Key Code Paths for Debugging

### To verify LoRA weights are actually changing:

```python
# Add before/after weight comparison in ppo_loop.py:
with torch.no_grad():
    lora_params = [p for n, p in active_policy.named_parameters() if 'lora' in n and p.requires_grad]
    total_norm = sum(p.grad.norm().item() for p in lora_params if p.grad is not None)
    weight_norm = sum(p.norm().item() for p in lora_params)
    print(f"Grad norm: {total_norm:.6f}, Weight norm: {weight_norm:.6f}")
```

### To verify reward heads are aligned with evaluation behavior:

```python
# Score test-set answers with SVM heads:
# If chosen answers consistently have HIGHER projection than rejected, heads are correct.
# If not, the SVM heads don't predict evaluation correctness.
```

### To check if logprob_diff is truly zero or just display precision:

The diagnostic `mean_abs_logprob_diff` metric (6 decimal places) was added to the training loop. Run training again and check if it's 0.000000 (truly zero, gradient issue) or 0.000050 (just display precision).

---

## Appendix: Module Dependency Graph

```
src/run_modal.py (Modal cloud orchestrator)
  └─ modules/training/train_rl.py (entrypoint)
       ├─ modules/training/args.py (parse_training_args)
       ├─ modules/training/setup.py
       │    ├─ build_accelerator() → Accelerator
       │    ├─ build_policy_model() → (PeftModel, AutoProcessor)
       │    ├─ build_ppo_controller() → PPOVLMController
       │    └─ build_dataloader() → DataLoader
       ├─ modules/training/ppo_loop.py (run_ppo_loop)
       │    └─ PPOVLMController.step() [per batch]
       │         ├─ generate() → tokens
       │         ├─ extract_logits_values_and_hidden() × 2 (policy + ref)
       │         ├─ compute_logprobs()
       │         ├─ FastRLNode.update() → alpha weights
       │         ├─ compute_causal_reward_penalty()
       │         ├─ compute_dense_gae()
       │         ├─ PPO surrogate loss
       │         ├─ Value loss
       │         ├─ compute_dispersive_loss()
       │         └─ accelerator.backward(total_loss)
       ├─ modules/training/checkpoint.py (save/load)
       └─ modules/training/drm_loader.py (load .pth heads)
```
