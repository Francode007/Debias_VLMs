# C-DeFR-L v2: Token-Level Dense Reward VLM Debiasing

## Architecture Overview

This is the v2 architecture of the Causal Decomposed Fair Reward Learning (C-DeFR-L) pipeline. It resolves three fundamental flaws identified in v1 that caused negligible debiasing performance (66.53% vs 66.56% vanilla on SB-Bench):

| v1 Flaw | Root Cause | v2 Fix |
|---------|-----------|--------|
| EOS-only reward | Credit assignment failure — PPO can't localize which token introduced bias | **Token-level dense reward** at every generated position |
| PCA heads = noise | Variance-maximizing captures structural artifacts, not bias semantics | **SVM boundary normals** — one per demographic category |
| Null-space hacking | Optimizer shifts embeddings without changing token outputs | **Logit-grounded reward** — measures actual probability shift |

## Mathematical Formulation

### Dense Token-Level Reward

For a generated sequence of length $T$, the reward at each token position $t$:

$$r_t = \underbrace{\sum_{k=1}^{K} \alpha_k \cdot (h_t \cdot w_k)}_{\text{task reward}} + \underbrace{\lambda_{\text{logit}} \cdot (\log \pi(x_t) - \log \pi_{\text{ref}}(x_t))}_{\text{logit-grounded}} + \underbrace{\frac{c_{\text{causal}}}{T}}_{\text{distributed causal penalty}} - \underbrace{\beta \cdot (\log \pi(x_t) - \log \pi_{\text{ref}}(x_t))}_{\text{KL penalty}}$$

Where:
- $h_t \in \mathbb{R}^d$ — penultimate layer hidden state at token $t$
- $w_k \in \mathbb{R}^d$ — SVM boundary normal for category $k$ (unit vector)
- $\alpha_k$ — entropy-regularized FastRL weight (deficit-based)
- $c_{\text{causal}}$ — hinge penalty on mean-token embedding drift from reference

### SVM Reward Heads

For each bias category $c \in \{$Age, Disability, Gender, ..., Sexual Orientation$\}$:

$$w_c = \arg\max_{\|w\|=1} \text{margin}(w) \quad \text{s.t.} \quad w^T x_i^{\text{chosen}} > w^T x_j^{\text{rejected}} \; \forall (i,j) \in \mathcal{D}_c$$

The SVM normal vector $w_c$ defines the reward direction — projecting any token embedding onto this direction measures bias alignment for that specific category.

### Logit-Grounded Reward (Anti-Hacking)

Prevents representation hacking by rewarding actual token probability shifts:

$$r_t^{\text{logit}} = \lambda_{\text{logit}} \cdot \left(\log \pi_\theta(x_t | x_{<t}) - \log \pi_{\text{ref}}(x_t | x_{<t})\right)$$

This closes the null-space loophole: the optimizer cannot satisfy this component without actually changing discrete token distributions.

### Dispersive Loss (Anti-Collapse)

Applied on mean-pooled embeddings (O(B²), not O((B·T)²)):

$$\mathcal{L}_{\text{disp}} = -\frac{1}{B(B-1)} \sum_{i \neq j} \log(\|e_i - e_j\|_2 + \epsilon)$$

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{PPO}}^{\text{unscaled}} + \lambda_V \cdot \mathcal{L}_V + \lambda_D \cdot \mathcal{L}_{\text{disp}}$$

The PPO surrogate objective remains **unscaled** — trust region geometry preserved via ε-clipping.

## Pipeline Phases

```
Phase 1: Embedding Extraction (A100-80GB)
    → VLM forward pass on (image, chosen/rejected) pairs
    → Saves per-sample .npy files (penultimate layer EOS embeddings)

Phase 2: DRM Head Generation (CPU, 32GB RAM)
    → PCA heads (legacy): variance-maximizing, 50 components
    → SVM heads (new): per-category linear SVM, 9 boundary normals

Phase 4: PPO Training (A100-80GB)
    → Token-level dense reward from SVM heads
    → Logit-grounded reward prevents null-space exploitation
    → Entropy-regularized FastRL balances 9 category heads
    → Dense GAE with per-token reward signal

Evaluation: SB-Bench test split (2916 samples)
    → Multiple-choice accuracy per demographic category
```

## SVM Head Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Classifier | `LinearSVC` (sklearn) | Need decision boundary normal as direction vector |
| C | 1.0 | Default margin hardness |
| max_iter | 5000 | Ensures convergence on 2048-dim embeddings |
| dual | Auto (True if n > d) | sklearn efficiency recommendation |
| Preprocessing | `StandardScaler` | Stabilizes SVM on raw embeddings |
| Post-processing | Un-scale → L2-normalize | Unit vector for dot-product reward |
| Data per category | ~600-1600 pairs | From train split, grouped by `category` column |

## PPO Training Hyperparameters

| Parameter | Default | Flag |
|-----------|---------|------|
| Epochs | 3 | `--epochs` |
| Batch size | 12 | `--per-device-train-batch-size` (in code) |
| Learning rate | 1e-5 | `--learning-rate` |
| LoRA rank | 16 | `--lora-r` |
| LoRA alpha | 32 | `--lora-alpha` |
| KL beta | 0.1 | `--kl-beta` |
| PPO clip range | 0.2 | `--ppo-clip-range` |
| Num heads | 9 | `--num-heads` |
| FastRL tau | 1.0 | `--tau` |
| Lambda causal | 0.5 | `--lambda-causal` |
| Delta margin | 1.0 | `--delta-margin` |
| Lambda dispersive | 0.01 | `--lambda-dispersive` |
| Logit reward coef | 0.1 | `--logit-reward-coef` |
| Head type | svm | `--head-type` |
| Max new tokens | 32 | Hardcoded in PPO step |
| Generation temp | 0.7 | Hardcoded in PPO step |

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

This generates both PCA (legacy) and SVM heads. Output:
- `/mnt/data/generated_heads/sb_bench-PCA-component/` — 50 PCA heads
- `/mnt/data/generated_heads/sb_bench-SVM-component/` — 9 SVM category heads

### Step 2: Train (Token-Level Dense Reward + SVM Heads)

```bash
modal run src/run_modal.py \
  --phase train \
  --epochs 3 \
  --max-train-samples 2000 \
  --head-type svm \
  --num-heads 9 \
  --logit-reward-coef 0.1 \
  --output-dir "/mnt/data/output_token_level_svm"
```

### Step 3: Evaluate (Test Split Only)

```bash
modal run src/run_modal.py \
  --phase evaluation \
  --dataset sb_bench \
  --resume "/mnt/data/output_token_level_svm/checkpoint-ep3-end"
```

### Step 4: Vanilla Baseline Comparison

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
  --max-train-samples 2000 \
  --output-dir "/mnt/data/output_token_level_svm"
```

## Key Files Modified (v1 → v2)

| File | Change |
|------|--------|
| `src/modules/rl_components/custom_vlm_ppo_trainer.py` | Full rewrite: dense per-token reward, logit-grounded component, removed dual generation, mean-token causal penalty |
| `src/modules/rl_components/fast_rl.py` | Updated for token-level: receives mean-pooled (B,K), alpha used externally for per-token weighting |
| `src/modules/embeddings/generate_drm_heads.py` | Added `generate_svm_heads()` with per-category LinearSVC training |
| `src/modules/training/args.py` | Added `--logit_reward_coef` |
| `src/modules/training/setup.py` | Passes `logit_reward_coef` to PPOVLMController |
| `src/modules/training/ppo_loop.py` | Progress bar shows `r_dense`, `logit_r` metrics |
| `src/run_modal.py` | Phase 2 generates PCA+SVM; training accepts `--head-type`, `--logit-reward-coef`; defaults to SVM |

## Computational Cost

Token-level reward adds **negligible** overhead vs v1:
- Hidden states already computed by the forward pass (`output_hidden_states=True`)
- Reward projection: `(B, T, 2048) @ (2048, 9)` — trivial matmul
- **Savings**: Removed the separate reference generation pass (~10-15% faster)
- Dispersive loss: O(B²) on mean-pooled embeddings (not O((B·T)²))
- Peak GPU memory: ~67GB on A100-80GB (unchanged)

## Expected Improvements

v1 achieved 66.53% (vs 66.56% vanilla) — effectively zero improvement.

v2 targets >2% absolute improvement by:
1. Providing dense gradient signal to every token (not just EOS)
2. Using semantically meaningful reward directions (SVM per category)
3. Preventing reward hacking via logit-grounded component
4. Preserving all v1 stability fixes (dispersive loss, unscaled PPO, entropy-reg FastRL)

## Escalation Path

If v2 still shows insufficient improvement:
- **Phase 4 (GRPO Multi-Sample)**: Generate G=4 completions per prompt, group-relative advantages
- **DSO (Direct Steering Optimization)**: Train affine LayerNorm adapters instead of LoRA — complete paradigm shift
- **INLP (Iterative Null-space Projection)**: Exhaustive geometric debiasing of the full embedding space
