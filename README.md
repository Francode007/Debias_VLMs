# Debias_VLMs: C-DeFR-L Pipeline

This repository implements the **C-DeFR-L (Causal Decomposed and Fair Reward Learning)** framework for debiasing Vision-Language Models:

- **Phase 1**: Extract orthogonal reward heads from preference data using PCA-based DRM decomposition.
- **Phase 2 (Fast-RL)**: Dynamically balance multi-dimensional rewards using Mirror Descent.
- **Phase 3 (CAA)**: Causality-Aware Alignment via custom PPO with per-sample causal feedback.

Targets **SB-Bench** with **Qwen2-VL / Qwen2.5-VL** architectures.

## Project Structure

```
Debias_VLMs/
├── src/
│   ├── run_modal.py                    # Main orchestrator (Modal cloud pipeline)
│   └── modules/
│       ├── utils/                      # Shared utilities
│       │   ├── config.py              # ScriptArguments dataclass
│       │   ├── device_manager.py      # Device detection & optimization
│       │   ├── model_loader.py        # Model loading with fallbacks
│       │   ├── model_architecture.py  # Custom forward for reward extraction
│       │   ├── dataset_builder.py     # Dataset processing & formatting
│       │   ├── data_collator.py       # Batch collation for reward training
│       │   └── reward_trainer.py      # Embedding extraction visualizer
│       ├── rl_components/              # Core RL components
│       │   ├── score_head.py          # MultipleHead reward scoring module
│       │   ├── fast_rl.py             # FastRLNode (mirror descent)
│       │   ├── caa_feedback.py        # Causality-Aware Alignment weights
│       │   ├── custom_vlm_ppo_trainer.py  # PPOVLMController (decoupled PPO)
│       │   ├── rl_data_collator.py    # RL-specific data collation
│       │   └── rl_dataset_builder.py  # RL dataset construction
│       ├── data/                       # Data downloading & adapters
│       │   ├── load_pope.py           # POPE dataset downloader
│       │   ├── load_sb_bench.py       # SB-Bench dataset downloader
│       │   ├── registry.py            # Dataset adapter registry
│       │   └── model_registry.py      # Model-family adapter registry
│       ├── embeddings/                 # Phase 1: embedding extraction & DRM
│       │   ├── extract.py             # VLM embedding extraction pipeline
│       │   └── generate_drm_heads.py  # PCA → orthogonal reward heads
│       ├── inference/                  # Answer generation
│       │   ├── generate_answers.py    # POPE answer generation
│       │   └── generate_sb_bench_answers.py
│       ├── evaluation/                 # Evaluation & metrics
│       │   ├── eval_pope.py           # POPE benchmark evaluation
│       │   ├── eval_sb_bench.py       # SB-Bench accuracy evaluation
│       │   ├── evaluate_drm_heads.py  # DRM head hypothesis evaluation
│       │   ├── pope_evaluator.py      # POPE evaluator class
│       │   ├── sb_bench_evaluator.py  # SB-Bench evaluator class
│       │   └── registry.py            # Evaluator registry
│       └── training/                   # Phase 2+3: RL training
│           ├── train_rl.py            # PPO training entrypoint
│           ├── args.py                # Training argument parsing
│           ├── setup.py               # Accelerator & model wiring
│           ├── ppo_loop.py            # Per-epoch PPO loop
│           ├── checkpoint.py          # Checkpoint save/load
│           └── drm_loader.py          # PCA component loading
├── scratch/                            # Development & debugging scripts
├── requirements.txt
├── .gitignore
└── README.md
```

## Testing

Run the architecture sanity check to verify all imports and entrypoints resolve:

```bash
source debias_env/bin/activate
python src/test_pipeline_sanity.py
```

This validates all 66 checks: module imports, class accessibility, data/model registries,
script entrypoints (`--help`), and the Modal app definition.

## Quick Start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running on Modal (Recommended)

```bash
# Full pipeline
modal run src/run_modal.py --phase all

# Individual phases
modal run src/run_modal.py --phase setup        # Download data/models
modal run src/run_modal.py --phase phase1       # Extract embeddings
modal run src/run_modal.py --phase phase2       # Generate DRM heads
modal run src/run_modal.py --phase train        # PPO training
modal run src/run_modal.py --phase evaluation --dataset pope --gen-file <path>
```

### 3. Running Locally (step-by-step)

```bash
cd src

# Download data
python -m modules.data.load_sb_bench

# Extract embeddings
python -m modules.embeddings.extract \
  --device cuda --data_path ./sb_bench_data/data \
  --cls_embs_path ./embeddings_output --batch_size 16

# Generate DRM heads
python -m modules.embeddings.generate_drm_heads \
  --input_dir ./embeddings_output --output_dir ./generated_heads \
  --n_components 50 --case_name sb_bench

# Evaluate DRM heads
python -m modules.evaluation.evaluate_drm_heads \
  --emb_dir ./embeddings_output \
  --score_head_weight ./generated_heads/sb_bench-PCA-component \
  --data_path ./sb_bench_data/data

# Train (PPO + Fast-RL + CAA)
python -m modules.training.train_rl \
  --policy_model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
  --reward_heads_dir "./generated_heads/sb_bench-PCA-component" \
  --num_heads 100 --fast_rl_strategy exponentiated
```

## GPU Selection

**Recommended: H100** (`gpu="H100"` in Modal)

| GPU | Rate | Est. time/epoch | Cost/epoch (incl. CPU+RAM) |
|-----|------|-----------------|---------------------------|
| A100-80GB | $2.50/hr | ~3.23 hrs | ~$12.60 |
| H100 | $3.95/hr | ~1.90 hrs | ~$10.17 |

H100 is cheaper overall due to ~1.7× throughput improvement offsetting the higher hourly rate.
Modal may auto-upgrade H100 → H200 (same price, 141 GB HBM3e), giving further headroom.

**Current training state**: Checkpoint at epoch 1. Target: 5 epochs (4 remaining).

## In-Domain Train/Test Split

When training and evaluating on the **same dataset** (in-domain), Phase 1 must use
an 80/20 split to avoid data leakage:

| Split | Proportion | Used In |
|-------|-----------|---------|
| Train | 80% | Phase 1 embedding extraction + DRM head generation + PPO training |
| Test | 20% | Evaluation only (held-out for measuring debiasing effectiveness) |

**Implementation notes:**
- The split is applied *before* embedding extraction (Phase 1a/1b) so that test
  samples never influence the PCA reward heads.
- Use a fixed random seed for reproducibility (`seed=42`).
- When evaluating on a *different* dataset (e.g., train on SB-Bench, eval on POPE),
  the full dataset can be used for training — the split only applies to in-domain evaluation.
- The split indices should be persisted to disk so that evaluation can load the
  exact same 20% test partition.

**TODO**: Implement the split in `modules/embeddings/extract.py` and wire the test
indices through to the evaluation scripts. This will be done when updating the
debiasing methodology or introducing a new dataset.

## Pipeline Overview

```
SB-Bench → modules/data/load_sb_bench.py → parquet
         → modules/embeddings/extract.py → emb_*.npy
         → modules/embeddings/generate_drm_heads.py → PCA .pth heads
         → modules/evaluation/evaluate_drm_heads.py → accuracy metrics
         → modules/training/train_rl.py (PPO + FastRL + CAA) → Debiased VLM
         → modules/inference/generate_*_answers.py → model outputs
         → modules/evaluation/eval_*.py → final benchmark scores
```
