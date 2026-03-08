# Debias_VLMs: C-DeFR-L Phase 1 — Orthogonal Reward Head Extraction

This repository implements **Phase 1 of the C-DeFR-L (Causal Decomposed and Fair Reward Learning)** framework: extracting orthogonal reward heads from preference data using the [DRMs (Decomposed Reward Models)](https://arxiv.org/abs/2502.13131) approach. It targets **SB-Bench** ([ucf-crcv/SB-Bench](https://huggingface.co/datasets/ucf-crcv/SB-Bench)) with **Qwen2-VL / Qwen2.5-VL** as the embedding model. The resulting reward heads are intended for later RL fine-tuning to debias open-source VLMs.

## Overview

- **Preference data:** SB-Bench (stereotype bias benchmark). Preferred = non-stereotypical answer (`label`); rejected = stereotypical options. Each example yields **2 preference pairs** (chosen vs each wrong answer).
- **Embedding model:** The base VLM (e.g. Qwen2.5-VL) is used as the embedding model; no separate reward model is trained. Last-layer hidden states at the last token position are extracted for chosen and rejected responses.
- **DRM heads:** PCA is run on difference vectors (chosen − rejected). Each component is saved as a reward head (PyTorch `.pth`), including both positive and negated directions.
- **Evaluation:** `evaluate_drm_heads.py` scores embeddings with the DRM heads and reports overall and per-category (9 bias types) accuracy.

## Pipeline

```
SB-Bench (HuggingFace) → load_sb_bench.py → parquet
       → cal_emb_modular.py → embeddings (emb_*.npy)
       → generate_drm_heads.py → PCA components (.pth)
       → score_head.py / evaluate_drm_heads.py → metrics (JSON)
```

## Quick Start (GPU)

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data

Download SB-Bench and save as parquet:

```bash
python load_sb_bench.py
```

Data is written to `./sb_bench_data/data/` (e.g. `sb_bench_data.parquet`). Accept the dataset terms on the [SB-Bench Hugging Face page](https://huggingface.co/datasets/ucf-crcv/SB-Bench) if required.

### 3. Models

The pipeline uses **Qwen2.5-VL** or **Qwen2-VL**. Paths can be set in `local_model_config.py`; otherwise models are loaded from HuggingFace.

- Default: `Qwen/Qwen2.5-VL-7B-Instruct`
- Fallback: `Qwen/Qwen2-VL-7B-Instruct`
- Lighter option: `Qwen/Qwen2-VL-2B-Instruct`

### 4. Step 1: Extract Embeddings

Runs the VLM in inference-only mode and saves (chosen, rejected, prompt) hidden-state embeddings per preference pair. No training or score head is used.

```bash
python cal_emb_modular.py \
  --device cuda \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data_path ./sb_bench_data/data \
  --cls_embs_path ./embeddings_output \
  --batch_size 1
```

Use `--use_smallset` for a small subset when debugging. Output: `embeddings_output/emb_0.npy`, `emb_1.npy`, … (each shape `(1, 3, hidden_size)` with chosen, rejected, prompt).

### 5. Step 2: Generate DRM Heads (PCA)

Builds difference vectors (chosen − rejected), runs PCA, and saves each component as a `.pth` state dict (plus negated direction).

```bash
python generate_drm_heads.py \
  --input_dir ./embeddings_output \
  --output_dir ./generated_heads \
  --n_components 50 \
  --case_name sb_bench
```

Output under `generated_heads/sb_bench-PCA-component/`:

- `sb_bench-PCA-component0.pth` … `sb_bench-PCA-component(k-1).pth` (positive)
- `sb_bench-PCA-componentk.pth` … `sb_bench-PCA-component(2k-1).pth` (negated)
- `explained_variance_ratio.npy`, `explained_variance.npy`, `orthogonal_heads.npy`

### 6. Step 3: Evaluate DRM Heads (optional)

Scores saved embeddings with the DRM heads and computes accuracy (overall and per SB-Bench category).

```bash
python evaluate_drm_heads.py \
  --emb_dir ./embeddings_output \
  --score_head_weight ./generated_heads/sb_bench-PCA-component \
  --data_path ./sb_bench_data/data \
  --output_json ./drm_head_results.json
```

Use `--num_heads N` to evaluate only the first N heads.

## Key Arguments

| Script | Argument | Description |
|--------|----------|-------------|
| `cal_emb_modular.py` | `--device` | `cuda`, `mps`, or `cpu` |
| | `--data_path` | Directory with SB-Bench parquet files |
| | `--cls_embs_path` | Where to save `emb_*.npy` |
| | `--batch_size` | Dataloader batch size (default 1) |
| | `--use_smallset` | Use small subset for debugging |
| `generate_drm_heads.py` | `--input_dir` | Directory of `emb_*.npy` |
| | `--output_dir` | Directory for PCA outputs and `case_name-PCA-component/` |
| | `--n_components` | Number of PCA components (default 50) |
| | `--case_name` | Prefix for `.pth` filenames (default `sb_bench`) |
| | `--full_composed` | Use k = hidden_dim components |
| `evaluate_drm_heads.py` | `--emb_dir` | Directory of `emb_*.npy` |
| | `--score_head_weight` | Directory of DRM `.pth` files |
| | `--data_path` | SB-Bench parquet dir (for category labels) |
| | `--output_json` | Output path for metrics JSON |

## Module Layout

- **`modules/`** — Config, device manager, model loader, dataset builder (SB-Bench, 2 pairs per example, user/assistant format), data collator, custom forward (hidden-state extraction only), reward trainer (embedding visualization).
- **`load_sb_bench.py`** — Download SB-Bench and save as parquet.
- **`cal_emb_modular.py`** — Entry point for embedding extraction.
- **`generate_drm_heads.py`** — PCA on (chosen − rejected), save `.pth` heads.
- **`score_head.py`** — `MultipleHead`: load `.pth` components and score embeddings `(N, 2, hidden_dim)` → (rewards_chosen, rewards_rejected).
- **`evaluate_drm_heads.py`** — Load embeddings and DRM heads, compute overall and per-category accuracy, write JSON.

## References

- **DRMs:** [Rethinking Diverse Human Preference Learning through Principal Component Analysis](https://arxiv.org/abs/2502.13131) (arXiv:2502.13131). Code: [amandaluof/DRMs](https://github.com/amandaluof/DRMs).
- **SB-Bench:** [Stereotype Bias Benchmark for Large Multimodal Models](https://huggingface.co/datasets/ucf-crcv/SB-Bench) (ucf-crcv/SB-Bench; arXiv:2502.08779).

## Troubleshooting

- **OOM:** Reduce `--batch_size` to 1, or use `--model Qwen/Qwen2-VL-2B-Instruct`.
- **NaNs (e.g. on MPS):** Try `--device cpu` or `--force_fp32`.
- **Missing SB-Bench:** Log in with `huggingface-cli login` and accept the dataset terms on the SB-Bench dataset page.
- **CUDA/MPS:** Device is auto-selected; override with `--device cuda` or `--device mps`.
