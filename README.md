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

**Output** (under `output_dir/`):

| File | Shape / count | Description |
|------|----------------|-------------|
| `explained_variance_ratio.npy` | `(k,)` | Fraction of total variance captured by each component (sum ≤ 1). |
| `explained_variance.npy` | `(k,)` | Eigenvalue (variance) of each component. |
| `orthogonal_heads.npy` | `(k, hidden_dim)` | All PCA component vectors (rows). |
| `{case_name}-PCA-component/*.pth` | 2k files | Component `i`: positive direction `w_i` (files `0..k-1`) and negated `-w_i` (files `k..2k-1`), each as PyTorch state dict `{"weight": (1, hidden_dim)}`. |

**What each PCA component is:** The script fits PCA on the matrix of difference vectors `d_n = φ(chosen_n) − φ(rejected_n)`. Each component is an orthogonal direction of maximum variance in that space: component 0 is the main axis of “chosen vs rejected” variation, component 1 is the next orthogonal axis, and so on. So each component is a **reward head**: for a response embedding `φ(y)`, the reward is `r_i(y) = w_i^T φ(y)`. Early components often capture broad stereotype vs non‑stereotypical; later ones can capture finer or category-specific bias. The negated heads (`-w_i`) give the opposite preference and are used for flexible composition in evaluation or RL.

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

---

## Testing the full flow

**Option A: One script (recommended)**

```bash
chmod +x run_phase1_full.sh
./run_phase1_full.sh
```

Overrides (env vars):

```bash
DEVICE=cuda MODEL=Qwen/Qwen2-VL-2B-Instruct ./run_phase1_full.sh   # smaller model
USE_SMALLSET=1 ./run_phase1_full.sh   # tiny data for quick test
N_COMPONENTS=10 ./run_phase1_full.sh   # fewer PCA components
```

**Option B: Step-by-step bash commands**

```bash
# 0. Environment and data
pip install -r requirements.txt
python load_sb_bench.py

# 1. Extract embeddings
python cal_emb_modular.py --device cuda --data_path ./sb_bench_data/data --cls_embs_path ./embeddings_output --batch_size 1

# 2. Generate DRM heads
python generate_drm_heads.py --input_dir ./embeddings_output --output_dir ./generated_heads --n_components 50 --case_name sb_bench

# 3. Evaluate (Phase 1 hypothesis)
python evaluate_drm_heads.py --emb_dir ./embeddings_output --score_head_weight ./generated_heads/sb_bench-PCA-component --data_path ./sb_bench_data/data --output_json ./drm_head_results.json
```

Results: `./drm_head_results.json` and printed overall + per-category accuracy.

---

## Evaluating Phase 1: `evaluate_drm_heads.py` and your hypothesis

**What the script does**

1. **Load embeddings** — Reads all `emb_*.npy` from `--emb_dir`. Each file has shape `(1, 3, hidden_dim)`; the script uses slices 0 and 1 (chosen and rejected response embeddings).
2. **Load DRM heads** — Loads all `.pth` files from `--score_head_weight` into `MultipleHead` (one linear layer per component).
3. **Score each pair** — For every preference pair, computes `reward_chosen = W @ chosen_emb` and `reward_rejected = W @ rejected_emb` (per head).
4. **Correct** — A pair is “correct” for a head when `reward_chosen > reward_rejected` (the head prefers the non-stereotypical answer).
5. **Aggregate** — Overall accuracy = fraction of pairs correct (averaged over heads or samples). Per-category: same metric restricted to samples in that SB-Bench category (Age, Gender, etc.), using `data_index // 2` to map back to the original row and thus to `category`.

**Output JSON (`--output_json`)**

| Field | Meaning |
|--------|--------|
| `overall_per_head` | List of accuracies, one per head: when using only that head, fraction of pairs where chosen beats rejected. |
| `overall_mean` | Mean of those accuracies (or equivalently, fraction correct over all head–sample pairs). |
| `num_samples`, `num_heads` | Counts. |
| `per_category` | For each of the 9 SB-Bench categories: `accuracy_per_head`, `accuracy_mean`, `count`. |

**How to use this to evaluate your Phase 1 hypothesis**

- **Hypothesis:** “PCA on (chosen − rejected) yields directions that separate non-stereotypical from stereotypical responses.”
- **Check 1 — Overall:** If `overall_mean` is clearly **above 0.5**, the DRM heads collectively assign higher reward to chosen (non-stereotypical) than to rejected; the decomposition is useful.
- **Check 2 — Per head:** If **some heads have much higher accuracy** than others, those axes are more predictive; you can prioritize or combine them in Stage 2 (RL).
- **Check 3 — Per category:** Use **per-category accuracy** to see which heads help for which bias type (e.g. Age vs Gender). That supports selecting or weighting heads for debiasing specific dimensions.

So: run the full flow, open `drm_head_results.json`, and interpret `overall_mean`, `overall_per_head`, and `per_category` as above to validate Phase 1 before moving to RL.

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
