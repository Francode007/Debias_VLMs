# Debias_VLMs: C-DeFR-L Pipeline (Phases 1, 2 & 3)

This repository implements the **C-DeFR-L (Causal Decomposed and Fair Reward Learning)** framework:
- **Phase 1**: Extracting orthogonal reward heads from preference data using the [DRMs (Decomposed Reward Models)](https://arxiv.org/abs/2502.13131) approach.
- **Phase 2 (Fast-RL)**: Dynamically balancing multi-dimensional rewards using Mirror Descent configurations.
- **Phase 3 (CAA)**: Causality-Aware Alignment applying causal interventional feedback via a custom PPO loop.

It targets **SB-Bench** ([ucf-crcv/SB-Bench](https://huggingface.co/datasets/ucf-crcv/SB-Bench)) with **Qwen2-VL / Qwen2.5-VL** architectures. The complete pipeline handles embedding extraction, DRM generation, and the final sample-weighted PPO decoupled reinforcement learning logic to debias open-source VLMs.

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
       → train_rl.py (custom_vlm_ppo_trainer.py + fast_rl.py + caa_feedback.py) → Debiased VLM
```

## Quick Start (GPU)

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.5. HuggingFace Remote Server Setup

When working on a remote GPU server, it's highly recommended to configure your Hugging Face cache directory and authenticate to access gated models/datasets (like SB-Bench).

```bash
# Optional: Set a custom cache directory if your home/user partition has limited space
export HF_HOME="/path/to/large/storage/huggingface"

# Login to Hugging Face (requires a token from your HF account)
huggingface-cli login
```

### 2. Data

**Option A: Automated via script (Recommended)**
Download SB-Bench and save as parquet:

```bash
python load_sb_bench.py
```
Data is written to `./sb_bench_data/data/` (e.g. `sb_bench_data.parquet`). Make sure you have accepted the dataset terms on the [SB-Bench Hugging Face page](https://huggingface.co/datasets/ucf-crcv/SB-Bench) prior to running.

**Option B: Manual Download via HF CLI**
If you prefer to pre-download the dataset explicitly using the Hugging Face CLI:
```bash
huggingface-cli download ucf-crcv/SB-Bench --repo-type dataset --local-dir ./sb_bench_data/raw
```

### 3. Models

The pipeline uses **Qwen2.5-VL** (default), **Qwen2-VL**, or **Qwen3.5-VL**.

There is no separate script solely for downloading models. Instead, the models are handled in two ways:

1. **Dynamic Download:** By default, the `transformers` library automatically downloads and caches models at runtime to your HuggingFace cache directory (e.g., the `HF_HOME` path set above).
2. **Manual Pre-download (Recommended for Remote Servers):** To avoid connection interruptions or downloading large weights dynamically, you can explicitly pre-download the models to a specific local path using the Hugging Face CLI. You will then need to reference this custom path in `local_model_config.py`.

```bash
# Download Qwen2.5-VL 3B (Default) to a specific local folder
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/Qwen2.5-VL-3B-Instruct

# Download Qwen2-VL 2B (Fallback) to a specific local folder
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./models/Qwen2-VL-2B-Instruct
```

- **Default:** `Qwen/Qwen2.5-VL-3B-Instruct` (Qwen2.5 series 3B; use 1.5B when available)
- **Fallback:** `Qwen/Qwen2-VL-2B-Instruct`
- **Supported model series and sizes:**
  - **Qwen2.5-VL (2024–2025):** 3B, 7B (0.5B/1.5B are text-only on HF; do not use 7B for testing unless explicitly mentioned)
  - **Qwen2-VL:** 2B, 7B
  - **Qwen3.5 series (2026):** 800M, 2B, 4B

### 4. Step 1: Extract Embeddings

Runs the VLM in inference-only mode and saves (chosen, rejected, prompt) hidden-state embeddings per preference pair. No training or score head is used.

```bash
python cal_emb_modular.py \
  --device cuda \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --data_path ./sb_bench_data/data \
  --cls_embs_path ./embeddings_output \
  --batch_size 32 \
  --dataloader_num_workers 8
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
  --batch_size 1024 \
  --output_json ./drm_head_results.json
```

Use `--num_heads N` to evaluate only the first N heads.

### 7. Step 4: Run Fast-RL + CAA PPO Training

After generating the `.pth` PCA heads, run the end-to-end memory-decoupled PPO execution script avoiding heavy abstract constraints. The script utilizes a specified Base model (policy) and a frozen Extractor model (reward tracking), employing LoRA adapter toggling to handle dynamic generation isolating VRAM loads efficiently on 80GB hardware logic instances.

```bash
python train_rl.py \
  --policy_model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
  --extractor_model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
  --reward_heads_dir "./generated_heads/sb_bench-PCA-component" \
  --fast_rl_strategy "exponentiated" \
  --eta 0.01 \
  --kl_beta 0.1 \
  --num_heads 100
```

**Fast-RL Strategy Types Supported**:
- `exponentiated`: Multiplicative weights / Exponentiated gradient steps keeping arrays inside Simplex space cleanly safely.
- `projected`: Projected gradient ascent bounding math parameters robustly.
- `adam`: Adam-style parameters scaling robust logs mapping cleanly back via softmax equations tracking batches.

---

## Testing the full flow

**Option A: One script (recommended)**

```bash
chmod +x run_phase1_full.sh
./run_phase1_full.sh
```

Overrides (env vars):

```bash
DEVICE=cuda MODEL=Qwen/Qwen2.5-VL-7B-Instruct ./run_phase1_full.sh   # larger model
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

## Optimized GPU Configuration (A100 80GB)

For a high-end GPU like the **80GB A100**, use these settings to saturate the tensor cores and avoid data starvation:

1. **Embeddings:** `python cal_emb_modular.py --device cuda --model Qwen/Qwen2-VL-2B-Instruct --batch_size 32 --dataloader_num_workers 8` (High batch size + parallel CPU loading).
2. **PCA:** Ensure `cuml` is installed to offload orthogonal head generation to the GPU.
3. **Evaluation:** Use a large evaluation batch size: `--batch_size 1024`.

These optimizations reduce the total pipeline estimate from ~171 hours to **under 1 hour**.

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

## Profiling the Pipeline

To estimate the full GPU runtime for the complete dataset (both Phase 1 and the RL loop), use the included profiling scripts. They will run a small subset and extrapolate the total time based on the dataset size.

**1. Profile Phase 1 (Embeddings + PCA + Evaluate):**
```bash
python profile_pipeline.py --batch_size 16
```
This generates `profiling_report.json` with the estimated time for creating the DRM heads.

**2. Profile Phase 2 & 3 (PPO True Dual Generation):**
```bash
python profile_rl_pipeline.py --batch_size 4 --gradient_accumulation 4
```
This generates `profiling_report_rl.json` with the estimated time for 1 full Epoch of the reinforcement learning loop. Sum the total extrapolated hours from both JSON reports to plan your A100 compute budget!

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

- **`modules/`** — Backend components including config, device manager, model loader, dataset builder, data collator, and architecture abstractions.
  - **`fast_rl.py`** — Evaluates and balances multi-dimensional orthogonal rewards dynamically.
  - **`caa_feedback.py`** — Causal interventional weighting functions (L2 norms, min-max batched tracking).
  - **`custom_vlm_ppo_trainer.py`** — A decoupled Accelerate PPO loop, applying FastRL, CAA feedback, KL penalties, and exact inference over visual elements via precise LoRA toggling logic.
- **`load_sb_bench.py`** — Download SB-Bench and save as parquet.
- **`cal_emb_modular.py`** — Entry point for embedding extraction.
- **`generate_drm_heads.py`** — PCA on (chosen − rejected), save `.pth` heads.
- **`score_head.py`** — `MultipleHead`: load `.pth` components and score embeddings.
- **`evaluate_drm_heads.py`** — Score embeddings and DRM heads statically.
- **`train_rl.py`** — The Master Reinforcement Learning orchestration script to execute the PPO VLM training loops.

## References

- **DRMs:** [Rethinking Diverse Human Preference Learning through Principal Component Analysis](https://arxiv.org/abs/2502.13131) (arXiv:2502.13131). Code: [amandaluof/DRMs](https://github.com/amandaluof/DRMs).
- **SB-Bench:** [Stereotype Bias Benchmark for Large Multimodal Models](https://huggingface.co/datasets/ucf-crcv/SB-Bench) (ucf-crcv/SB-Bench; arXiv:2502.08779).

## Troubleshooting

- **OOM:** Reduce `--batch_size` to 1, or use `--model Qwen/Qwen2-VL-2B-Instruct`.
- **NaNs in embeddings / PCA "Input X contains NaN":** On **MPS** and **CPU**, the pipeline automatically uses **float32** (`--force_fp32` is set by `run_phase1_full.sh` when `DEVICE=mps` or `DEVICE=cpu`). On CUDA you can pass `--force_fp32` manually if you see NaNs. See below for how fp32 affects runs.
- **Missing SB-Bench:** Log in with `huggingface-cli login` and accept the dataset terms on the SB-Bench dataset page.
- **CUDA/MPS:** Device is auto-selected; override with `--device cuda` or `--device mps`.
- **Invalid buffer size / size mismatch:** On Mac or limited GPU memory, use `MODEL=Qwen/Qwen2-VL-2B-Instruct` and `DEVICE=mps` or `DEVICE=cpu`. For Qwen2.5-VL-7B you need a recent `transformers` with `Qwen2_5VLForConditionalGeneration`; otherwise the loader skips to Qwen2-VL-7B.

---

## Common warnings (are they safe?)

| Warning | Safe? | What to do |
|--------|--------|------------|
| **urllib3 NotOpenSSLWarning** (LibreSSL vs OpenSSL) | Yes | Ignore. Your Python’s SSL is LibreSSL; HTTPS still works. |
| **TRL FutureWarning** (Python 3.9 dropped later) | Yes | Ignore for now; plan to use Python 3.10+ when convenient. |
| **Flash attention not available, falling back to eager** | Yes | Normal on Mac/CPU. Eager is correct, just slower. |
| **Image processor loaded as fast processor** | Yes | Informational; no change needed. |
| **`torch_dtype` is deprecated, use `dtype`** | Yes | Upstream deprecation; safe to ignore. |
| **qwen2_5_vl instantiated as qwen2_vl** | No | Old transformers loaded Qwen2.5 with wrong class. Upgrade `transformers` or use `Qwen2-VL-7B` / `Qwen2-VL-2B`. |
| **Invalid buffer size: 14.41 GiB** | No | 7B model too large for device. Use `DEVICE=mps` or `cpu` on Mac and/or `MODEL=Qwen/Qwen2-VL-2B-Instruct`. |
| **size mismatch for bias (3584 vs 1280)** | No | Architecture mismatch (Qwen2.5 loaded as Qwen2). Fixed by not mixing classes; use correct model or smaller model on Mac. |

### How does `--force_fp32` (or fp32 on MPS/CPU) affect runs?

When the device is **MPS** or **CPU**, the pipeline forces **float32** for model and embedding calculations. When the device is **CUDA**, you can pass `--force_fp32` yourself.

| Effect | What it means |
|--------|----------------|
| **Stability** | Reduces or avoids NaN/inf in embeddings. bfloat16/fp16 on MPS or CPU can be incomplete or produce NaNs; fp32 is more reliable. |
| **Memory** | Uses roughly **2×** the activation memory of fp16 and **2×** the model weight memory of bfloat16. On memory‑limited machines, use a smaller model (e.g. 2B/3B) or `--batch_size 1`. |
| **Speed** | Slower than bfloat16/fp16 on GPUs that support them. On MPS/CPU, fp32 is usually the stable option anyway, so the main trade-off is memory. |
| **Quality** | Embeddings and DRM heads are numerically more stable; downstream PCA and evaluation are unaffected aside from avoiding NaNs. |
