# Debias_VLMs: Reward Model Training & DRM Head Generation

This repository provides a pipeline for training reward models and generating Orthogonal DRM (Direction of Reward Model) heads for Vision-Language Models (VLMs), specifically focusing on the Qwen2-VL family.

## 🚀 Quick Start (GPU)

This guide assumes you have a CUDA-capable GPU and appropriate drivers installed.

### 1. Environment Setup

Ensure you have Python 3.10+ and install the dependencies:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Data Preparation

Download and prepare the **SB-Bench** dataset. This script downloads the dataset from HuggingFace and converts it to the parquet format required by the training pipeline.

```bash
python load_sb_bench.py
```

*This will save the data to `./sb_bench_data/data/sb_bench_data.parquet`.*

### 3. Model Setup

The pipeline is configured to use **Qwen2.5-VL** or **Qwen2-VL** models. By default, it attempts to load models from a local path configured in `local_model_config.py`. If not found, it falls back to downloading from HuggingFace.

**Supported Models:**
- `Qwen/Qwen2.5-VL-7B-Instruct` (Default)
- `Qwen/Qwen2-VL-7B-Instruct` (Fallback)
- `Qwen/Qwen2-VL-2B-Instruct` (For faster testing/lower memory)

### 4. Step 1: Extract Embeddings (`cal_emb_modular.py`)

This step runs the model on the input data, extracts hidden states, and computes reward scores.

**Run on GPU:**

```bash
python cal_emb_modular.py \
    --device cuda \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir run_logs \
    --cls_embs_path embeddings_output
```

**Common Arguments:**
- `--device`: `cuda` for GPU, `cpu` for CPU (slow), `mps` for Mac (experimental).
- `--model`: Path or HuggingFace ID of the model.
- `--batch_size`: Batch size for data processing (default: 1).
- `--use_smallset`: Use a tiny subset of data for debugging.
- `--cls_embs_path`: Directory to save extracted embeddings.

**Note on Memory:**
If you encounter OOM (Out of Memory) errors:
1. Reduce `--batch_size` (e.g., to 1).
2. Use a smaller model: `--model Qwen/Qwen2-VL-2B-Instruct`.
3. Enable 8-bit loading (requires `bitsandbytes`): `--load_in_8bit`.

### 5. Step 2: Generate DRM Heads (`generate_drm_heads.py`)

This step uses the extracted embeddings to generate orthogonal direction vectors (DRM heads) using PCA. These heads represent the "direction" of the reward/preference in the embedding space.

**Run Generation:**

```bash
python generate_drm_heads.py \
    --input_dir embeddings_output \
    --output_dir generated_heads \
    --n_components 5
```

**Arguments:**
- `--input_dir`: Directory containing the `.npy` embedding files from Step 1.
- `--output_dir`: Directory to save the generated heads.
- `--n_components`: Number of orthogonal heads to generate (default: 5).

### 6. Output

The final output in `generated_heads/` will contain:
- `orthogonal_heads.npy`: The generated DRM heads (shape: `[n_components, hidden_dim]`).
- `explained_variance.npy`: Variance explained by each head.

These heads can now be used as reward signals for subsequent Reinforcement Learning (RL) stages.

---

## 🛠️ Troubleshooting

- **NaN Values:** If you see NaN values in embeddings (especially on Mac/MPS), ensure you use `--device cpu`. On CUDA/GPU, this should not be an issue.
- **Missing Models:** If the script fails to find a local model, it will try HuggingFace. Ensure you have internet access and a valid `HF_TOKEN` environment variable if assessing gated models.