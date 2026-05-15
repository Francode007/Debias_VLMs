import modal
import os
import subprocess

# 1. Configuration
APP_NAME = "debias-vlms-pipeline"
GITHUB_REPO_URL = "https://github.com/Francode007/Debias_VLMs.git"
VOLUME_NAME = "debias-vlm-persistent-storage"

# 2. Define the Image (shared across all functions)
vlm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "git-lfs", "build-essential", "cmake", "clang")
    .pip_install("packaging", "ninja", "wheel")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6%2Bcu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")
    .add_local_dir(".", remote_path="/root/debias-vlms", ignore=[".venv", ".git", "debias_env", "models_cache", "sb_bench_data", "__pycache__"])
)

# 3. Persistent Storage
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(name=APP_NAME)


# ─── Helper ───────────────────────────────────────────────────────────────────
def _setup_env():
    """Common environment setup for all functions."""
    os.environ["HF_HOME"] = "/mnt/data/huggingface"
    os.environ["DATA_PATH"] = "/mnt/data/sb_bench_data"
    os.environ["OUTPUT_PATH"] = "/mnt/data/embeddings_output"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
    os.chdir("/root/debias-vlms")


# ─── Phase 1a: Dataset Preprocessing (CPU only, no GPU) ──────────────────────
@app.function(
    image=vlm_image,
    gpu=None,                # No GPU needed — processor runs on CPU
    cpu=16.0,
    memory=131072,           # 128GB RAM for large pixel_values
    volumes={"/mnt/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_preprocess():
    """Build and cache the preprocessed dataset chunks (CPU only)."""
    _setup_env()
    print("📦 Phase 1a: Dataset Preprocessing (CPU only)...")
    subprocess.run([
        "python", "cal_emb_modular.py",
        "--preprocess_only",
        "--data_path", os.environ["DATA_PATH"],
        "--max_length", "2048",
    ], check=True)
    volume.commit()
    print("✅ Preprocessing complete. Chunks cached on volume.")


# ─── Phase 1b: Embedding Extraction / Inference (A100-80GB) ──────────────────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",         # Full 80GB for batch_size=16 with large images
    cpu=16.0,
    memory=131072,
    volumes={"/mnt/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_inference():
    """Load model + cached dataset, extract embeddings via forward pass."""
    _setup_env()
    print("🚀 Phase 1b: Embedding Extraction (A100-80GB)...")
    subprocess.run([
        "python", "cal_emb_modular.py",
        "--device", "cuda",
        "--data_path", os.environ["DATA_PATH"],
        "--cls_embs_path", os.environ["OUTPUT_PATH"],
        "--batch_size", "16",
        "--max_length", "2048",
        "--dataloader_num_workers", "12"
    ], check=True)
    volume.commit()
    print("✅ Embedding extraction complete.")


# ─── Phase 2: DRM Head Generation (CPU only) ─────────────────────────────────
@app.function(
    image=vlm_image,
    gpu=None,                # sklearn PCA is CPU-only
    cpu=16.0,
    memory=32768,            # 32GB RAM — PCA on ~500MB of embeddings
    volumes={"/mnt/data": volume},
    timeout=3600,            # PCA finishes in minutes
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_drm_generation():
    """Run PCA on embeddings to generate DRM reward heads."""
    _setup_env()
    print("🧬 Phase 2: DRM Head Generation (CPU only)...")
    subprocess.run([
        "python", "generate_drm_heads.py",
        "--input_dir", os.environ["OUTPUT_PATH"],
        "--output_dir", "/mnt/data/generated_heads",
        "--n_components", "50"
    ], check=True)
    volume.commit()
    print("✅ DRM heads generated.")


# ─── Phase 4: PPO Training (A100-80GB) ───────────────────────────────────────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",
    cpu=16.0,
    memory=131072,
    volumes={"/mnt/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_training():
    """RL fine-tuning with PPO using DRM reward heads."""
    _setup_env()
    print("🤖 Phase 4: PPO Training (A100-80GB)...")
    subprocess.run([
        "python", "train_rl.py",
        "--policy_model_name", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--extractor_model_name", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--reward_heads_dir", "/mnt/data/generated_heads/sb_bench-PCA-component",
        "--num_heads", "100",
        "--per_device_train_batch_size", "12",
        "--max_length", "2048",
        "--data_path", os.environ["DATA_PATH"],
        "--output_dir", "/mnt/data/output_ppo_debiased"
    ], check=True)
    volume.commit()
    print("✅ PPO training complete.")


# ─── Setup (download data/models) ────────────────────────────────────────────
@app.function(
    image=vlm_image,
    gpu=None,
    cpu=4.0,
    memory=16384,
    volumes={"/mnt/data": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_setup():
    """Download SB-Bench data and model weights to volume."""
    _setup_env()
    print("📥 Setup: Downloading data and models...")
    if not os.path.exists(os.environ["DATA_PATH"]):
        subprocess.run(["python", "load_sb_bench.py"], check=True)
    else:
        print("✅ Data already exists.")
    print("📥 Downloading Model: Qwen/Qwen2.5-VL-3B-Instruct")
    subprocess.run(["hf", "download", "Qwen/Qwen2.5-VL-3B-Instruct"], check=True)
    volume.commit()
    print("✅ Setup complete.")


# ─── Local Entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(phase: str = "all"):
    """
    Run pipeline phases with optimized GPU allocation.
    
    Phases:
      setup       - Download data/models (CPU, 16GB RAM)
      preprocess  - Build dataset chunks (CPU, 128GB RAM) 
      inference   - Extract embeddings (A100-80GB)
      phase1      - preprocess + inference combined
      phase2      - Generate DRM heads (CPU, 32GB RAM)
      train       - PPO training (A100-80GB)
      all         - Full pipeline
    """
    if phase in ["all", "setup"]:
        run_setup.remote()
    
    if phase in ["all", "phase1", "preprocess"]:
        run_preprocess.remote()
    
    if phase in ["all", "phase1", "inference"]:
        run_inference.remote()
    
    if phase in ["all", "phase2"]:
        run_drm_generation.remote()
    
    if phase in ["all", "train"]:
        run_training.remote()
