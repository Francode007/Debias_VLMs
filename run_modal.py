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
    os.environ["POPE_DATA_PATH"] = "/mnt/data/pope_data"
    os.environ["OUTPUT_PATH"] = "/mnt/data/embeddings_output"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
    os.chdir("/root/debias-vlms")


# ─── Phase 1a: Dataset Preprocessing (CPU only, no GPU) ──────────────────────
@app.function(
    image=vlm_image,
    gpu=None,                # No GPU needed — processor runs on CPU
    cpu=8.0,
    memory=131072,           # 128GB RAM for large pixel_values
    volumes={"/mnt/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_preprocess(dataset: str = "sb_bench", model_family: str = "qwen"):
    """Build and cache the preprocessed dataset chunks (CPU only)."""
    _setup_env()
    print(f"📦 Phase 1a: Dataset Preprocessing (CPU only) for {dataset}...")
    subprocess.run([
        "python", "cal_emb_modular.py",
        "--preprocess_only",
        "--dataset_name", dataset,
        "--model_family", model_family,
        "--data_path", os.environ["DATA_PATH"],
        "--max_length", "2048",
    ], check=True)
    volume.commit()
    print("✅ Preprocessing complete. Chunks cached on volume.")


# ─── Phase 1b: Embedding Extraction / Inference (A100-80GB) ──────────────────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",         # Full 80GB for batch_size=16 with large images
    cpu=8.0,
    memory=131072,
    volumes={"/mnt/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_inference(dataset: str = "sb_bench", model_family: str = "qwen"):
    """Load model + cached dataset, extract embeddings via forward pass."""
    _setup_env()
    print(f"🚀 Phase 1b: Embedding Extraction (A100-80GB) for {dataset}...")
    subprocess.run([
        "python", "cal_emb_modular.py",
        "--dataset_name", dataset,
        "--model_family", model_family,
        "--device", "cuda",
        "--data_path", os.environ["DATA_PATH"],
        "--cls_embs_path", os.environ["OUTPUT_PATH"],
        "--batch_size", "16",
        "--max_length", "2048",
        "--dataloader_num_workers", "6"
    ], check=True)
    volume.commit()
    print("✅ Embedding extraction complete.")


# ─── Phase 2: DRM Head Generation (CPU only) ─────────────────────────────────
@app.function(
    image=vlm_image,
    gpu=None,                # sklearn PCA is CPU-only
    cpu=8.0,
    memory=32768,            # 32GB RAM — PCA on ~500MB of embeddings
    volumes={"/mnt/data": volume},
    timeout=14400,           # 4h — loading 21k files from volume is I/O-bound
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
    cpu=8.0,
    memory=131072,
    volumes={"/mnt/data": volume},
    timeout=43200,            # 12 hours
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_training(epochs: int = 1, output_dir: str = "/mnt/data/output_ppo_debiased", resume_from: str = None, dataset: str = "sb_bench", model_family: str = "qwen"):
    """RL fine-tuning with PPO using DRM reward heads."""
    _setup_env()
    print(f"🤖 Phase 4: PPO Training (A100-80GB) — {epochs} epoch(s) on {dataset}...")
    cmd = [
        "python", "train_rl.py",
        "--dataset_name", dataset,
        "--model_family", model_family,
        "--policy_model_name", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--extractor_model_name", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--reward_heads_dir", "/mnt/data/generated_heads/sb_bench-PCA-component",
        "--num_heads", "100",
        "--per_device_train_batch_size", "12",
        "--gradient_accumulation_steps", "4",
        "--max_length", "2048",
        "--epochs", str(epochs),
        "--data_path", os.environ["DATA_PATH"],
        "--output_dir", output_dir
    ]
    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])
    subprocess.run(cmd, check=True)
    volume.commit()
    print("✅ PPO training complete.")


# ─── Setup (download data/models) ────────────────────────────────────────────
@app.function(
    image=vlm_image,
    gpu=None,
    cpu=2.0,
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
        print("✅ SB-Bench Data already exists.")
        
    pope_jsonl = os.path.join(os.environ["POPE_DATA_PATH"], "pope_data.jsonl")
    if not os.path.exists(os.environ["POPE_DATA_PATH"]) or not os.path.exists(pope_jsonl):
        subprocess.run(["python", "load_pope.py"], check=True)
    else:
        print("✅ POPE Data already exists.")
    print("📥 Downloading Model: Qwen/Qwen2.5-VL-3B-Instruct")
    subprocess.run(["hf", "download", "Qwen/Qwen2.5-VL-3B-Instruct"], check=True)
    volume.commit()
    print("✅ Setup complete.")


# ─── Phase: Generation ────────────────────────────────────────────────────────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    volumes={"/mnt/data": volume},
    timeout=14400,
)
def run_generation(dataset: str, checkpoint_dir: str, data_path: str, output_jsonl: str):
    """Run generation (inference) for a given dataset."""
    _setup_env()
    print(f"🧠 Starting Generation Phase for dataset: {dataset}")
    
    if dataset.lower() == "pope":
        cmd = [
            "python", "generate_answers.py",
            "--data_path", data_path,
            "--output_jsonl", output_jsonl,
            "--batch_size", "16"
        ]
        if checkpoint_dir:
            cmd += ["--checkpoint_dir", checkpoint_dir]
        subprocess.run(cmd, check=True)
        volume.commit()
        print(f"✅ Generation complete. Results saved to {output_jsonl}")

    elif dataset.lower() == "sb_bench":
        cmd = [
            "python", "generate_sb_bench_answers.py",
            "--data_path", data_path,
            "--output_jsonl", output_jsonl,
            "--batch_size", "8"
        ]
        if checkpoint_dir:
            cmd += ["--checkpoint_dir", checkpoint_dir]
        subprocess.run(cmd, check=True)
        volume.commit()
        print(f"✅ SB-Bench generation complete. Results saved to {output_jsonl}")

    else:
        print(f"❌ Error: Generation not implemented for dataset '{dataset}'")


# ─── Phase: Evaluation ────────────────────────────────────────────────────────
@app.function(
    image=vlm_image,
    gpu=None,
    cpu=4.0,
    memory=16384,
    volumes={"/mnt/data": volume},
    timeout=3600,
)
def run_evaluation(dataset: str, gt_file: str, gen_file: str):
    """Run evaluation scripts based on the selected dataset."""
    _setup_env()
    print(f"📊 Starting Evaluation Phase for dataset: {dataset}")
    
    import sys
    sys.path.append("/root/debias-vlms")

    if dataset.lower() == "pope":
        if not gt_file or not gen_file:
            print("❌ Error: Both --gt-file and --gen-file must be provided for POPE evaluation.")
            return
        print(f"Evaluating POPE using GT: {gt_file} and Gens: {gen_file}")
        subprocess.run([
            "python", "eval_pope.py",
            "--gt_files", gt_file,
            "--gen_files", gen_file
        ], check=True)

    elif dataset.lower() == "sb_bench":
        if not gen_file:
            print("❌ Error: --gen-file must be provided for SB-Bench evaluation.")
            return
        print(f"Evaluating SB-Bench generations: {gen_file}")
        subprocess.run([
            "python", "eval_sb_bench.py",
            "--gen_file", gen_file
        ], check=True)

    else:
        print(f"❌ Error: Unknown dataset '{dataset}'. Supported: pope, sb_bench")


# ─── Local Entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(phase: str = "all", epochs: int = 1, resume: str = "", output_dir: str = "", dataset: str = "sb_bench", model_family: str = "qwen", gt_file: str = "", gen_file: str = "", vanilla: bool = False):
    """
    Run pipeline phases with optimized GPU allocation.
    
    Phases:
      setup       - Download data/models (CPU, 16GB RAM)
      preprocess  - Build dataset chunks (CPU, 128GB RAM) 
      inference   - Extract embeddings (A100-80GB)
      phase1      - preprocess + inference combined
      phase2      - Generate DRM heads (CPU, 32GB RAM)
      train       - PPO training (A100-80GB). Use --epochs N for multi-epoch.
      train5      - PPO training for 5 epochs (separate output dir)
      train10     - PPO training for 10 epochs (separate output dir)
      evaluation  - Run evaluation metrics on generated outputs (requires --dataset)
      all         - Full pipeline
    
    Flags:
      --vanilla   Skip LoRA; run raw Qwen2.5-VL-3B-Instruct as baseline (evaluation phase only)
    
    Resume: pass --resume <checkpoint_path> e.g. /mnt/data/output_ppo_debiased/checkpoint-ep1-50pct
    """
    if phase in ["all", "setup"]:
        run_setup.remote()
    
    if phase in ["all", "phase1", "preprocess"]:
        run_preprocess.remote(dataset=dataset, model_family=model_family)
    
    if phase in ["all", "phase1", "inference"]:
        run_inference.remote(dataset=dataset, model_family=model_family)
    
    if phase in ["all", "phase2"]:
        run_drm_generation.remote()
    
    resume_ckpt = resume if resume else None
    
    if phase in ["all", "train"]:
        train_out = output_dir if output_dir else "/mnt/data/output_ppo_debiased"
        run_training.remote(epochs=epochs, output_dir=train_out, resume_from=resume_ckpt, dataset=dataset, model_family=model_family)
    
    if phase == "train5":
        run_training.remote(epochs=5, output_dir="/mnt/data/output_ppo_debiased_5ep", resume_from=resume_ckpt, dataset=dataset, model_family=model_family)
    
    if phase == "train10":
        run_training.remote(epochs=10, output_dir="/mnt/data/output_ppo_debiased_10ep", resume_from=resume_ckpt, dataset=dataset, model_family=model_family)

    if phase == "evaluation":
        if dataset.lower() == "pope":
            gen_data_path = "/mnt/data/pope_data/pope_data.parquet"
            if vanilla:
                checkpoint = None
                default_gen_file = "/mnt/data/vanilla_baseline/pope_generations.jsonl"
                print("ℹ️ Vanilla mode: generating with raw Qwen2.5-VL-3B-Instruct (no debiasing adapter).")
            else:
                checkpoint = resume_ckpt if resume_ckpt else "/mnt/data/output_ppo_debiased/checkpoint-ep1-end"
                default_gen_file = f"{checkpoint}/generations.jsonl"
            if not gen_file:
                gen_file = default_gen_file
                print(f"ℹ️ No --gen-file provided. Running generation first → {gen_file}")
                run_generation.remote(
                    dataset="pope",
                    checkpoint_dir=checkpoint,
                    data_path=gen_data_path,
                    output_jsonl=gen_file
                )
            if not gt_file:
                gt_file = "/mnt/data/pope_data/pope_data.jsonl"
            run_evaluation.remote(dataset=dataset, gt_file=gt_file, gen_file=gen_file)

        elif dataset.lower() == "sb_bench":
            gen_data_path = "/mnt/data/sb_bench_data/sb_bench_data.parquet"
            if vanilla:
                checkpoint = None
                default_gen_file = "/mnt/data/sb_bench_vanilla_baseline/sb_bench_generations.jsonl"
                print("ℹ️ Vanilla mode: generating SB-Bench answers with raw Qwen2.5-VL-3B-Instruct.")
            else:
                checkpoint = resume_ckpt if resume_ckpt else "/mnt/data/output_ppo_debiased/checkpoint-ep1-end"
                default_gen_file = f"{checkpoint}/sb_bench_generations.jsonl"
            if not gen_file:
                gen_file = default_gen_file
                print(f"ℹ️ No --gen-file provided. Running SB-Bench generation first → {gen_file}")
                run_generation.remote(
                    dataset="sb_bench",
                    checkpoint_dir=checkpoint,
                    data_path=gen_data_path,
                    output_jsonl=gen_file
                )
            run_evaluation.remote(dataset=dataset, gt_file="", gen_file=gen_file)

        else:
            print(f"❌ Unsupported dataset for evaluation: '{dataset}'. Choose: pope, sb_bench")
