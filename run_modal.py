import modal
import os
import subprocess

# 1. Configuration: Update these with your specific details
APP_NAME = "debias-vlms-pipeline"
GITHUB_REPO_URL = "https://github.com/Francode007/Debias_VLMs.git" # Your GitHub Repository
VOLUME_NAME = "debias-vlm-persistent-storage"

# 2. Define the Image (Environment & Code Persistence)
vlm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "git-lfs", "build-essential", "cmake", "clang")
    .pip_install("packaging", "ninja", "wheel") # Pre-install build dependencies for flash-attn
    .pip_install_from_requirements("requirements.txt") # Assumes requirements.txt is local
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6%2Bcu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl") # Install pre-compiled flash-attn wheel to bypass source compilation
    .run_commands(
        f"git clone {GITHUB_REPO_URL} /root/debias-vlms"
    )
)

# 3. Define Persistent Storage (for Models, Data, and Embeddings)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(name=APP_NAME)

# 4. The Execution Function
@app.function(
    image=vlm_image,
    gpu="A100",              # Optimized for your pipeline
    cpu=8.0,                 # High CPU count for data loading
    memory=65536,            # 64GB RAM
    volumes={"/mnt/data": volume}, # Mount point for persistent storage
    timeout=86400,           # 24-hour timeout for long RL training
    secrets=[modal.Secret.from_name("huggingface-secret")] # Required for SB-Bench/Gated models
)
def run_pipeline(phase: str = "all"):
    # Set environment variables for persistent caching on the Volume
    os.environ["HF_HOME"] = "/mnt/data/huggingface"
    os.environ["DATA_PATH"] = "/mnt/data/sb_bench_data"
    os.environ["OUTPUT_PATH"] = "/mnt/data/embeddings_output"
    
    # Change directory to the cloned repository
    os.chdir("/root/debias-vlms")

    # Step 0: Setup Phase (Data & Model Download)
    if phase in ["setup"]:
        print("📥 Starting Setup Phase...")
        
        # Download Data
        if not os.path.exists(os.environ["DATA_PATH"]):
            print("📥 Loading SB-Bench data...")
            subprocess.run(["python", "load_sb_bench.py"], check=True)
            # Copy or move the generated data if it wasn't saved to the right path
            # load_sb_bench.py saves to ./sb_bench_data/data by default
            volume.commit() # Save changes to storage
        else:
            print("✅ Data already exists.")
            
        # Download Models (Base & Extractors)
        print("📥 Downloading Model: Qwen/Qwen2.5-VL-3B-Instruct")
        subprocess.run([
            "hf", "download", "Qwen/Qwen2.5-VL-3B-Instruct"
        ], check=True)
        volume.commit()
        print("✅ Setup complete.")
        return

    # Phase: Profiling
    if phase in ["profiling"]:
        print("⏱️ Starting Profiling Phase...")
        
        print("-> Profiling Pipeline (Phase 1)...")
        subprocess.run(["python", "profile_pipeline.py", "--batch_size", "16"], check=True)
        
        print("-> Profiling RL Pipeline (Phase 2 & 3)...")
        subprocess.run(["python", "profile_rl_pipeline.py", "--batch_size", "4", "--gradient_accumulation", "4"], check=True)
        
        volume.commit()
        print("✅ Profiling complete.")
        return

    # Step 1: Extract Embeddings
    if phase in ["all", "phase1"]:
        print("🚀 Starting Phase 1: Embedding Extraction...")
        subprocess.run([
            "python", "cal_emb_modular.py",
            "--device", "cuda",
            "--data_path", "./sb_bench_data/data", # Or use os.environ["DATA_PATH"] depending on how you moved it
            "--cls_embs_path", os.environ["OUTPUT_PATH"],
            "--batch_size", "32"
        ], check=True)
        volume.commit()

    # Step 2: Generate DRM Heads
    if phase in ["all", "phase2"]:
        print("🧬 Starting Phase 2: Generating DRM Heads...")
        subprocess.run([
            "python", "generate_drm_heads.py",
            "--input_dir", os.environ["OUTPUT_PATH"],
            "--output_dir", "/mnt/data/generated_heads",
            "--n_components", "50"
        ], check=True)
        volume.commit()

    # Step 4: Run RL Training
    if phase in ["all", "train"]:
        print("🤖 Starting Phase 4: PPO Training...")
        subprocess.run([
            "python", "train_rl.py",
            "--policy_model_name", "Qwen/Qwen2.5-VL-3B-Instruct",
            "--extractor_model_name", "Qwen/Qwen2.5-VL-3B-Instruct", # From README
            "--reward_heads_dir", "/mnt/data/generated_heads/sb_bench-PCA-component",
            "--num_heads", "100"
        ], check=True)
        volume.commit()

@app.local_entrypoint()
def main(phase: str = "all"):
    # valid phases: 'setup', 'profiling', 'phase1', 'phase2', 'train', 'all'
    run_pipeline.remote(phase=phase)
