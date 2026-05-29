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
    .add_local_dir(".", remote_path="/root/debias-vlms", ignore=[".venv", ".git", "debias_env", "models_cache", "sb_bench_data", "scratch", "__pycache__"])
)

# 3. Persistent Storage
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(name=APP_NAME)


# ─── Helper ───────────────────────────────────────────────────────────────────
SPLIT_INDICES_PATH = "/mnt/data/split_indices.json"

def _setup_env():
    """Common environment setup for all functions."""
    os.environ["HF_HOME"] = "/mnt/data/huggingface"
    os.environ["DATA_PATH"] = "/mnt/data/sb_bench_data"
    os.environ["POPE_DATA_PATH"] = "/mnt/data/pope_data"
    os.environ["VLBIAS_DATA_PATH"] = "/mnt/data/vlbiasbench_data"
    os.environ["OUTPUT_PATH"] = "/mnt/data/embeddings_output"
    os.environ["SPLIT_INDICES_PATH"] = SPLIT_INDICES_PATH
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
    os.chdir("/root/debias-vlms/src")


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
def run_preprocess(dataset: str = "sb_bench", model_family: str = "qwen",
                   completion_format: str = "free_text", split: str = "train"):
    """Build and cache the preprocessed dataset chunks (CPU only).

    Phase 0.6 D3: pass --completion-format letter to cache a separate
    chunked dataset that conditions on a single-letter assistant turn instead
    of the free-text answer. The dataset cache key includes completion_format
    and split_mode, so train/test and free_text/letter caches coexist.
    """
    _setup_env()
    print(f"📦 Phase 1a: Dataset Preprocessing (CPU only) for {dataset} "
          f"[completion_format={completion_format}, split={split}]...")
    subprocess.run([
        "python", "-m", "modules.embeddings.extract",
        "--preprocess_only",
        "--dataset_name", dataset,
        "--model_family", model_family,
        "--data_path", os.environ["DATA_PATH"],
        "--max_length", "2048",
        "--split", split,
        "--split_indices_path", SPLIT_INDICES_PATH,
        "--completion_format", completion_format,
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
def run_inference(dataset: str = "sb_bench", model_family: str = "qwen",
                  completion_format: str = "free_text", token_position: str = "eos",
                  split: str = "train"):
    """Load model + cached dataset, extract embeddings via forward pass.

    Phase 0.6 D3: pass --completion-format letter --token-position post_letter to
    extract embeddings aligned with what PPO scores at training time. The
    extract script auto-suffixes the output path to avoid clobbering legacy
    embeddings_output/. Pass --split test to produce held-out emb files for
    run_drm_eval (train and test files coexist in the same emb dir because
    their data_indices are disjoint).
    """
    _setup_env()
    print(f"🚀 Phase 1b: Embedding Extraction (A100-80GB) for {dataset} "
          f"[completion_format={completion_format}, token_position={token_position}, split={split}]...")
    subprocess.run([
        "python", "-m", "modules.embeddings.extract",
        "--dataset_name", dataset,
        "--model_family", model_family,
        "--device", "cuda",
        "--data_path", os.environ["DATA_PATH"],
        "--cls_embs_path", os.environ["OUTPUT_PATH"],
        "--batch_size", "16",
        "--max_length", "2048",
        "--dataloader_num_workers", "6",
        "--split", split,
        "--split_indices_path", SPLIT_INDICES_PATH,
        "--completion_format", completion_format,
        "--token_position", token_position,
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
def run_drm_generation(completion_format: str = "free_text", token_position: str = "eos",
                       n_components: int = 50):
    """Run PCA and SVM on embeddings to generate DRM reward heads.

    Reads embeddings from the same auto-suffixed path produced by run_inference
    for the given (completion_format, token_position) combo, and writes heads
    into a parallel suffixed dir under /mnt/data/generated_heads/.
    """
    _setup_env()
    print(f"🧬 Phase 2: DRM Head Generation (CPU only) "
          f"[completion_format={completion_format}, token_position={token_position}]...")

    # Mirror extract.py's auto-suffixing rule so head-build reads the right
    # embeddings dir and writes into a parallel heads dir.
    input_dir = os.environ["OUTPUT_PATH"]
    heads_root = "/mnt/data/generated_heads"
    if (completion_format != "free_text") or (token_position != "eos"):
        suffix = f"_{completion_format}_{token_position}"
        input_dir = input_dir.rstrip("/") + suffix
        heads_root = heads_root.rstrip("/") + suffix

    # Generate PCA heads (legacy, kept for comparison)
    print(f"  → PCA heads (input_dir={input_dir}, output_dir={heads_root})...")
    subprocess.run([
        "python", "-m", "modules.embeddings.generate_drm_heads",
        "--input_dir", input_dir,
        "--output_dir", heads_root,
        "--n_components", str(n_components),
        "--head_type", "pca",
        "--split", "train",
        "--split_indices_path", SPLIT_INDICES_PATH,
    ], check=True)

    # Generate SVM heads (category-specific boundary normals)
    print(f"  → SVM heads (input_dir={input_dir}, output_dir={heads_root})...")
    subprocess.run([
        "python", "-m", "modules.embeddings.generate_drm_heads",
        "--input_dir", input_dir,
        "--output_dir", heads_root,
        "--head_type", "svm",
        "--data_path", os.environ["DATA_PATH"],
        "--split", "train",
        "--split_indices_path", SPLIT_INDICES_PATH,
    ], check=True)

    volume.commit()
    print("✅ DRM heads generated (PCA + SVM).")


# ─── Phase 3: DRM Head Evaluation (held-out test split) ──────────────────────
@app.function(
    image=vlm_image,
    gpu="A100-40GB",         # GPU just for fast matmul; CPU works too
    cpu=8.0,
    memory=32768,
    volumes={"/mnt/data": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_drm_eval(completion_format: str = "free_text",
                 token_position: str = "eos",
                 head_type: str = "svm",
                 split: str = "test"):
    """Evaluate DRM heads on held-out split.

    For each head, reports fraction of (chosen, rejected) pairs where
    w·φ_chosen > w·φ_rejected. Acceptance gate #1 requires
    overall_mean ≥ 0.65 AND per-category min ≥ 0.55 on split=test.
    """
    _setup_env()
    print(f"📊 Phase 3: DRM Head Evaluation "
          f"[completion_format={completion_format}, token_position={token_position}, "
          f"head_type={head_type}, split={split}]...")

    emb_dir = os.environ["OUTPUT_PATH"]
    heads_root = "/mnt/data/generated_heads"
    if (completion_format != "free_text") or (token_position != "eos"):
        suffix = f"_{completion_format}_{token_position}"
        emb_dir = emb_dir.rstrip("/") + suffix
        heads_root = heads_root.rstrip("/") + suffix

    head_subdir = "sb_bench-SVM-component" if head_type.lower() == "svm" else "sb_bench-PCA-component"
    score_head_weight = os.path.join(heads_root, head_subdir)
    output_json = os.path.join(heads_root, f"drm_head_eval_{head_type}_{split}.json")

    print(f"  → emb_dir={emb_dir}")
    print(f"  → score_head_weight={score_head_weight}")
    print(f"  → output_json={output_json}")

    subprocess.run([
        "python", "-m", "modules.evaluation.evaluate_drm_heads",
        "--emb_dir", emb_dir,
        "--score_head_weight", score_head_weight,
        "--data_path", os.environ["DATA_PATH"],
        "--output_json", output_json,
        "--split", split,
        "--split_indices_path", SPLIT_INDICES_PATH,
        "--head_type", head_type,
    ], check=True)

    volume.commit()
    print(f"✅ DRM head eval written to {output_json}")


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
def run_training(epochs: int = 1, output_dir: str = "/mnt/data/output_ppo_debiased", resume_from: str = None, dataset: str = "sb_bench", model_family: str = "qwen",
                 kl_beta: float = 0.1, ppo_clip_range: float = 0.2, learning_rate: float = 1e-5,
                 lora_r: int = 16, lora_alpha: int = 32, max_train_samples: int = None, eta: float = 0.01,
                 num_heads: int = 0, tau: float = 1.0, lambda_causal: float = 0.5,
                 delta_margin: float = 1.0, lambda_dispersive: float = 0.01,
                 logit_reward_coef: float = 0.1, head_type: str = "svm",
                 max_gen_tokens: int = 256, batch_size: int = 12,
                 reward_mode: str = "svm",
                 # Phase 0 collapse-mitigation knobs.
                 target_kl: float = 0.0,
                 kl_adapt_rate: float = 0.1,
                 kl_beta_min: float = 0.05,
                 kl_beta_max: float = 5.0,
                 value_clip_range: float = 0.0,
                 lr_schedule: str = "linear_warmup",
                 min_lr_ratio: float = 0.2,
                 warmup_ratio: float = 0.05,
                 ckpt_every_steps: int = 0,
                 value_learning_rate: float = 0.0,
                 early_stop_patience: int = 0,
                 early_stop_threshold: float = 0.05,
                 early_stop_window: int = 10,
                 gradient_accumulation_steps: int = 4,
                 max_grad_norm: float = 1.0,
                 midtrain_eval_every_steps: int = 0,
                 midtrain_eval_samples: int = 64,
                 use_eval_for_early_stop: bool = False,
                 # Phase 0.6 D-blocker flags.
                 use_frozen_phi: bool = False,
                 kept_heads_filter: str = None,
                 heads_suffix: str = ""):
    """RL fine-tuning with PPO using DRM reward heads."""
    _setup_env()
    print(f"🤖 Phase 4: PPO Training (A100-80GB) — {epochs} epoch(s) on {dataset}...")
    print(f"   Hyperparams: kl_beta={kl_beta}, clip={ppo_clip_range}, lr={learning_rate}, "
          f"lora_r={lora_r}, lora_alpha={lora_alpha}, max_samples={max_train_samples}, "
          f"num_heads={num_heads}, tau={tau}, lambda_causal={lambda_causal}, "
          f"delta_margin={delta_margin}, lambda_dispersive={lambda_dispersive}, "
          f"logit_reward_coef={logit_reward_coef}, head_type={head_type}, "
          f"max_gen_tokens={max_gen_tokens}, batch_size={batch_size}, "
          f"reward_mode={reward_mode}")
    print(f"   Phase0: target_kl={target_kl}, kl_adapt_rate={kl_adapt_rate}, "
          f"kl_beta_min={kl_beta_min}, kl_beta_max={kl_beta_max}, "
          f"value_clip_range={value_clip_range}, lr_schedule={lr_schedule}, "
          f"min_lr_ratio={min_lr_ratio}, warmup_ratio={warmup_ratio}, "
          f"ckpt_every_steps={ckpt_every_steps}")

    # Select reward heads directory based on head type
    heads_root = "/mnt/data/generated_heads" + (heads_suffix or "")
    if head_type == "svm":
        reward_heads_dir = os.path.join(heads_root, "sb_bench-SVM-component")
    else:
        reward_heads_dir = os.path.join(heads_root, "sb_bench-PCA-component")

    cmd = [
        "python", "-m", "modules.training.train_rl",
        "--dataset_name", dataset,
        "--model_family", model_family,
        "--policy_model_name", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--extractor_model_name", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--reward_heads_dir", reward_heads_dir,
        "--num_heads", str(num_heads),
        "--per_device_train_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--max_grad_norm", str(max_grad_norm),
        "--max_length", "2048",
        "--epochs", str(epochs),
        "--data_path", os.environ["DATA_PATH"],
        "--output_dir", output_dir,
        "--split", "train",
        "--split_indices_path", SPLIT_INDICES_PATH,
        "--kl_beta", str(kl_beta),
        "--ppo_clip_range", str(ppo_clip_range),
        "--learning_rate", str(learning_rate),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--eta", str(eta),
        "--tau", str(tau),
        "--lambda_causal", str(lambda_causal),
        "--delta_margin", str(delta_margin),
        "--lambda_dispersive", str(lambda_dispersive),
        "--logit_reward_coef", str(logit_reward_coef),
        "--max_gen_tokens", str(max_gen_tokens),
        "--reward_mode", reward_mode,
        # Phase 0 collapse-mitigation flags.
        "--target_kl", str(target_kl),
        "--kl_adapt_rate", str(kl_adapt_rate),
        "--kl_beta_min", str(kl_beta_min),
        "--kl_beta_max", str(kl_beta_max),
        "--value_clip_range", str(value_clip_range),
        "--lr_schedule", lr_schedule,
        "--min_lr_ratio", str(min_lr_ratio),
        "--warmup_ratio", str(warmup_ratio),
        "--ckpt_every_steps", str(ckpt_every_steps),
    ]
    if early_stop_patience > 0:
        cmd.extend([
            "--early_stop_patience", str(early_stop_patience),
            "--early_stop_threshold", str(early_stop_threshold),
            "--early_stop_window", str(early_stop_window),
        ])
    if midtrain_eval_every_steps > 0:
        cmd.extend([
            "--midtrain_eval_every_steps", str(midtrain_eval_every_steps),
            "--midtrain_eval_samples", str(midtrain_eval_samples),
        ])
        if use_eval_for_early_stop:
            cmd.append("--use_eval_for_early_stop")
    if value_learning_rate and value_learning_rate > 0.0:
        cmd.extend(["--value_learning_rate", str(value_learning_rate)])
    if max_train_samples:
        cmd.extend(["--max_train_samples", str(max_train_samples)])
    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])
    # Phase 0.6 D1 / D2 flags.
    if use_frozen_phi:
        cmd.append("--use_frozen_phi")
    if kept_heads_filter:
        cmd.extend(["--kept_heads_filter", kept_heads_filter])
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
        subprocess.run(["python", "-m", "modules.data.load_sb_bench"], check=True)
    else:
        print("✅ SB-Bench Data already exists.")
        
    pope_jsonl = os.path.join(os.environ["POPE_DATA_PATH"], "pope_data.jsonl")
    if not os.path.exists(os.environ["POPE_DATA_PATH"]) or not os.path.exists(pope_jsonl):
        subprocess.run(["python", "-m", "modules.data.load_pope"], check=True)
    else:
        print("✅ POPE Data already exists.")

    vlbias_parquet = os.path.join(os.environ["VLBIAS_DATA_PATH"], "vlbiasbench_close_ended.parquet")
    # Rebuild if missing OR if the old embedded-images parquet exists (>1 GB).
    needs_rebuild = not os.path.exists(vlbias_parquet)
    if not needs_rebuild and os.path.getsize(vlbias_parquet) > 1_000_000_000:
        print("⚠️  Old embedded-images parquet detected (>1 GB). Removing and rebuilding...")
        os.remove(vlbias_parquet)
        needs_rebuild = True
    if needs_rebuild:
        subprocess.run(["python", "-u", "-m", "modules.data.load_vlbiasbench"], check=True)
    else:
        print("✅ VLBiasBench data already exists.")
    
    # Generate train/test split (80/20) if not already present
    if not os.path.exists(SPLIT_INDICES_PATH):
        print("📊 Generating 80/20 train/test split...")
        subprocess.run([
            "python", "-c",
            f"from modules.utils.split import generate_split; generate_split('{os.environ['DATA_PATH']}', '{SPLIT_INDICES_PATH}')"
        ], check=True)
    else:
        print("✅ Split indices already exist.")
    
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
            "python", "-m", "modules.inference.generate_answers",
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
            "python", "-m", "modules.inference.generate_sb_bench_answers",
            "--data_path", data_path,
            "--output_jsonl", output_jsonl,
            "--batch_size", "8",
            "--split", "test",
            "--split_indices_path", SPLIT_INDICES_PATH,
        ]
        if checkpoint_dir:
            cmd += ["--checkpoint_dir", checkpoint_dir]
        subprocess.run(cmd, check=True)
        volume.commit()
        print(f"✅ SB-Bench generation complete. Results saved to {output_jsonl}")

    else:
        print(f"❌ Error: Generation not implemented for dataset '{dataset}'")


# ─── Phase 0a: SVM Head Activation-Steering Sanity Gate (A100-80GB) ─────────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    volumes={"/mnt/data": volume},
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_phase0_steering(
    num_samples: int = 256,
    batch_size: int = 4,
    layers: str = "12,18,24,30,34",
    lambdas: str = "-3.0,-1.5,0.0,1.5,3.0",
    heads: str = "",
    output_json: str = "/mnt/data/phase0/steering.json",
    heads_dir: str = "/mnt/data/generated_heads/sb_bench-SVM-component",
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
):
    """Phase 0a: inject λ·w_k at chosen layers; measure SB-Bench Δ accuracy."""
    _setup_env()
    os.chdir("/root/debias-vlms")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    print(f"🔬 Phase 0a: Activation-steering sweep "
          f"(layers={layers}, lambdas={lambdas}, n={num_samples})")
    cmd = [
        "python", "-m", "scripts.phase0_activation_steering",
        "--base_model", base_model,
        "--data_path", "/mnt/data/sb_bench_data/sb_bench_data.parquet",
        "--heads_dir", heads_dir,
        "--split_indices_path", SPLIT_INDICES_PATH,
        "--split", "test",
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--layers", *layers.split(","),
        "--lambdas", *lambdas.split(","),
        "--output_json", output_json,
    ]
    if heads:
        cmd += ["--heads", *heads.split(",")]
    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"✅ Phase 0a complete → {output_json}")


# ─── Phase 0b: Reward / Metric Correlation (A100-80GB) ──────────────────────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    volumes={"/mnt/data": volume},
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_phase0_correlation(
    num_samples: int = 512,
    output_json: str = "/mnt/data/phase0/correlation.json",
    heads_dir: str = "/mnt/data/generated_heads/sb_bench-SVM-component",
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
):
    """Phase 0b: does the SVM head rank correct letter > wrong letters?"""
    _setup_env()
    os.chdir("/root/debias-vlms")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    print(f"🔬 Phase 0b: Reward/metric correlation (n={num_samples})")
    subprocess.run([
        "python", "-m", "scripts.phase0_reward_correlation",
        "--base_model", base_model,
        "--data_path", "/mnt/data/sb_bench_data/sb_bench_data.parquet",
        "--heads_dir", heads_dir,
        "--split_indices_path", SPLIT_INDICES_PATH,
        "--split", "test",
        "--num_samples", str(num_samples),
        "--output_json", output_json,
    ], check=True)
    volume.commit()
    print(f"✅ Phase 0b complete → {output_json}")


# ─── Phase 0c: McNemar Paired Test (CPU only) ───────────────────────────────
@app.function(
    image=vlm_image,
    gpu=None,
    cpu=2.0,
    memory=4096,
    volumes={"/mnt/data": volume},
    timeout=600,
)
def run_phase0_mcnemar(vanilla_jsonl: str, v4_jsonl: str):
    """Phase 0c: paired exact-binomial McNemar test on two answer JSONLs."""
    _setup_env()
    os.chdir("/root/debias-vlms")
    print(f"🔬 Phase 0c: McNemar test\n  vanilla = {vanilla_jsonl}\n  v4      = {v4_jsonl}")
    subprocess.run([
        "python", "-m", "scripts.phase0_mcnemar",
        "--vanilla_jsonl", vanilla_jsonl,
        "--v4_jsonl",      v4_jsonl,
    ], check=True)
    print("✅ Phase 0c complete.")


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
            "python", "-m", "modules.evaluation.eval_pope",
            "--gt_files", gt_file,
            "--gen_files", gen_file
        ], check=True)

    elif dataset.lower() == "sb_bench":
        if not gen_file:
            print("❌ Error: --gen-file must be provided for SB-Bench evaluation.")
            return
        print(f"Evaluating SB-Bench generations: {gen_file}")
        subprocess.run([
            "python", "-m", "modules.evaluation.eval_sb_bench",
            "--gen_file", gen_file
        ], check=True)

    else:
        print(f"❌ Error: Unknown dataset '{dataset}'. Supported: pope, sb_bench")


# ─── Phase 0 Eval Sweep: generate + eval every checkpoint, single combined JSON ──
@app.function(
    image=vlm_image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    volumes={"/mnt/data": volume},
    timeout=43200,  # 12h — plenty for sweeping ~10-30 ckpts
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_phase0_eval_sweep(
    train_output_dir: str = "/mnt/data/output_ppo_phase0",
    combined_json: str = "/mnt/data/output_ppo_phase0/phase0_eval_sweep.json",
    data_path: str = "/mnt/data/sb_bench_data/sb_bench_data.parquet",
    dataset: str = "sb_bench",
    gen_subdir: str = "generations",
    only_tags: str = "",   # comma-separated list of tags to evaluate; empty = all
    skip_existing: bool = True,
    force: bool = False,
    shuffle_answers: str = "none",  # Phase 0.7 P2: {none, cyclic_1, cyclic_2}
):
    """
    Sweep every `checkpoint-*` under `train_output_dir`, run generation + eval
    on each one, and incrementally update a single combined JSON file at
    `combined_json`. Safe to restart: each completed checkpoint is written to
    disk and the volume is committed before moving on, so the next run picks
    up where this one left off (unless --force).

    Output schema (combined_json):
    {
      "train_output_dir": "...",
      "dataset": "sb_bench",
      "data_path": "...",
      "results": {
        "ep1-step10":   { "metrics": {...}, "per_category": {...},
                          "checkpoint_dir": "...", "gen_file": "...",
                          "eval_json": "...", "completed_at": "..." },
        "ep1-step20":   { ... },
        ...
      },
      "ordered_tags": ["ep1-step10", "ep1-step20", ...],
      "summary": {
        "best": {"tag": "...", "accuracy": ...},
        "last_updated": "..."
      }
    }
    """
    _setup_env()
    import glob
    import json
    import re
    import time

    os.makedirs(os.path.dirname(combined_json), exist_ok=True)
    gen_root = os.path.join(train_output_dir, gen_subdir)
    os.makedirs(gen_root, exist_ok=True)

    # ── Discover checkpoints ────────────────────────────────────────────
    ckpt_paths = sorted(glob.glob(os.path.join(train_output_dir, "checkpoint-*")))
    if not ckpt_paths:
        print(f"❌ No checkpoint-* directories under {train_output_dir}")
        return

    def _tag_of(p):
        # /mnt/data/output_ppo_phase0/checkpoint-ep1-step10 → ep1-step10
        return os.path.basename(p).replace("checkpoint-", "", 1)

    # Sort: epoch first, then numeric step / pct
    def _sort_key(p):
        tag = _tag_of(p)
        m_ep = re.search(r"ep(\d+)", tag)
        ep = int(m_ep.group(1)) if m_ep else 0
        m_step = re.search(r"step(\d+)", tag)
        m_pct = re.search(r"(\d+)pct", tag)
        if m_step:
            return (ep, 0, int(m_step.group(1)))
        if m_pct:
            return (ep, 1, int(m_pct.group(1)))
        return (ep, 2, 0)

    ckpt_paths.sort(key=_sort_key)
    allow = set(t.strip() for t in only_tags.split(",") if t.strip()) if only_tags else None

    print(f"🔍 Found {len(ckpt_paths)} checkpoints under {train_output_dir}")
    for p in ckpt_paths:
        print(f"   - {_tag_of(p)}")

    # ── Load / init combined JSON ───────────────────────────────────────
    if os.path.exists(combined_json) and not force:
        with open(combined_json) as f:
            combined = json.load(f)
        print(f"📂 Resuming from existing {combined_json} "
              f"({len(combined.get('results', {}))} already done)")
    else:
        combined = {
            "train_output_dir": train_output_dir,
            "dataset": dataset,
            "data_path": data_path,
            "results": {},
            "ordered_tags": [],
            "summary": {"best": None, "last_updated": None},
        }

    def _flush():
        # Recompute summary
        best_tag, best_acc = None, -1.0
        for tag, r in combined["results"].items():
            acc = r.get("metrics", {}).get("accuracy")
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_tag = tag
        combined["summary"] = {
            "best": {"tag": best_tag, "accuracy": best_acc} if best_tag else None,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = combined_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(combined, f, indent=2)
        os.replace(tmp, combined_json)
        volume.commit()

    # Make sure ordered_tags reflects current disc order (preserves history).
    seen = set(combined["ordered_tags"])
    for p in ckpt_paths:
        tag = _tag_of(p)
        if tag not in seen:
            combined["ordered_tags"].append(tag)
            seen.add(tag)

    # ── Loop ────────────────────────────────────────────────────────────
    for ckpt in ckpt_paths:
        tag = _tag_of(ckpt)
        if allow is not None and tag not in allow:
            continue
        if skip_existing and tag in combined["results"] and not force:
            print(f"⏭️  {tag} already in combined JSON — skipping.")
            continue

        gen_file = os.path.join(gen_root, f"{tag}.jsonl")
        eval_json = os.path.join(gen_root, f"{tag}_eval_results.json")

        print(f"\n▶ [{tag}] generation → {gen_file}")
        try:
            if dataset.lower() == "sb_bench":
                subprocess.run([
                    "python", "-m", "modules.inference.generate_sb_bench_answers",
                    "--data_path", data_path,
                    "--output_jsonl", gen_file,
                    "--batch_size", "8",
                    "--split", "test",
                    "--split_indices_path", SPLIT_INDICES_PATH,
                    "--checkpoint_dir", ckpt,
                    "--shuffle_answers", shuffle_answers,
                ], check=True)
            elif dataset.lower() == "pope":
                subprocess.run([
                    "python", "-m", "modules.inference.generate_answers",
                    "--data_path", data_path,
                    "--output_jsonl", gen_file,
                    "--batch_size", "16",
                    "--checkpoint_dir", ckpt,
                ], check=True)
            else:
                print(f"❌ Unknown dataset '{dataset}' — skipping {tag}")
                continue
        except subprocess.CalledProcessError as e:
            print(f"❌ Generation failed for {tag}: {e}. Continuing to next ckpt.")
            continue

        print(f"▶ [{tag}] evaluation → {eval_json}")
        try:
            if dataset.lower() == "sb_bench":
                subprocess.run([
                    "python", "-m", "modules.evaluation.eval_sb_bench",
                    "--gen_file", gen_file,
                    "--output_json", eval_json,
                ], check=True)
            else:
                # POPE eval requires gt_file; user can run it separately.
                print(f"⚠️  POPE eval needs --gt_file; skipping eval step for {tag}.")
                continue
        except subprocess.CalledProcessError as e:
            print(f"❌ Eval failed for {tag}: {e}. Continuing to next ckpt.")
            continue

        # Load per-ckpt result and merge into combined.
        try:
            with open(eval_json) as f:
                per_ckpt = json.load(f)
        except Exception as e:
            print(f"❌ Could not load {eval_json}: {e}")
            continue

        combined["results"][tag] = {
            "metrics": per_ckpt.get("metrics", {}),
            "per_category": per_ckpt.get("per_category", {}),
            "checkpoint_dir": ckpt,
            "gen_file": gen_file,
            "eval_json": eval_json,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _flush()
        acc = per_ckpt.get("metrics", {}).get("accuracy", float("nan"))
        print(f"✅ [{tag}] accuracy={acc:.4f} → combined JSON updated.")

    # Final flush (in case nothing was processed, still write summary).
    _flush()
    print("\n" + "=" * 60)
    print(f"SWEEP COMPLETE → {combined_json}")
    best = combined["summary"].get("best")
    if best:
        print(f"Best: {best['tag']}  accuracy={best['accuracy']:.4f}")
    print("=" * 60)


# ─── Phase 0.7 P3: POPE regression eval (A100-80GB) ─────────────────────────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    volumes={"/mnt/data": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_pope_eval(
    checkpoint_dir: str = "",
    output_dir: str = "/mnt/data/phase07_pope",
    tag: str = "",
    data_path: str = "/mnt/data/pope_data/pope_data.parquet",
    gt_file: str = "/mnt/data/pope_data/pope_data.jsonl",
    batch_size: int = 16,
):
    """Phase 0.7 P3: generate POPE answers from a (LoRA) checkpoint and score.

    Pass --checkpoint-dir "" (empty) to score the vanilla base model. The tag
    defaults to basename(checkpoint_dir) or "base" if vanilla.
    """
    _setup_env()
    os.makedirs(output_dir, exist_ok=True)

    if not tag:
        tag = os.path.basename(checkpoint_dir.rstrip("/")) if checkpoint_dir else "base"
    gen_file = os.path.join(output_dir, f"{tag}_pope_gen.jsonl")
    summary_json = os.path.join(output_dir, f"{tag}_pope_results.json")

    print(f"▶ POPE generation [{tag}] → {gen_file}")
    gen_cmd = [
        "python", "-m", "modules.inference.generate_answers",
        "--data_path", data_path,
        "--output_jsonl", gen_file,
        "--batch_size", str(batch_size),
    ]
    if checkpoint_dir:
        gen_cmd += ["--checkpoint_dir", checkpoint_dir]
    subprocess.run(gen_cmd, check=True)

    print(f"▶ POPE eval [{tag}] → {summary_json}")
    # eval_pope.py auto-writes <gen_file>.replace('.jsonl','_pope_results.json')
    subprocess.run([
        "python", "-m", "modules.evaluation.eval_pope",
        "--gt_files", gt_file,
        "--gen_files", gen_file,
    ], check=True)

    # Move the auto-named output to the canonical summary path
    auto_out = gen_file.replace(".jsonl", "_pope_results.json")
    if os.path.exists(auto_out) and auto_out != summary_json:
        os.replace(auto_out, summary_json)
    volume.commit()
    print(f"✅ POPE eval [{tag}] complete → {summary_json}")


# ─── Phase 0.7 G4a: VLBiasBench transfer eval (A100-80GB) ───────────────────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    volumes={"/mnt/data": volume},
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_vlbiasbench_eval(
    checkpoint_dir: str = "",
    output_dir: str = "/mnt/data/phase07_vlbiasbench",
    tag: str = "",
    data_path: str = "/mnt/data/vlbiasbench_data/vlbiasbench_close_ended.parquet",
    batch_size: int = 8,
    num_samples: int = 2000,
    condition: str = "all",
    qformat: str = "base,scene,scene_text",
    seed: int = 42,
):
    """Phase 0.7 G4a: VLBiasBench close-ended transfer eval.

    Pass --checkpoint-dir "" (empty) to score the vanilla base model. The tag
    defaults to basename(checkpoint_dir) or "base" if vanilla.
    """
    _setup_env()
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"📥 VLBiasBench parquet missing → building it now ({data_path})")
        subprocess.run(["python", "-m", "modules.data.load_vlbiasbench"], check=True)
        volume.commit()

    if not tag:
        tag = os.path.basename(checkpoint_dir.rstrip("/")) if checkpoint_dir else "base"
    gen_file = os.path.join(output_dir, f"{tag}_vlbias_gen.jsonl")
    summary_json = os.path.join(output_dir, f"{tag}_vlbias_results.json")

    print(f"▶ VLBiasBench generation [{tag}] (n={num_samples}, cond={condition}, qf={qformat}) → {gen_file}")
    gen_cmd = [
        "python", "-m", "modules.inference.generate_vlbiasbench_answers",
        "--data_path", data_path,
        "--image_root", "/mnt/data/vlbiasbench_data/unpacked/close_ended/images",
        "--output_jsonl", gen_file,
        "--batch_size", str(batch_size),
        "--num_samples", str(num_samples),
        "--condition", condition,
        "--qformat", qformat,
        "--seed", str(seed),
    ]
    if checkpoint_dir:
        gen_cmd += ["--checkpoint_dir", checkpoint_dir]
    subprocess.run(gen_cmd, check=True)

    print(f"▶ VLBiasBench eval [{tag}] → {summary_json}")
    subprocess.run([
        "python", "-m", "modules.evaluation.eval_vlbiasbench",
        "--gen_file", gen_file,
        "--output_json", summary_json,
    ], check=True)

    volume.commit()
    print(f"✅ VLBiasBench eval [{tag}] complete → {summary_json}")


# ─── Phase 0.7 T1.1: Offline reward scoring on VLBiasBench (A100-80GB) ──────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    volumes={"/mnt/data": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_vlbias_offline_score(
    variant: str = "base",
    head_type: str = "svm,pca",
    gen_dir: str = "/mnt/data/phase07_vlbiasbench",
    heads_root: str = "/mnt/data/generated_heads_letter_post_letter",
    use_kept_heads: bool = True,
    output_dir: str = "/mnt/data/phase07_offline_reward",
    reward_base: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    parquet: str = "/mnt/data/vlbiasbench_data/vlbiasbench_close_ended.parquet",
    image_root: str = "/mnt/data/vlbiasbench_data/unpacked/close_ended/images",
    batch_size: int = 4,
    max_samples: int = 0,
    max_per_cell: int = 0,
    max_pixels: int = 0,   # 0 = uncapped, matches generate_vlbiasbench_answers.py
    token_position: str = "post_letter",
):
    """Phase 0.7 T1.1: score an existing <variant>_vlbias_gen.jsonl with the
    reward heads. No PPO, no new training. Scores with ALL head types in one
    forward pass (one model load, one embedding extraction, multiple head matmuls).

    --head-type accepts comma-separated list: "svm,pca" (default) to score both
    in a single ~60min GPU session rather than 2x ~60min each.
    """
    _setup_env()
    os.makedirs(output_dir, exist_ok=True)

    gen_jsonl = os.path.join(gen_dir, f"{variant}_vlbias_gen.jsonl")
    if not os.path.exists(gen_jsonl):
        raise FileNotFoundError(f"gen_jsonl missing: {gen_jsonl}")

    # Parse comma-separated head types
    head_types = [h.strip() for h in head_type.split(",") if h.strip()]
    heads_dirs = []
    kept_heads_args = []
    for ht in head_types:
        subdir = "sb_bench-SVM-component" if ht.lower() == "svm" else "sb_bench-PCA-component"
        heads_dirs.append(os.path.join(heads_root, subdir))
        if use_kept_heads:
            cand = (os.path.join(heads_root, "kept_heads.json")
                    if ht.lower() == "svm"
                    else os.path.join(heads_root, "kept_heads_pca.json"))
            kept_heads_args.append(cand if os.path.exists(cand) else "none")
        else:
            kept_heads_args.append("none")

    print(f"▶ Offline reward scoring [variant={variant} heads={head_types}]")
    print(f"  gen_jsonl={gen_jsonl}")
    for ht, hd, kh in zip(head_types, heads_dirs, kept_heads_args):
        print(f"  [{ht}] dir={hd}  kept={kh}")

    cmd = [
        "python", "-m", "modules.evaluation.score_vlbias_offline",
        "--gen_jsonl", gen_jsonl,
        "--parquet", parquet,
        "--image_root", image_root,
        "--reward_base", reward_base,
        "--heads_dir", *heads_dirs,
        "--head_type", *head_types,
        "--variant", variant,
        "--output_dir", output_dir,
        "--batch_size", str(batch_size),
        "--max_samples", str(max_samples),
        "--max_per_cell", str(max_per_cell),
        "--max_pixels", str(max_pixels),
        "--token_position", token_position,
        "--kept_heads", *kept_heads_args,
    ]
    subprocess.run(cmd, check=True)

    volume.commit()
    print(f"✅ Offline reward scoring complete for variant={variant}")


# ─── Phase 0.8 A1: Per-layer linear probes on hidden states (A100-80GB) ─────
@app.function(
    image=vlm_image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    volumes={"/mnt/data": volume},
    timeout=10800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_layer_probe(
    variant: str = "base",
    gen_dir: str = "/mnt/data/phase07_vlbiasbench",
    output_dir: str = "/mnt/data/phase08_probe_results",
    reward_base: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    parquet: str = "/mnt/data/vlbiasbench_data/vlbiasbench_close_ended.parquet",
    image_root: str = "/mnt/data/vlbiasbench_data/unpacked/close_ended/images",
    batch_size: int = 4,
    max_per_cell: int = 30,           # 10 axes x 3 conds x 30 = 900 records on primary
    max_pixels: int = 0,
    holdout_tag: str = "base_qformat_text",  # gen JSONL tag for the qformat=text holdout
    max_holdout: int = 100,
    token_position: str = "post_letter",
    cv_folds: int = 5,
    save_plot: bool = True,
    cache_npz: str = "",              # set to a path to cache hidden states
):
    """Phase 0.8 A1: extract per-LM-layer hidden states at `post_letter` and
    run 3 logistic-regression probes per layer (P1/P2/P3). Decisive metric
    is `p3_acc_holdout` evaluated on a separate `qformat=text` gen JSONL.

    Pre-req: a primary gen JSONL at <gen_dir>/<variant>_vlbias_gen.jsonl
    (already produced by phase07_g4_vlbiasbench.sh). For the holdout, the
    caller is responsible for first running run_vlbiasbench_eval with
    qformat='text' and tag=<holdout_tag>; if that file is missing the probe
    runs without holdout and prints a warning.
    """
    _setup_env()
    os.makedirs(output_dir, exist_ok=True)

    gen_jsonl = os.path.join(gen_dir, f"{variant}_vlbias_gen.jsonl")
    if not os.path.exists(gen_jsonl):
        raise FileNotFoundError(f"primary gen_jsonl missing: {gen_jsonl}")

    holdout_jsonl = os.path.join(gen_dir, f"{holdout_tag}_vlbias_gen.jsonl")
    if not os.path.exists(holdout_jsonl):
        print(f"⚠  holdout missing: {holdout_jsonl} — proceeding without it. "
              "Run `run_vlbiasbench_eval(tag='{holdout_tag}', qformat='text', "
              "checkpoint_dir='')` first to enable bias-blind selection.")
        holdout_jsonl = ""

    print(f"▶ Layer probe [variant={variant}]  primary={gen_jsonl}  holdout={holdout_jsonl or '<none>'}")

    cmd = [
        "python", "-m", "modules.evaluation.probe_layers",
        "--gen_jsonl", gen_jsonl,
        "--parquet", parquet,
        "--image_root", image_root,
        "--reward_base", reward_base,
        "--variant", variant,
        "--output_dir", output_dir,
        "--batch_size", str(batch_size),
        "--max_per_cell", str(max_per_cell),
        "--max_pixels", str(max_pixels),
        "--token_position", token_position,
        "--cv_folds", str(cv_folds),
    ]
    if holdout_jsonl:
        cmd += ["--holdout_gen_jsonl", holdout_jsonl,
                "--max_holdout", str(max_holdout)]
    if save_plot:
        cmd += ["--save_plot"]
    if cache_npz:
        cmd += ["--cache_npz", cache_npz]

    subprocess.run(cmd, check=True)

    volume.commit()
    print(f"✅ Layer probe complete for variant={variant}")


# ─── Phase 0.7 G4a: VLBiasBench EDA (CPU only) ──────────────────────────────
@app.function(
    image=vlm_image,
    gpu=None,
    cpu=2.0,
    memory=8192,
    volumes={"/mnt/data": volume},
    timeout=1800,
)
def run_vlbiasbench_eda(
    unpacked_root: str = "/mnt/data/vlbiasbench_data/unpacked/close_ended",
    n_sample_records: int = 3,
):
    """Inspect the VLBiasBench close-ended unpacked tree on the volume.

    Prints: directory structure, JSON file inventory, per-file record counts,
    per-record schema (keys + types), sample records, condition/label
    distributions, image-path format examples. Read-only.
    """
    _setup_env()
    import json as _json
    from collections import Counter, defaultdict
    from pathlib import Path

    root = Path(unpacked_root)
    if not root.exists():
        print(f"❌ Unpacked root does not exist: {root}")
        return

    json_root = root / "json"
    img_root = root / "images"
    print(f"\n{'='*72}")
    print(f"VLBiasBench EDA — {root}")
    print(f"{'='*72}\n")

    # ── 1. Top-level dirs ──────────────────────────────────────────────
    print("── Top-level under close_ended/ ────────────────────────────")
    for p in sorted(root.iterdir()):
        kind = "DIR " if p.is_dir() else "FILE"
        print(f"  {kind}  {p.name}")
    print()

    # ── 2. JSON dir structure ──────────────────────────────────────────
    print("── json/ subdirs ──────────────────────────────────────────")
    if not json_root.exists():
        print(f"  ❌ Missing {json_root}")
        return
    json_dirs = sorted([d for d in json_root.iterdir() if d.is_dir()])
    for d in json_dirs:
        files = sorted(d.glob("*.json"))
        print(f"  {d.name:20s}  {len(files):3d} JSON files")
        for f in files[:3]:
            print(f"      e.g. {f.name}")
        if len(files) > 3:
            print(f"      ... ({len(files) - 3} more)")
    print()

    # ── 3. Per-file record counts + first-record schema ────────────────
    print("── First record per JSON file ─────────────────────────────")
    all_keys = Counter()
    all_records = []
    file_inventory = []
    for d in json_dirs:
        for jf in sorted(d.glob("*.json")):
            with open(jf, "r") as f:
                data = _json.load(f)
            if isinstance(data, dict):
                data = [data]
            file_inventory.append((d.name, jf.name, len(data)))
            all_records.extend([(d.name, r) for r in data])
            for r in data:
                all_keys.update(r.keys())

    # File inventory table
    print(f"{'subdir':22s} {'file':40s} {'records':>10s}")
    print("-" * 75)
    for sub, fn, n in file_inventory:
        print(f"{sub:22s} {fn:40s} {n:>10d}")
    total_records = sum(n for _, _, n in file_inventory)
    print(f"{'TOTAL':22s} {'':40s} {total_records:>10d}")
    print()

    # ── 4. Key frequency across all records ────────────────────────────
    print("── Key frequency across all records ───────────────────────")
    for k, cnt in all_keys.most_common():
        pct = 100 * cnt / total_records
        print(f"  {k:30s}  {cnt:>7d}  ({pct:5.1f}%)")
    print()

    # ── 5. Sample records ──────────────────────────────────────────────
    print(f"── {n_sample_records} sample records (first from each of first {n_sample_records} files) ──")
    shown = 0
    for sub, rec in all_records:
        if shown >= n_sample_records:
            break
        print(f"\n[from subdir={sub}]")
        for k, v in rec.items():
            vs = str(v)
            if len(vs) > 200:
                vs = vs[:200] + " ...[truncated]"
            print(f"  {k}: {vs}")
        shown += 1
    print()

    # ── 6. Distributions on key fields ─────────────────────────────────
    print("── condition distribution ──────────────────────────────────")
    cond_counter = Counter(r.get("condition", "<MISSING>") for _, r in all_records)
    for k, v in cond_counter.most_common():
        print(f"  {k:20s}  {v:>7d}")
    print()

    print("── category distribution (internal field) ─────────────────")
    cat_counter = Counter(r.get("category", "<MISSING>") for _, r in all_records)
    for k, v in cat_counter.most_common(20):
        print(f"  {str(k):30s}  {v:>7d}")
    if len(cat_counter) > 20:
        print(f"  ... ({len(cat_counter) - 20} more)")
    print()

    print("── label distribution ──────────────────────────────────────")
    label_counter = Counter(r.get("label", "<MISSING>") for _, r in all_records)
    for k, v in sorted(label_counter.items(), key=lambda x: str(x[0])):
        print(f"  label={str(k):15s}  {v:>7d}")
    print()

    # ── 7. label × condition cross-tab ─────────────────────────────────
    print("── label × condition cross-tab ────────────────────────────")
    cross = defaultdict(int)
    for _, r in all_records:
        cross[(r.get("condition", "<MISSING>"), r.get("label", "<MISSING>"))] += 1
    conds = sorted({c for c, _ in cross.keys()}, key=str)
    labs = sorted({l for _, l in cross.keys()}, key=str)
    header = f"{'condition':15s} " + " ".join(f"label={str(l):>8s}" for l in labs)
    print(header)
    for c in conds:
        row = f"{str(c):15s} " + " ".join(f"{cross[(c, l)]:>14d}" for l in labs)
        print(row)
    print()

    # ── 8. image_path examples + existence check ──────────────────────
    print("── image_path examples (first 5) ──────────────────────────")
    img_examples = []
    for _, r in all_records[:200]:
        if "image_path" in r:
            img_examples.append(r["image_path"])
        if len(img_examples) >= 5:
            break
    for p in img_examples:
        full = img_root / p
        exists = "✅" if full.exists() else "❌"
        print(f"  {exists}  {p}")
    print()

    # ── 9. Sample images on disk ───────────────────────────────────────
    print("── images/ subdir layout (depth 1) ────────────────────────")
    if img_root.exists():
        sub_imgs = sorted([p for p in img_root.iterdir() if p.is_dir()])[:10]
        for p in sub_imgs:
            count = sum(1 for _ in p.rglob("*"))
            print(f"  {p.name}/  ({count} entries)")
        loose = sorted([p for p in img_root.iterdir() if p.is_file()])[:5]
        if loose:
            print(f"  + {len(loose)} loose files at root, e.g. {[p.name for p in loose[:3]]}")
    else:
        print(f"  ❌ {img_root} missing")
    print()

    print(f"{'='*72}")
    print("EDA complete.")
    print(f"{'='*72}\n")


# ─── Local Entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(phase: str = "all", epochs: int = 1, resume: str = "", output_dir: str = "", dataset: str = "sb_bench", model_family: str = "qwen", gt_file: str = "", gen_file: str = "", vanilla: bool = False,
         kl_beta: float = 0.1, ppo_clip_range: float = 0.2, learning_rate: float = 1e-5,
         lora_r: int = 16, lora_alpha: int = 32, max_train_samples: int = 0, eta: float = 0.01,
         num_heads: int = 9, tau: float = 1.0, lambda_causal: float = 0.5,
         delta_margin: float = 1.0, lambda_dispersive: float = 0.01,
         logit_reward_coef: float = 0.1, head_type: str = "svm",
         max_gen_tokens: int = 256, batch_size: int = 12,
         reward_mode: str = "svm",
         # Phase 0 knobs
         p0_num_samples: int = 256, p0_batch_size: int = 4,
         p0_layers: str = "12,18,24,30,34",
         p0_lambdas: str = "-3.0,-1.5,0.0,1.5,3.0",
         p0_heads: str = "",
         p0_output_json: str = "",
         vanilla_jsonl: str = "/mnt/data/sb_bench_vanilla_baseline/sb_bench_generations.jsonl",
         v4_jsonl: str = ""):
    """
    Run pipeline phases with optimized GPU allocation.
    
    Phases:
      setup       - Download data/models (CPU, 16GB RAM)
      preprocess  - Build dataset chunks (CPU, 128GB RAM) 
      inference   - Extract embeddings (A100-80GB)
      phase1      - preprocess + inference combined
      phase2      - Generate DRM heads (CPU, 32GB RAM)
      phase0a     - Activation-steering sanity gate on SVM heads (A100-80GB)
      phase0b     - Reward / metric correlation on SVM heads (A100-80GB)
      phase0c     - McNemar paired test on vanilla vs v4 JSONLs (CPU)
                    Requires --v4-jsonl <path>; vanilla defaults to
                    /mnt/data/sb_bench_vanilla_baseline/sb_bench_generations.jsonl
      phase0      - Runs phase0a + phase0b; runs phase0c iff --v4-jsonl given
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

    # ── Phase 0 sanity gates ──────────────────────────────────────────────────
    if phase == "phase0a":
        out = p0_output_json or "/mnt/data/phase0/steering.json"
        run_phase0_steering.remote(
            num_samples=p0_num_samples, batch_size=p0_batch_size,
            layers=p0_layers, lambdas=p0_lambdas, heads=p0_heads,
            output_json=out,
        )

    if phase == "phase0b":
        out = p0_output_json or "/mnt/data/phase0/correlation.json"
        run_phase0_correlation.remote(
            num_samples=p0_num_samples, output_json=out,
        )

    if phase == "phase0c":
        if not v4_jsonl:
            raise SystemExit(
                "phase0c requires --v4-jsonl <path>  (and optionally --vanilla-jsonl)"
            )
        run_phase0_mcnemar.remote(vanilla_jsonl=vanilla_jsonl, v4_jsonl=v4_jsonl)

    if phase == "phase0":
        run_phase0_steering.remote(
            num_samples=p0_num_samples, batch_size=p0_batch_size,
            layers=p0_layers, lambdas=p0_lambdas, heads=p0_heads,
            output_json=p0_output_json or "/mnt/data/phase0/steering.json",
        )
        run_phase0_correlation.remote(
            num_samples=p0_num_samples,
            output_json="/mnt/data/phase0/correlation.json",
        )
        if v4_jsonl:
            run_phase0_mcnemar.remote(vanilla_jsonl=vanilla_jsonl, v4_jsonl=v4_jsonl)
        else:
            print("ℹ️  Skipping Phase 0c (McNemar): no --v4-jsonl provided.")
    
    resume_ckpt = resume if resume else None
    train_kwargs = dict(
        kl_beta=kl_beta, ppo_clip_range=ppo_clip_range, learning_rate=learning_rate,
        lora_r=lora_r, lora_alpha=lora_alpha, eta=eta,
        max_train_samples=max_train_samples if max_train_samples > 0 else None,
        num_heads=num_heads, tau=tau, lambda_causal=lambda_causal,
        delta_margin=delta_margin, lambda_dispersive=lambda_dispersive,
        logit_reward_coef=logit_reward_coef, head_type=head_type,
        max_gen_tokens=max_gen_tokens, batch_size=batch_size,
        reward_mode=reward_mode,
    )
    
    if phase in ["all", "train"]:
        train_out = output_dir if output_dir else "/mnt/data/output_ppo_debiased"
        run_training.remote(epochs=epochs, output_dir=train_out, resume_from=resume_ckpt, dataset=dataset, model_family=model_family, **train_kwargs)
    
    if phase == "train5":
        run_training.remote(epochs=5, output_dir="/mnt/data/output_ppo_debiased_5ep", resume_from=resume_ckpt, dataset=dataset, model_family=model_family, **train_kwargs)
    
    if phase == "train10":
        run_training.remote(epochs=10, output_dir="/mnt/data/output_ppo_debiased_10ep", resume_from=resume_ckpt, dataset=dataset, model_family=model_family, **train_kwargs)

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
