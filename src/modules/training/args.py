"""
Training argument parsing for the PPO RL debiasing pipeline.

Extracted from train_rl.py to keep the main script clean.
"""
import argparse


def parse_training_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the PPO RL training script.

    Returns:
        argparse.Namespace: Parsed training arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Fast-RL and CAA phase using Custom VLM PPO Training"
    )

    # Model arguments
    parser.add_argument(
        "--policy_model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="The base VLM policy model (HF name or local path)",
    )
    parser.add_argument(
        "--extractor_model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Extractor model name (usually same architecture as policy)",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="qwen",
        help="Model family key used by the model wrapper registry (e.g. 'qwen')",
    )

    # DRM reward head arguments
    parser.add_argument(
        "--reward_heads_dir",
        type=str,
        default="./generated_heads/sb_bench-PCA-component",
        help="Directory containing Phase 1 .pth PCA component files",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=9,
        help="Number of PCA components (DRM heads) to load (9 = one per SB-Bench bias category)",
    )

    # Fast-RL arguments
    parser.add_argument(
        "--fast_rl_strategy",
        type=str,
        default="exponentiated",
        choices=["exponentiated", "projected", "adam"],
        help="(Deprecated) Mirror Descent strategy — now uses deficit-based update",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.01,
        help="(Deprecated) Legacy Fast-RL learning rate — replaced by tau",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Fast-RL entropy temperature (prevents simplex collapse). Higher=more uniform, lower=more focused.",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.1,
        help="KL divergence penalty coefficient",
    )
    parser.add_argument(
        "--ppo_clip_range",
        type=float,
        default=0.2,
        help="PPO surrogate clipping range",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for policy and value optimizers",
    )
    parser.add_argument(
        "--lambda_causal",
        type=float,
        default=0.5,
        help="Causal deviation penalty strength (in reward signal)",
    )
    parser.add_argument(
        "--delta_margin",
        type=float,
        default=1.0,
        help="Tolerance margin for embedding drift before causal penalty activates",
    )
    parser.add_argument(
        "--lambda_dispersive",
        type=float,
        default=0.01,
        help="Dispersive regularization loss weight (anti-collapse)",
    )
    parser.add_argument(
        "--logit_reward_coef",
        type=float,
        default=0.1,
        help="Weight of logit-grounded reward component (prevents null-space hacking)",
    )
    parser.add_argument(
        "--max_gen_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate per PPO step (longer = richer reward signal)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (r parameter)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit training to this many samples (for experiments)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sb_bench",
        help="Dataset name for prompt formatting (registry key, e.g. 'sb_bench', 'pope')",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./sb_bench_data/data",
        help="Path to the parquet dataset directory",
    )

    # Training loop arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_ppo_debiased",
        help="Directory to save PEFT adapter checkpoints",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--use_smallset",
        action="store_true",
        help="Use a tiny subset of the dataset for smoke-testing",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum token sequence length",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from",
    )

    # Train/test split arguments
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "all"],
        help="Which data split to use: 'train' (80%%), 'test' (20%%), or 'all'",
    )
    parser.add_argument(
        "--split_indices_path",
        type=str,
        default=None,
        help="Path to split_indices.json (generated once, reused across phases)",
    )

    # Reward mode
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="svm",
        choices=["svm", "binary"],
        help="Reward signal: 'svm' (dense SVM projection) or 'binary' (+1/-1 correctness)",
    )

    # ── Phase 0 (collapse-mitigation) controls ────────────────────────────
    parser.add_argument(
        "--target_kl",
        type=float,
        default=0.0,
        help=(
            "If >0, enable adaptive KL controller targeting this per-token KL "
            "(Schulman PPO/Ouyang 2022 recipe). 0.02 is the typical RLHF target. "
            "0 disables; kl_beta stays fixed at --kl_beta."
        ),
    )
    parser.add_argument(
        "--kl_adapt_rate",
        type=float,
        default=0.1,
        help="Step size of the adaptive KL controller (fraction per update).",
    )
    parser.add_argument(
        "--kl_beta_min",
        type=float,
        default=0.05,
        help="Floor for the adaptive kl_beta.",
    )
    parser.add_argument(
        "--kl_beta_max",
        type=float,
        default=5.0,
        help="Ceiling for the adaptive kl_beta.",
    )
    parser.add_argument(
        "--value_clip_range",
        type=float,
        default=0.0,
        help=(
            "If >0, clip value-function updates to |v_new - v_old| <= "
            "value_clip_range (Schulman PPO). 0 disables. 0.2 is typical."
        ),
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="linear_warmup",
        choices=["linear_warmup", "cosine", "constant"],
        help=(
            "LR schedule. 'linear_warmup' (default, existing behaviour): warmup "
            "then linear decay to 0. 'cosine': warmup then cosine decay to "
            "min_lr_ratio*lr. 'constant': flat after warmup."
        ),
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.2,
        help="Floor LR as a fraction of peak LR (only used by --lr_schedule cosine).",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total update steps used as linear LR warmup.",
    )
    parser.add_argument(
        "--ckpt_every_steps",
        type=int,
        default=0,
        help=(
            "If >0, save a checkpoint every N PPO steps (in addition to "
            "end-of-epoch). 0 keeps the legacy quarter-epoch cadence."
        ),
    )
    parser.add_argument(
        "--value_learning_rate",
        type=float,
        default=None,
        help="Override value-head LR. Defaults to 5x --learning_rate.",
    )

    return parser.parse_args()
