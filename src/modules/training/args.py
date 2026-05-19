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
        default=100,
        help="Number of PCA components (DRM heads) to load",
    )

    # Fast-RL arguments
    parser.add_argument(
        "--fast_rl_strategy",
        type=str,
        default="exponentiated",
        choices=["exponentiated", "projected", "adam"],
        help="Mirror Descent update strategy for Fast-RL",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.01,
        help="Learning rate for the Fast-RL node",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.1,
        help="KL divergence penalty coefficient",
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

    return parser.parse_args()
