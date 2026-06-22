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
        default=0,
        help="Number of DRM reward heads to load. 0 (default) = auto-detect "
             "and load ALL component*.pth files present in --reward_heads_dir "
             "(9 for SB-Bench SVM, 50 for PCA). Set a positive integer to "
             "truncate to the first N sorted components.",
    )
    parser.add_argument(
        "--kept_heads_filter",
        type=str,
        default=None,
        help="Phase 0.6 D2: path to kept_heads.json produced by evaluate_drm_heads.py. "
             "When set, only heads whose original index is in kept_indices are loaded.",
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
    # Phase 0.8 §4½.15: cap image resolution at the processor layer so that
    # high-res SB-Bench composites don't blow past --max_length (which would
    # silently drop them in rl_dataset_builder.py) and don't OOM at rollout.
    # 512x512 ≈ 262k px ≈ ~336 visual tokens with Qwen2.5-VL's 28x28 patch +
    # 2x2 merger. Matches generate_sb_bench_answers.py default.
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=512 * 512,
        help="Cap image resolution (pixels) at the processor. 0 disables.",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=0,
        help="Floor on image resolution (pixels). 0 disables.",
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
        choices=["svm", "binary", "bias_aligned"],
        help=(
            "Reward signal: 'svm' (dense SVM projection, Phase 0.5/0.6), "
            "'binary' (+1/-1 correctness, Phase 0.6), or 'bias_aligned' "
            "(Phase 0.8 probe-as-head: correctness + negated probe "
            "projection at the answer-letter position + ambig-preservation "
            "bonus, validated bi-directionally on VLBias ↔ SB-Bench at L13)."
        ),
    )
    # ── Phase 0.8 (probe-as-head) controls ─────────────────────────────────
    parser.add_argument(
        "--reward_head_layer",
        type=int,
        default=-2,
        help=(
            "Index into outputs.hidden_states (0 = embedding, 1..N = block "
            "outputs, -2 = penultimate). Phase 0.8 uses 13 to match the "
            "probe-fit lead layer chosen in §4½.10. Default -2 reproduces "
            "the legacy penultimate-layer behaviour for backward compat."
        ),
    )
    parser.add_argument(
        "--bias_aligned_coef",
        type=float,
        default=1.0,
        help="Coefficient w_bias on the negated probe-projection term (Phase 0.8).",
    )
    parser.add_argument(
        "--ambig_preservation_coef",
        type=float,
        default=0.5,
        help=(
            "Coefficient w_ambig on the ambig-preservation bonus 1[gold==C∧pred==C] "
            "(§4½.11 ambig-collapse guard)."
        ),
    )
    parser.add_argument(
        "--correctness_coef",
        type=float,
        default=1.0,
        help="Coefficient w_corr on the binary correctness term (Phase 0.8).",
    )
    # ── Phase 0.9 R3 (multi-layer ensemble reward) ────────────────────────
    parser.add_argument(
        "--ensemble_bundle_dir",
        type=str,
        default=None,
        help=(
            "Phase 0.9 R3: directory containing a multi-layer probe bundle "
            "(L{N}.pth files + ensemble_metadata.json) built by "
            "scripts/phase09_build_ensemble_bundle.py. When set, the trainer "
            "loads N per-layer probe heads via load_ensemble_probe_bundle, "
            "captures hidden states at each layer in the bundle window, "
            "z-normalises per-layer using offline-calibrated mu/sigma "
            "baked into the metadata, then mean-pools across layers to a "
            "single scalar reward (matches Phase0.8/ensemble/README.md "
            "zscore_mean recipe). Bias-aligned mode is required; the "
            "single-layer --reward_head_layer is ignored when this is set."
        ),
    )
    parser.add_argument(
        "--ensemble_layers",
        type=str,
        default=None,
        help=(
            "Phase 0.9 R3: comma-separated layer indices "
            "(e.g. '17,21,25,29,33'). If set, overrides the bundle's "
            "layers list to a subset. Each requested layer must be present "
            "in the bundle. Leave unset to use all layers in the bundle."
        ),
    )
    parser.add_argument(
        "--ensemble_pool",
        type=str,
        default=None,
        choices=[None, "zmean", "mean", "max"],
        help=(
            "Phase 0.9 R3: pool strategy across the ensemble window. "
            "'zmean' = per-layer z-score then mean (recommended, "
            "Phase0.8/ensemble/README.md Test A winner). "
            "'mean' = mean of raw projections (deep layers dominate). "
            "'max' = max z-score. Leave unset to use the bundle metadata's "
            "default (typically 'zmean')."
        ),
    )
    parser.add_argument(
        "--use_frozen_phi",
        action="store_true",
        help="Phase 0.6 D1: project DRM reward heads against the LoRA-disabled "
             "(frozen) penultimate hidden state \u03c6_ref instead of the active "
             "\u03c6_active. Required for SVM/PCA rewards whose heads were fit on "
             "the un-adapted base model, to prevent representation hacking.",
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

    # ── Early stopping (eval-driven, based on rolling training accuracy) ──────
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help=(
            "Stop training if rolling binary_accuracy stays more than "
            "--early_stop_threshold below best for this many consecutive "
            "batch steps. 0 disables."
        ),
    )
    parser.add_argument(
        "--early_stop_threshold",
        type=float,
        default=0.05,
        help="Fraction below best accuracy that triggers patience countdown.",
    )
    parser.add_argument(
        "--early_stop_window",
        type=int,
        default=10,
        help="Rolling window size for smoothing binary_accuracy before comparison.",
    )

    # ── Gradient clipping ─────────────────────────────────────────────────────
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help=(
            "Max L2 norm for gradient clipping (policy + value head). "
            "Lower values (e.g. 0.1) tame the warmup-end policy lurch that "
            "triggered the Phase 0 KL spike at step ~40."
        ),
    )

    # ── Mid-training held-out eval ────────────────────────────────────────────
    parser.add_argument(
        "--midtrain_eval_every_steps",
        type=int,
        default=0,
        help=(
            "Run constrained greedy eval on a held-out subset every N "
            "micro-batch steps. 0 disables. Result logged to metrics.jsonl "
            "as `midtrain_eval_acc`."
        ),
    )
    parser.add_argument(
        "--midtrain_eval_samples",
        type=int,
        default=64,
        help=(
            "Number of samples to hold out from the training set for the "
            "mid-training eval signal. Held out from the TAIL of the dataset "
            "before the train shuffle, so it is deterministic across runs."
        ),
    )
    parser.add_argument(
        "--use_eval_for_early_stop",
        action="store_true",
        help=(
            "When set, the early-stopping rolling-window signal uses "
            "`midtrain_eval_acc` instead of noisy training-batch "
            "`binary_accuracy`. Requires --midtrain_eval_every_steps > 0."
        ),
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Master RNG seed. Drives torch / numpy / random / accelerate "
            "and the train shuffle. Phase 0.8 seed-triple replication uses "
            "{1, 2, 3}. Default 42 reproduces the headline 2k run."
        ),
    )

    return parser.parse_args()
