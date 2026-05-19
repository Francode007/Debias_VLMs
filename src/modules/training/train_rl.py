"""
train_rl.py — PPO RL Debiasing Orchestrator

This script is the thin entrypoint that wires together modular components:

  modules/training/args.py        → argument parsing
  modules/training/drm_loader.py  → DRM (PCA) head loading
  modules/training/setup.py       → Accelerator + model + dataloader
  modules/training/checkpoint.py  → checkpoint save / load
  modules/training/ppo_loop.py    → per-epoch / per-batch training loop

  modules/fast_rl.py              → Fast-RL mirror descent node
  modules/custom_vlm_ppo_trainer  → PPOVLMController (step logic)

Usage (local dry-run):
    python train_rl.py --use_smallset --dataset_name sb_bench --model_family qwen

Usage (Modal — via run_modal.py):
    modal run run_modal.py --phase train --dataset sb_bench --model_family qwen
"""

import functools
import json
import logging
import warnings

# Suppress kernel version warning from accelerate (Modal host kernel is 4.4.0)
warnings.filterwarnings("ignore", message=".*Detected kernel version.*")

import torch

from modules.rl_components import FastRLNode
from modules.training.args import parse_training_args
from modules.training.checkpoint import load_checkpoint, save_checkpoint, save_final_model
from modules.training.drm_loader import load_pca_components
from modules.training.ppo_loop import run_ppo_loop
from modules.training.setup import (
    build_accelerator,
    build_dataloader,
    build_policy_model,
    build_ppo_controller,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    import time
    t_start_setup = time.time()

    # ── 1. Arguments ──────────────────────────────────────────────────────────
    args = parse_training_args()

    # ── 2. Accelerator ────────────────────────────────────────────────────────
    accelerator = build_accelerator(args)

    # ── 3. Policy model + processor ───────────────────────────────────────────
    logger.info("Loading policy model and injecting LoRA adapter...")
    active_policy, processor = build_policy_model(args, accelerator)

    # ── 4. DRM reward heads ───────────────────────────────────────────────────
    logger.info("Loading Phase 1 PCA DRM heads...")
    try:
        reward_heads_weight = load_pca_components(
            args.reward_heads_dir, args.num_heads, accelerator.device
        )
        args.num_heads = reward_heads_weight.shape[0]   # actual count after load
    except Exception as e:
        logger.error(f"Could not load reward heads: {e}")
        return

    # ── 5. Fast-RL node ───────────────────────────────────────────────────────
    fast_rl_node = FastRLNode(
        num_heads=args.num_heads,
        strategy=args.fast_rl_strategy,
        eta=args.eta,
        device=accelerator.device,
    )

    # ── 6. PPO controller ─────────────────────────────────────────────────────
    ppo_controller = build_ppo_controller(
        active_policy, reward_heads_weight, accelerator, fast_rl_node, args
    )

    # ── 7. Optimizers ─────────────────────────────────────────────────────────
    optimizer_policy = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, active_policy.parameters()), lr=1e-5
    )
    optimizer_value = torch.optim.AdamW(
        ppo_controller.value_head.parameters(), lr=1e-5
    )

    # ── 8. Dataset + DataLoader ───────────────────────────────────────────────
    logger.info("Building RL dataset and DataLoader...")
    _, train_dataloader, _ = build_dataloader(args, processor, accelerator, ppo_controller)

    # ── 9. Accelerator prepare ────────────────────────────────────────────────
    (
        active_policy,
        ppo_controller.value_head,
        optimizer_policy,
        optimizer_value,
        train_dataloader,
    ) = accelerator.prepare(
        active_policy,
        ppo_controller.value_head,
        optimizer_policy,
        optimizer_value,
        train_dataloader,
    )

    setup_duration = time.time() - t_start_setup
    logger.info(f"Setup completed in {setup_duration:.2f}s")

    # ── 10. Resume from checkpoint (optional) ─────────────────────────────────
    start_epoch, resume_step, global_step = 0, 0, 0
    if args.resume_from_checkpoint and __import__("os").path.exists(
        args.resume_from_checkpoint
    ):
        start_epoch, resume_step, global_step = load_checkpoint(
            ckpt_dir=args.resume_from_checkpoint,
            accelerator=accelerator,
            active_policy=active_policy,
            ppo_controller=ppo_controller,
            fast_rl_node=fast_rl_node,
            optimizer_policy=optimizer_policy,
            optimizer_value=optimizer_value,
        )

    # ── 11. Training loop ─────────────────────────────────────────────────────
    logger.info("Starting PPO training...")
    _save_ckpt = functools.partial(
        save_checkpoint,
        accelerator=accelerator,
        active_policy=active_policy,
        ppo_controller=ppo_controller,
        fast_rl_node=fast_rl_node,
        optimizer_policy=optimizer_policy,
        optimizer_value=optimizer_value,
        output_dir=args.output_dir,
        global_step=global_step,
    )

    timing = run_ppo_loop(
        args=args,
        accelerator=accelerator,
        active_policy=active_policy,
        ppo_controller=ppo_controller,
        optimizer_policy=optimizer_policy,
        optimizer_value=optimizer_value,
        train_dataloader=train_dataloader,
        fast_rl_node=fast_rl_node,
        start_epoch=start_epoch,
        resume_step=resume_step,
        global_step=global_step,
        save_checkpoint_fn=_save_ckpt,
    )

    # ── 12. Final model save ──────────────────────────────────────────────────
    save_final_model(
        accelerator=accelerator,
        active_policy=active_policy,
        ppo_controller=ppo_controller,
        fast_rl_node=fast_rl_node,
        output_dir=args.output_dir,
    )

    timing["setup_time"] = setup_duration
    print(f"\nTIMING_REPORT_JSON: {json.dumps(timing)}")


if __name__ == "__main__":
    main()
