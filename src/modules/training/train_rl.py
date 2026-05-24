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
        tau=getattr(args, 'tau', 1.0),
        device=accelerator.device,
    )

    # ── 6. PPO controller ─────────────────────────────────────────────────────
    ppo_controller = build_ppo_controller(
        active_policy, reward_heads_weight, accelerator, fast_rl_node, args
    )

    # ── 7. Optimizers ─────────────────────────────────────────────────────────
    lr = getattr(args, 'learning_rate', 1e-5)
    # The value head is a fresh Linear(D, 1) layer that must learn the reward
    # baseline from scratch. Coupling it to the LoRA LR (which has to stay
    # small for stability on a 3B backbone) starves it. Decouple: value LR
    # defaults to 5× policy LR unless explicitly overridden via --value-learning-rate.
    value_lr = getattr(args, 'value_learning_rate', None) or (lr * 5.0)
    logger.info(f"Optimizer LR: policy={lr}, value={value_lr}")
    optimizer_policy = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, active_policy.parameters()), lr=lr
    )
    optimizer_value = torch.optim.AdamW(
        ppo_controller.value_head.parameters(), lr=value_lr
    )

    # ── 8. Dataset + DataLoader ───────────────────────────────────────────────
    logger.info("Building RL dataset and DataLoader...")
    _, train_dataloader, _, eval_dataloader = build_dataloader(
        args, processor, accelerator, ppo_controller
    )

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
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    # CRITICAL: accelerator.prepare() may wrap/return a new model object.
    # Re-bind ppo_controller.policy to the prepared model so all forward
    # passes (old_logprobs, curr_logprobs, generate) use the same weights
    # that the optimizer is actually updating.
    ppo_controller.policy = active_policy

    # Phase 0 hygiene: configurable LR schedule.
    #   - linear_warmup (legacy): warmup then linear decay to 0
    #   - cosine: warmup then cosine decay to min_lr_ratio*peak (Phase 0 recommended)
    #   - constant: warmup then flat
    from transformers import get_linear_schedule_with_warmup
    total_update_steps = max(
        1,
        (len(train_dataloader) * args.epochs) // max(args.gradient_accumulation_steps, 1),
    )
    warmup_ratio = getattr(args, "warmup_ratio", 0.05)
    warmup_steps = int(total_update_steps * warmup_ratio)
    lr_schedule = getattr(args, "lr_schedule", "linear_warmup")
    min_lr_ratio = float(getattr(args, "min_lr_ratio", 0.2))
    logger.info(
        f"LR schedule: {lr_schedule} (warmup {warmup_steps}/{total_update_steps}, "
        f"min_lr_ratio={min_lr_ratio})"
    )

    def _build_scheduler(opt):
        if lr_schedule == "linear_warmup":
            return get_linear_schedule_with_warmup(
                opt,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_update_steps,
            )
        if lr_schedule == "constant":
            from transformers import get_constant_schedule_with_warmup
            return get_constant_schedule_with_warmup(
                opt, num_warmup_steps=warmup_steps
            )
        if lr_schedule == "cosine":
            # Cosine decay from peak -> min_lr_ratio*peak after warmup.
            # We implement directly via LambdaLR so we can pin the floor.
            import math
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(
                    1, total_update_steps - warmup_steps
                )
                cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cos

            return LambdaLR(opt, lr_lambda=lr_lambda)
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    lr_scheduler_policy = _build_scheduler(optimizer_policy)
    lr_scheduler_value = _build_scheduler(optimizer_value)
    lr_scheduler_policy, lr_scheduler_value = accelerator.prepare(
        lr_scheduler_policy, lr_scheduler_value
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
        eval_dataloader=eval_dataloader,
        fast_rl_node=fast_rl_node,
        start_epoch=start_epoch,
        resume_step=resume_step,
        global_step=global_step,
        save_checkpoint_fn=_save_ckpt,
        lr_scheduler_policy=lr_scheduler_policy,
        lr_scheduler_value=lr_scheduler_value,
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
