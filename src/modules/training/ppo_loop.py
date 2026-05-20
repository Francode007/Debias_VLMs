"""
PPO training loop — per-epoch / per-batch execution.

Isolated from setup and checkpointing so it can be independently
tested, profiled, or replaced (e.g. with a GRPO variant).
"""
import logging
import time
from typing import Callable, Dict, Set

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_ppo_loop(
    *,
    args,
    accelerator,
    active_policy,
    ppo_controller,
    optimizer_policy,
    optimizer_value,
    train_dataloader,
    fast_rl_node,
    start_epoch: int,
    resume_step: int,
    global_step: int,
    save_checkpoint_fn: Callable,
) -> Dict:
    """
    Execute the PPO training loop across all epochs.

    Args:
        args:                 Parsed training arguments.
        accelerator:          HF Accelerate instance.
        active_policy:        The (accelerator-prepared) LoRA policy.
        ppo_controller:       PPOVLMController.
        optimizer_policy:     Accelerator-prepared policy optimizer.
        optimizer_value:      Accelerator-prepared value head optimizer.
        train_dataloader:     Accelerator-prepared DataLoader.
        fast_rl_node:         FastRLNode instance.
        start_epoch:          First epoch index (0 for fresh, >0 when resuming).
        resume_step:          First step to execute in start_epoch (0 if not mid-epoch).
        global_step:          Starting global step counter.
        save_checkpoint_fn:   Callable(tag, epoch, step_in_epoch) → None.

    Returns:
        timing_report dict: setup_time, total_training_time, avg_batch_time,
                            num_batches, total_samples.
    """
    total_batches_per_epoch = len(train_dataloader)
    quarter_steps: Set[int] = {
        int(total_batches_per_epoch * q) for q in [0.25, 0.50, 0.75]
    }
    logger.info(
        f"Quarter-epoch checkpoints at steps: {sorted(quarter_steps)} "
        f"(total batches/epoch: {total_batches_per_epoch})"
    )

    # Resolve epoch/step for old-style checkpoints (no training_progress.json)
    if resume_step == 0 and start_epoch == 0 and global_step > 0:
        start_epoch = global_step // total_batches_per_epoch
        resume_step = global_step % total_batches_per_epoch
        logger.info(
            f"Computed resume position: epoch={start_epoch}, "
            f"step_in_epoch={resume_step}"
        )

    t_start_training = time.time()
    batch_times = []

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        step_in_epoch = 0

        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}") as pbar:
            for batch in pbar:
                # Skip already-processed steps when resuming mid-epoch
                if epoch == start_epoch and step_in_epoch < resume_step:
                    step_in_epoch += 1
                    global_step += 1
                    continue

                t_batch_start = time.time()
                with accelerator.accumulate(active_policy):
                    metrics = ppo_controller.step(
                        batch, optimizer_policy, optimizer_value
                    )
                    torch.cuda.empty_cache()

                t_batch_end = time.time()
                batch_times.append(t_batch_end - t_batch_start)

                pbar.set_postfix(
                    {
                        "loss": f"{metrics['loss']:.4f}",
                        "reward": f"{metrics['reward']:.4f}",
                        "causal": f"{metrics['causal_penalty']:.4f}",
                        "disp": f"{metrics['dispersive_loss']:.4f}",
                        "gpu_mem_gb": (
                            f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}"
                        ),
                    }
                )

                global_step += 1
                step_in_epoch += 1

                # Quarter-epoch checkpoint
                if step_in_epoch in quarter_steps:
                    pct = int(100 * step_in_epoch / total_batches_per_epoch)
                    save_checkpoint_fn(
                        tag=f"ep{epoch + 1}-{pct}pct",
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                    )

        # End-of-epoch checkpoint
        save_checkpoint_fn(
            tag=f"ep{epoch + 1}-end",
            epoch=epoch + 1,
            step_in_epoch=0,
        )

    total_training_time = time.time() - t_start_training
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0

    return {
        "total_training_time": total_training_time,
        "avg_batch_time": avg_batch_time,
        "num_batches": len(batch_times),
        "total_samples": len(batch_times) * args.per_device_train_batch_size,
    }
