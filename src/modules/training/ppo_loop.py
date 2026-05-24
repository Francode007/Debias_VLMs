"""
PPO training loop — per-epoch / per-batch execution.

Isolated from setup and checkpointing so it can be independently
tested, profiled, or replaced (e.g. with a GRPO variant).
"""
import json
import logging
import os
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
    lr_scheduler_policy=None,
    lr_scheduler_value=None,
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
    # Phase 0: optionally checkpoint every N steps (in addition to quarter-epoch
    # cadence). Used to densely capture the pre-collapse window in the first epoch.
    ckpt_every_steps = int(getattr(args, "ckpt_every_steps", 0) or 0)
    logger.info(
        f"Quarter-epoch checkpoints at steps: {sorted(quarter_steps)} "
        f"(total batches/epoch: {total_batches_per_epoch}); "
        f"ckpt_every_steps={ckpt_every_steps}"
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

    # ── Persistent per-batch metrics JSONL ──────────────────────────────────
    # Write every batch's full metrics dict to `<output_dir>/metrics.jsonl`
    # so we get full visibility into training without scraping stdout.
    # Only the main process should write to avoid duplicate lines.
    metrics_log_path = None
    metrics_log_fh = None
    if getattr(accelerator, "is_main_process", True):
        out_dir = getattr(args, "output_dir", None)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            metrics_log_path = os.path.join(out_dir, "metrics.jsonl")
            # Append so resumed runs preserve prior history
            metrics_log_fh = open(metrics_log_path, "a", buffering=1)  # line-buffered
            logger.info(f"Per-batch metrics → {metrics_log_path}")

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

                # Phase 3: step LR schedulers on accumulation boundaries only.
                if accelerator.sync_gradients:
                    if lr_scheduler_policy is not None:
                        lr_scheduler_policy.step()
                    if lr_scheduler_value is not None:
                        lr_scheduler_value.step()

                t_batch_end = time.time()
                batch_times.append(t_batch_end - t_batch_start)

                postfix_dict = {
                        "loss": f"{metrics['loss']:.4f}",
                        "r_bin": f"{metrics.get('reward_binary_mean', metrics.get('reward_dense_mean', 0)):.3f}",
                        "adv": f"{metrics.get('adv_abs_mean', 0):.3f}",
                        "pg_gn": f"{metrics.get('policy_grad_norm', 0):.4f}",
                        "v_gn": f"{metrics.get('value_grad_norm', 0):.4f}",
                        "kl": f"{metrics.get('kl', 0):.4f}",
                        "|Δlogp|": f"{metrics.get('mean_abs_logprob_diff', 0):.5f}",
                        "gpu_mem_gb": (
                            f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}"
                        ),
                }
                if "binary_accuracy" in metrics:
                    postfix_dict["acc"] = f"{metrics['binary_accuracy']:.3f}"
                if "parse_success_rate" in metrics:
                    postfix_dict["parse"] = f"{metrics['parse_success_rate']:.2f}"
                if "pred_letter_offset_mean" in metrics:
                    postfix_dict["off"] = f"{metrics['pred_letter_offset_mean']:.2f}"
                pbar.set_postfix(postfix_dict)

                # Persist full metrics to JSONL (one line per batch).
                if metrics_log_fh is not None:
                    record = {
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                        "batch_time_s": t_batch_end - t_batch_start,
                        "gpu_mem_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                    }
                    # Only serialize JSON-friendly scalars from metrics
                    for k, v in metrics.items():
                        if isinstance(v, (int, float, bool)) or v is None:
                            record[k] = v
                        else:
                            try:
                                record[k] = float(v)
                            except Exception:
                                pass
                    metrics_log_fh.write(json.dumps(record) + "\n")

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

                # Phase 0: optional dense per-N-step checkpoint cadence.
                if (
                    ckpt_every_steps > 0
                    and step_in_epoch > 0
                    and step_in_epoch % ckpt_every_steps == 0
                    and step_in_epoch not in quarter_steps  # avoid dup
                ):
                    save_checkpoint_fn(
                        tag=f"ep{epoch + 1}-step{step_in_epoch}",
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

    if metrics_log_fh is not None:
        metrics_log_fh.close()

    return {
        "total_training_time": total_training_time,
        "avg_batch_time": avg_batch_time,
        "num_batches": len(batch_times),
        "total_samples": len(batch_times) * args.per_device_train_batch_size,
    }
