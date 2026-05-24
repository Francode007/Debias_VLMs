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
    eval_dataloader=None,
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

    # ── Early stopping state ─────────────────────────────────────────────
    early_stop_patience = int(getattr(args, "early_stop_patience", 0) or 0)
    early_stop_threshold = float(getattr(args, "early_stop_threshold", 0.05))
    early_stop_window = int(getattr(args, "early_stop_window", 10))
    es_best_acc = 0.0
    es_patience_counter = 0
    es_acc_history: list = []  # rolling window of binary_accuracy values
    early_stopped = False
    if early_stop_patience > 0:
        logger.info(
            f"Early stopping enabled: patience={early_stop_patience} steps, "
            f"threshold={early_stop_threshold}, window={early_stop_window}"
        )

    # ── Mid-training eval state ──────────────────────────────────────────
    midtrain_eval_every = int(getattr(args, "midtrain_eval_every_steps", 0) or 0)
    use_eval_for_early_stop = bool(getattr(args, "use_eval_for_early_stop", False))
    if midtrain_eval_every > 0 and eval_dataloader is None:
        logger.warning(
            "midtrain_eval_every_steps > 0 but no eval_dataloader provided; "
            "mid-training eval will be skipped."
        )
        midtrain_eval_every = 0
    if midtrain_eval_every > 0:
        logger.info(
            f"Mid-training eval: every {midtrain_eval_every} steps, "
            f"use_eval_for_early_stop={use_eval_for_early_stop}"
        )
    last_midtrain_eval_acc = None

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

                # ── Mid-training held-out eval ─────────────────────────────
                # Runs constrained greedy decoding on the held-out subset.
                # Logged as midtrain_eval_acc; optionally used as the
                # early-stop signal in place of noisy training-batch acc.
                if (
                    midtrain_eval_every > 0
                    and step_in_epoch > 0
                    and step_in_epoch % midtrain_eval_every == 0
                ):
                    t_eval0 = time.time()
                    eval_metrics = ppo_controller.evaluate_subset(eval_dataloader)
                    eval_time = time.time() - t_eval0
                    if eval_metrics:
                        last_midtrain_eval_acc = eval_metrics.get(
                            "midtrain_eval_acc"
                        )
                        logger.info(
                            f"[midtrain-eval] step={step_in_epoch} "
                            f"acc={last_midtrain_eval_acc:.4f} "
                            f"n={eval_metrics.get('midtrain_eval_n')} "
                            f"parse={eval_metrics.get('midtrain_eval_parse_rate'):.3f} "
                            f"predA={eval_metrics.get('midtrain_eval_pred_A'):.2f} "
                            f"predB={eval_metrics.get('midtrain_eval_pred_B'):.2f} "
                            f"predC={eval_metrics.get('midtrain_eval_pred_C'):.2f} "
                            f"({eval_time:.1f}s)"
                        )
                        if metrics_log_fh is not None:
                            rec = {
                                "event": "midtrain_eval",
                                "epoch": epoch,
                                "step_in_epoch": step_in_epoch,
                                "global_step": global_step,
                                "midtrain_eval_time_s": eval_time,
                            }
                            rec.update(eval_metrics)
                            metrics_log_fh.write(json.dumps(rec) + "\n")

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

                # ── Early stopping check ────────────────────────────────────
                # Default signal is rolling training-batch binary_accuracy;
                # when --use_eval_for_early_stop is set, we instead drive
                # the rolling window from the held-out eval acc (much lower
                # variance, but only updated every midtrain_eval_every steps).
                if early_stop_patience > 0:
                    if use_eval_for_early_stop:
                        if last_midtrain_eval_acc is None:
                            cur_acc = None
                        else:
                            cur_acc = last_midtrain_eval_acc
                            # Only push to rolling history when a NEW eval
                            # value arrived, to avoid pinning the window.
                            if (
                                step_in_epoch > 0
                                and step_in_epoch % midtrain_eval_every == 0
                            ):
                                es_acc_history.append(cur_acc)
                    else:
                        cur_acc = metrics.get("binary_accuracy")
                        if cur_acc is not None:
                            es_acc_history.append(cur_acc)

                if early_stop_patience > 0 and es_acc_history:
                    if len(es_acc_history) > early_stop_window:
                        es_acc_history.pop(0)
                    rolling_acc = sum(es_acc_history) / len(es_acc_history)

                    if rolling_acc > es_best_acc:
                        es_best_acc = rolling_acc
                        es_patience_counter = 0
                        # Save best checkpoint
                        save_checkpoint_fn(
                            tag=f"ep{epoch + 1}-best",
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                        )
                    elif rolling_acc < es_best_acc - early_stop_threshold:
                        es_patience_counter += 1
                        if es_patience_counter >= early_stop_patience:
                            logger.warning(
                                f"Early stopping triggered at epoch {epoch+1}, "
                                f"step {step_in_epoch}: rolling_acc={rolling_acc:.4f} "
                                f"< best={es_best_acc:.4f} - {early_stop_threshold} "
                                f"for {early_stop_patience} consecutive steps."
                            )
                            # Write a marker to metrics log
                            if metrics_log_fh is not None:
                                metrics_log_fh.write(json.dumps({
                                    "event": "early_stop",
                                    "epoch": epoch,
                                    "step_in_epoch": step_in_epoch,
                                    "global_step": global_step,
                                    "rolling_acc": rolling_acc,
                                    "best_acc": es_best_acc,
                                }) + "\n")
                            early_stopped = True
                            break
                    else:
                        es_patience_counter = 0

            if early_stopped:
                break

        # End-of-epoch checkpoint (skip if early-stopped since best is already saved)
        if not early_stopped:
            save_checkpoint_fn(
                tag=f"ep{epoch + 1}-end",
                epoch=epoch + 1,
                step_in_epoch=0,
            )

        if early_stopped:
            break

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
