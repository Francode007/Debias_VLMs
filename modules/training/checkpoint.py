"""
Checkpoint save / load utilities for PPO RL training.

Handles persisting and restoring:
  - LoRA adapter weights
  - Value head state dict
  - Fast-RL alpha state
  - Policy and value optimizer states
  - Training progress (epoch, step, global_step)
"""
import json
import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    *,
    accelerator,
    active_policy,
    ppo_controller,
    fast_rl_node,
    optimizer_policy,
    optimizer_value,
    output_dir: str,
    tag: str,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
) -> None:
    """
    Save a complete training checkpoint.

    Only executes on the main process to avoid duplicate writes.

    Args:
        accelerator:      HF Accelerate instance.
        active_policy:    The active LoRA-wrapped policy model.
        ppo_controller:   PPOVLMController (for value_head state).
        fast_rl_node:     FastRLNode (for alpha state).
        optimizer_policy: Policy AdamW optimizer.
        optimizer_value:  Value head AdamW optimizer.
        output_dir:       Root directory for all checkpoints.
        tag:              Checkpoint identifier (e.g. "ep1-50pct").
        epoch:            Current epoch index.
        step_in_epoch:    Current step index within the epoch.
        global_step:      Total training steps so far.
    """
    if not accelerator.is_main_process:
        return

    checkpoint_path = os.path.join(output_dir, f"checkpoint-{tag}")
    os.makedirs(checkpoint_path, exist_ok=True)
    logger.info(f"Saving checkpoint → {checkpoint_path}")

    # LoRA adapter
    unwrapped = accelerator.unwrap_model(active_policy)
    unwrapped.save_pretrained(checkpoint_path)

    # Fast-RL alpha vector
    with open(os.path.join(checkpoint_path, "fast_rl_state.json"), "w") as f:
        json.dump({"alpha": fast_rl_node.alpha.detach().cpu().tolist()}, f)

    # Value head
    torch.save(
        ppo_controller.value_head.state_dict(),
        os.path.join(checkpoint_path, "value_head.pth"),
    )

    # Optimizer states
    torch.save(
        optimizer_policy.state_dict(),
        os.path.join(checkpoint_path, "optimizer_policy.pth"),
    )
    torch.save(
        optimizer_value.state_dict(),
        os.path.join(checkpoint_path, "optimizer_value.pth"),
    )

    # Training progress
    with open(os.path.join(checkpoint_path, "training_progress.json"), "w") as f:
        json.dump(
            {
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": global_step,
            },
            f,
        )

    logger.info(
        f"Checkpoint saved: epoch={epoch}, step={step_in_epoch}, global_step={global_step}"
    )


def load_checkpoint(
    *,
    ckpt_dir: str,
    accelerator,
    active_policy,
    ppo_controller,
    fast_rl_node,
    optimizer_policy,
    optimizer_value,
) -> Tuple[int, int, int]:
    """
    Restore training state from a checkpoint directory.

    Args:
        ckpt_dir:         Path to a previously saved checkpoint directory.
        accelerator:      HF Accelerate instance.
        active_policy:    The active LoRA-wrapped policy model.
        ppo_controller:   PPOVLMController (for value_head restore).
        fast_rl_node:     FastRLNode (for alpha restore).
        optimizer_policy: Policy AdamW optimizer.
        optimizer_value:  Value head AdamW optimizer.

    Returns:
        (start_epoch, resume_step, global_step) — training position to resume from.
    """
    import re

    from peft import set_peft_model_state_dict

    logger.info(f"Resuming from checkpoint: {ckpt_dir}")
    unwrapped = accelerator.unwrap_model(active_policy)

    # --- LoRA adapter ---
    adapter_safetensors = os.path.join(ckpt_dir, "adapter_model.safetensors")
    adapter_bin = os.path.join(ckpt_dir, "adapter_model.bin")
    if os.path.exists(adapter_safetensors):
        from safetensors.torch import load_file
        adapter_state = load_file(adapter_safetensors)
    elif os.path.exists(adapter_bin):
        adapter_state = torch.load(adapter_bin, map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weights found in {ckpt_dir}")
    set_peft_model_state_dict(unwrapped, adapter_state)
    del adapter_state
    logger.info("Restored LoRA adapter weights")

    # --- Value head ---
    vh_path = os.path.join(ckpt_dir, "value_head.pth")
    if os.path.exists(vh_path):
        ppo_controller.value_head.load_state_dict(
            torch.load(vh_path, map_location="cpu")
        )
        logger.info("Restored value head")

    # --- Fast-RL alpha ---
    frl_path = os.path.join(ckpt_dir, "fast_rl_state.json")
    if os.path.exists(frl_path):
        with open(frl_path) as f:
            frl_state = json.load(f)
        fast_rl_node.alpha = torch.tensor(
            frl_state["alpha"], device=accelerator.device
        )
        logger.info("Restored Fast-RL alpha state")

    # --- Optimizer states ---
    opt_policy_path = os.path.join(ckpt_dir, "optimizer_policy.pth")
    opt_value_path = os.path.join(ckpt_dir, "optimizer_value.pth")
    if os.path.exists(opt_policy_path):
        optimizer_policy.load_state_dict(
            torch.load(opt_policy_path, map_location="cpu")
        )
        logger.info("Restored policy optimizer state")
    if os.path.exists(opt_value_path):
        optimizer_value.load_state_dict(
            torch.load(opt_value_path, map_location="cpu")
        )
        logger.info("Restored value optimizer state")

    # --- Training progress ---
    progress_path = os.path.join(ckpt_dir, "training_progress.json")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        start_epoch = progress.get("epoch", 0)
        resume_step = progress.get("step_in_epoch", 0)
        global_step = progress.get("global_step", 0)
        logger.info(
            f"Resuming from epoch={start_epoch}, step={resume_step}, global_step={global_step}"
        )
    else:
        # Fallback: infer from checkpoint directory name (e.g. "checkpoint-200")
        ckpt_match = re.search(r"checkpoint-(\d+)$", ckpt_dir.rstrip("/"))
        if ckpt_match:
            global_step = int(ckpt_match.group(1))
            start_epoch = 0
            resume_step = 0
            logger.info(
                f"No training_progress.json. Inferred global_step={global_step} from checkpoint name."
            )
        else:
            logger.warning(
                "No training_progress.json and could not infer step from name. Starting from 0."
            )
            start_epoch = 0
            resume_step = 0
            global_step = 0

    return start_epoch, resume_step, global_step


def save_final_model(
    *,
    accelerator,
    active_policy,
    ppo_controller,
    fast_rl_node,
    output_dir: str,
) -> None:
    """
    Persist the final debiased model artifacts after training completes.

    Args:
        accelerator:    HF Accelerate instance.
        active_policy:  The trained LoRA-wrapped policy.
        ppo_controller: PPOVLMController (for value_head).
        fast_rl_node:   FastRLNode (for final alpha vector).
        output_dir:     Root training output directory.
    """
    if not accelerator.is_main_process:
        return

    final_dir = os.path.join(output_dir, "final_debiased_model")
    os.makedirs(final_dir, exist_ok=True)

    unwrapped = accelerator.unwrap_model(active_policy)
    unwrapped.save_pretrained(final_dir)

    with open(os.path.join(final_dir, "fast_rl_state.json"), "w") as f:
        json.dump({"alpha": fast_rl_node.alpha.detach().cpu().tolist()}, f)

    torch.save(
        ppo_controller.value_head.state_dict(),
        os.path.join(final_dir, "value_head.pth"),
    )

    logger.info(f"Final debiased model saved to {final_dir}")
