"""
Training session setup: Accelerator, policy model, PPO controller, dataloader.

Each builder function is independently testable and returns a single
well-typed object (or small tuple), making unit testing and future
swapping trivial.
"""
import logging
import multiprocessing
from typing import Tuple

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from modules.utils import ScriptArguments, DeviceManager, ModelLoader
from modules.rl_components import PPOVLMController, FastRLNode, RLDataCollatorWithPadding, RLDatasetBuilder

logger = logging.getLogger(__name__)


def build_accelerator(args) -> Accelerator:
    """
    Create and return an HF Accelerator with bf16 mixed precision.

    Args:
        args: Parsed training arguments (expects gradient_accumulation_steps).

    Returns:
        Configured Accelerator instance.
    """
    return Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


def build_policy_model(args, accelerator: Accelerator) -> Tuple:
    """
    Load the base VLM, inject LoRA adapters, and enable gradient checkpointing.

    Args:
        args:         Parsed training arguments.
        accelerator:  Active Accelerator (for device info).

    Returns:
        (active_policy, processor) — LoRA-wrapped model and its AutoProcessor.
    """
    script_cfg = ScriptArguments(
        model=args.policy_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=args.max_length,
        use_smallset=args.use_smallset,
        dataset_name=args.dataset_name,
        model_family=args.model_family,
    )
    loader = ModelLoader(script_cfg, DeviceManager())

    # Processor (left-padded for Flash Attention compatibility)
    processor = AutoProcessor.from_pretrained(args.policy_model_name, use_fast=True)
    processor.tokenizer.padding_side = "left"

    policy_base, _ = loader.load_model_and_processor()

    lora_r = getattr(args, 'lora_r', 16)
    lora_alpha = getattr(args, 'lora_alpha', 32)
    logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    active_policy = get_peft_model(policy_base, lora_config)

    if hasattr(active_policy, "gradient_checkpointing_enable"):
        logger.info("Enabling gradient checkpointing on policy model")
        active_policy.gradient_checkpointing_enable()

    return active_policy, processor


def build_ppo_controller(
    active_policy,
    reward_heads_weight: torch.Tensor,
    accelerator: Accelerator,
    fast_rl_node: FastRLNode,
    args,
) -> PPOVLMController:
    """
    Construct the PPOVLMController with the given components.

    Args:
        active_policy:        LoRA-wrapped policy model.
        reward_heads_weight:  (K, hidden_dim) reward head tensor.
        accelerator:          Active Accelerator.
        fast_rl_node:         Instantiated FastRLNode.
        args:                 Parsed training arguments.

    Returns:
        Configured PPOVLMController.
    """
    return PPOVLMController(
        active_policy=active_policy,
        reward_heads_weight=reward_heads_weight,
        accelerator=accelerator,
        fast_rl_node=fast_rl_node,
        kl_beta=args.kl_beta,
        ppo_clip_range=getattr(args, 'ppo_clip_range', 0.2),
        lambda_causal=getattr(args, 'lambda_causal', 0.5),
        delta_margin=getattr(args, 'delta_margin', 1.0),
        lambda_dispersive=getattr(args, 'lambda_dispersive', 0.01),
        logit_reward_coef=getattr(args, 'logit_reward_coef', 0.1),
        max_gen_tokens=getattr(args, 'max_gen_tokens', 256),
    )


def build_dataloader(
    args,
    processor,
    accelerator: Accelerator,
    ppo_controller: PPOVLMController,
) -> Tuple:
    """
    Build the RL dataset, collator, DataLoader, and prepare them with Accelerate.

    Args:
        args:             Parsed training arguments.
        processor:        Loaded AutoProcessor.
        accelerator:      Active Accelerator.
        ppo_controller:   PPOVLMController (value_head needed for prepare()).

    Returns:
        (active_policy_prepared, value_head_prepared, optimizer_policy,
         optimizer_value, train_dataloader_prepared)
        — All wrapped by accelerator.prepare().

    Note:
        Callers must unpack accordingly and pass the returned active_policy
        back to ppo_controller.policy.
    """
    script_cfg = ScriptArguments(
        max_length=args.max_length,
        use_smallset=args.use_smallset,
        dataset_name=args.dataset_name,
        model_family=args.model_family,
        split=getattr(args, 'split', 'all'),
        split_indices_path=getattr(args, 'split_indices_path', None),
    )
    dataset_builder = RLDatasetBuilder(script_cfg)
    train_dataset = dataset_builder.build_dataset(
        data_path=args.data_path, processor=processor
    )

    # Limit training samples if requested (for experiments)
    max_samples = getattr(args, 'max_train_samples', None)
    if max_samples and max_samples < len(train_dataset):
        logger.info(f"Limiting training set: {len(train_dataset)} → {max_samples} samples")
        train_dataset = train_dataset.select(range(max_samples))

    collator = RLDataCollatorWithPadding(processor=processor)

    num_cpus = multiprocessing.cpu_count()
    num_workers = min(12, max(1, int(num_cpus * 0.8)))
    logger.info(
        f"DataLoader: num_workers={num_workers} "
        f"(capped at 12, machine reports {num_cpus} cores)"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataset, train_dataloader, collator
