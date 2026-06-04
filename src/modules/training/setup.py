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
from modules.rl_components import (
    PPOVLMController,
    Phase08PPOController,
    FastRLNode,
    RLDataCollatorWithPadding,
    RLDatasetBuilder,
)

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
    proc_kwargs = {"use_fast": True}
    max_px = getattr(args, "max_pixels", 0)
    min_px = getattr(args, "min_pixels", 0)
    if max_px and max_px > 0:
        proc_kwargs["max_pixels"] = max_px
    if min_px and min_px > 0:
        proc_kwargs["min_pixels"] = min_px
    processor = AutoProcessor.from_pretrained(args.policy_model_name, **proc_kwargs)
    processor.tokenizer.padding_side = "left"
    if max_px or min_px:
        logger.info(f"Processor pixel caps: max_pixels={max_px} min_pixels={min_px}")

    policy_base, _ = loader.load_model_and_processor()

    lora_r = getattr(args, 'lora_r', 16)
    lora_alpha = getattr(args, 'lora_alpha', 32)
    logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}")
    # Use regex to target ONLY text-model layers (model.layers.N.*),
    # excluding the vision tower (model.visual.blocks.*) which also has
    # gate_proj/up_proj/down_proj in its MLP blocks.
    lora_target_modules = r".*layers\.\d+\.\w+\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    # Phase 3 hygiene: refuse to LoRA-adapt the vision tower (would corrupt
    # the visual encoder we use to extract clean image features for the
    # reference policy and for SVM head training).
    _vision_substrings = ("vision", "visual", "vit", "image_encoder")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    active_policy = get_peft_model(policy_base, lora_config)

    # Post-hoc safety: enumerate every adapter-attached module and confirm
    # none of them live under a vision/visual submodule path.
    for name, _module in active_policy.named_modules():
        if "lora_" in name:
            assert not any(s in name.lower() for s in _vision_substrings), (
                f"LoRA adapter attached to vision-tower path: {name}"
            )

    if hasattr(active_policy, "gradient_checkpointing_enable"):
        logger.info("Enabling gradient checkpointing on policy model (use_reentrant=False)")
        # CRITICAL: With LoRA + frozen base + multimodal (Qwen2.5-VL fuses
        # frozen vision + frozen text embeddings before the LM decoder),
        # the *reentrant* checkpoint variant requires at least one input
        # tensor to have requires_grad=True and silently produces None
        # gradients otherwise. enable_input_require_grads() only hooks the
        # TEXT embedding, which is insufficient because the combined
        # text+vision tensor is what actually enters the checkpointed
        # decoder blocks. Use the non-reentrant variant which uses
        # saved_tensors_hooks and does not have this restriction — this is
        # the standard fix for PEFT + gradient_checkpointing.
        active_policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(active_policy, "enable_input_require_grads"):
            # Belt-and-suspenders: also register the embedding hook so any
            # downstream code path that *does* use reentrant checkpointing
            # (e.g. inside the vision tower) still sees grad-tracking inputs.
            logger.info("Enabling input require_grads for LoRA + checkpointing compatibility")
            active_policy.enable_input_require_grads()

    # Attach tokenizer to model for reward computation (binary mode needs it)
    active_policy.tokenizer = processor.tokenizer

    return active_policy, processor


def build_ppo_controller(
    active_policy,
    reward_heads_weight: torch.Tensor,
    accelerator: Accelerator,
    fast_rl_node: FastRLNode,
    args,
):
    """
    Construct the appropriate PPO controller based on ``args.reward_mode``.

    Returns ``Phase08PPOController`` when ``reward_mode == 'bias_aligned'``
    (Phase 0.8 probe-as-head), otherwise the legacy ``PPOVLMController``
    used by Phase 0.5/0.6 (``'svm'`` / ``'binary'``). Both controllers
    expose the same ``step(batch, optimizer_policy, optimizer_value)``
    surface so ``ppo_loop.run_ppo_loop`` is reward-mode agnostic.

    Args:
        active_policy:        LoRA-wrapped policy model.
        reward_heads_weight:  (K, hidden_dim) reward head tensor. For
                              ``bias_aligned`` this is the single probe
                              direction loaded from the
                              ``sb_bench-PROBE-component/`` head dir.
        accelerator:          Active Accelerator.
        fast_rl_node:         Instantiated FastRLNode (only meaningful for
                              ``svm`` mode; ignored when ``K==1``).
        args:                 Parsed training arguments.
    """
    reward_mode = getattr(args, "reward_mode", "svm")
    common_kwargs = dict(
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
        reward_mode=reward_mode,
        target_kl=getattr(args, 'target_kl', 0.0),
        kl_adapt_rate=getattr(args, 'kl_adapt_rate', 0.1),
        kl_beta_min=getattr(args, 'kl_beta_min', 0.05),
        kl_beta_max=getattr(args, 'kl_beta_max', 5.0),
        value_clip_range=getattr(args, 'value_clip_range', 0.0),
        max_grad_norm=getattr(args, 'max_grad_norm', 1.0),
        use_frozen_phi=getattr(args, 'use_frozen_phi', False),
    )

    if reward_mode == "bias_aligned":
        logger.info(
            "Building Phase08PPOController (reward_mode=bias_aligned, "
            f"layer={getattr(args, 'reward_head_layer', 13)}, "
            f"w_corr={getattr(args, 'correctness_coef', 1.0)}, "
            f"w_bias={getattr(args, 'bias_aligned_coef', 1.0)}, "
            f"w_ambig={getattr(args, 'ambig_preservation_coef', 0.5)})"
        )
        # Phase 0.8 requires frozen-φ projection (the probe was fit on
        # un-adapted base hidden states). The controller forces this on
        # internally, but flip the flag here too so downstream code and
        # metrics see a consistent value.
        common_kwargs["use_frozen_phi"] = True
        return Phase08PPOController(
            **common_kwargs,
            reward_head_layer=int(getattr(args, "reward_head_layer", 13)),
            bias_aligned_coef=float(getattr(args, "bias_aligned_coef", 1.0)),
            ambig_preservation_coef=float(
                getattr(args, "ambig_preservation_coef", 0.5)
            ),
            correctness_coef=float(getattr(args, "correctness_coef", 1.0)),
        )

    return PPOVLMController(**common_kwargs)


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

    # Optionally hold out the tail of the dataset as a deterministic eval
    # split for mid-training accuracy signal. This is the same data
    # distribution as training (intentional — we want a low-variance signal
    # on the model's current decision surface, not OOD generalisation).
    eval_dataloader = None
    midtrain_eval_n = int(getattr(args, "midtrain_eval_samples", 0) or 0)
    if midtrain_eval_n > 0 and getattr(args, "midtrain_eval_every_steps", 0) > 0:
        if midtrain_eval_n >= len(train_dataset):
            logger.warning(
                f"midtrain_eval_samples={midtrain_eval_n} >= dataset size "
                f"{len(train_dataset)}; disabling mid-training eval."
            )
        else:
            train_size = len(train_dataset) - midtrain_eval_n
            eval_dataset = train_dataset.select(
                range(train_size, len(train_dataset))
            )
            train_dataset = train_dataset.select(range(train_size))
            logger.info(
                f"Held-out mid-training eval split: "
                f"train={len(train_dataset)}, eval={len(eval_dataset)}"
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.per_device_train_batch_size,
                collate_fn=collator,
                shuffle=False,
                num_workers=min(4, num_workers),
                pin_memory=True,
            )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataset, train_dataloader, collator, eval_dataloader
