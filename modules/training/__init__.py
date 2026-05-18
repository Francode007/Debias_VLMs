"""
Training sub-package for PPO RL debiasing pipeline.

Modules:
    args        — CLI argument parsing (TrainingArgs)
    drm_loader  — PCA DRM head loading utilities
    setup       — Accelerator, model, dataloader wiring
    checkpoint  — Checkpoint save / load logic
    ppo_loop    — Per-epoch / per-batch PPO training loop
"""
from .args import parse_training_args
from .drm_loader import load_pca_components, load_debiased_model
from .setup import build_accelerator, build_policy_model, build_ppo_controller, build_dataloader
from .checkpoint import save_checkpoint, load_checkpoint, save_final_model
from .ppo_loop import run_ppo_loop

__all__ = [
    "parse_training_args",
    "load_pca_components",
    "load_debiased_model",
    "build_accelerator",
    "build_policy_model",
    "build_ppo_controller",
    "build_dataloader",
    "save_checkpoint",
    "load_checkpoint",
    "save_final_model",
    "run_ppo_loop",
]
