"""
Modules Package

This package contains modularized components for the C-DeFR-L
debiasing pipeline for vision-language models.

Subpackages:
- utils: Shared utilities (config, device, model loading, data processing)
- rl_components: Core RL (Fast-RL, CAA, PPO controller, RL data handling)
- data: Dataset downloading (SB-Bench, POPE)
- embeddings: Phase 1 embedding extraction and DRM head generation
- inference: Answer generation for benchmarks
- evaluation: Evaluation scripts and evaluator classes
- training: Phase 2+3 PPO RL training pipeline
"""

from .utils import (
    ScriptArguments,
    DeviceManager,
    ModelLoader,
    DatasetBuilder,
    create_custom_forward,
    RewardDataCollatorWithPadding,
    RewardVisualizer,
)
from .rl_components import FastRLNode, compute_causal_reward_penalty, compute_dispersive_loss, PPOVLMController, MultipleHead

__all__ = [
    'ScriptArguments',
    'DeviceManager',
    'ModelLoader',
    'DatasetBuilder',
    'create_custom_forward',
    'RewardDataCollatorWithPadding',
    'RewardVisualizer',
    'MultipleHead',
    'FastRLNode',
    'compute_causal_reward_penalty',
    'compute_dispersive_loss',
    'PPOVLMController',
]
