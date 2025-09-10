"""
Modules Package

This package contains modularized components for the reward model training
and visualization pipeline for vision-language models.

Components:
- config: Configuration and parameter definitions
- device_manager: Device detection and optimization
- model_loader: Model loading with fallback mechanisms
- dataset_builder: Dataset processing and formatting
- model_architecture: Custom forward functions for reward models
- data_collator: Batch processing and collation
- reward_trainer: Custom trainer with visualization capabilities
"""

from .config import ScriptArguments
from .device_manager import DeviceManager
from .model_loader import ModelLoader
from .dataset_builder import DatasetBuilder
from .model_architecture import create_custom_forward
from .data_collator import RewardDataCollatorWithPadding
from .reward_trainer import RewardVisualizer

__all__ = [
    'ScriptArguments',
    'DeviceManager',
    'ModelLoader',
    'DatasetBuilder',
    'create_custom_forward',
    'RewardDataCollatorWithPadding',
    'RewardVisualizer'
]
