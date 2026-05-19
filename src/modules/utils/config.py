"""
Configuration Module

This module defines all configuration parameters and arguments for the reward model
training and visualization pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScriptArguments:
    """
    Configuration arguments for the reward model training and visualization script.
    
    This class centralizes all configuration parameters, including training hyperparameters,
    model settings, device configurations, and path specifications.
    
    The configuration is organized into logical groups:
    - Training parameters: batch sizes, learning rates, epochs
    - Memory management: batch processing and memory optimization settings
    - Model parameters: model selection, quantization, LoRA settings
    - Device parameters: device selection and precision settings
    - Paths and logging: data paths, output directories, logging configuration
    - Training configuration: loss types, evaluation settings, debugging options
    """
    
    # Training parameters
    per_device_train_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Training batch size per device. Keep small for memory efficiency."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Evaluation batch size per device. Keep small for memory efficiency."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16,
        metadata={"help": "Number of steps to accumulate gradients before updating. Increases effective batch size."}
    )
    learning_rate: Optional[float] = field(
        default=5e-6,
        metadata={"help": "Learning rate for training. Conservative value for vision-language models."}
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Number of training epochs. Usually 1 is sufficient for reward model training."}
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "Optimizer type. AdamW is standard for transformer models."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler. Cosine provides smooth decay."}
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum sequence length for tokenized inputs. Affects memory usage significantly."}
    )
    
    # Memory management
    batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "Batch size for data processing. Controls memory usage during dataset processing."}
    )
    dataloader_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "DataLoader batch size. Defaults to batch_size if not set. Used for training/evaluation."}
    )
    dataloader_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of worker processes for data loading. Increase to avoid GPU starvation."}
    )
    
    # Model parameters
    model: Optional[str] = field(
        default='Qwen/Qwen2.5-VL-3B-Instruct',
        metadata={"help": "Primary model name or path. Can be HuggingFace model ID or local path. Default: Qwen2.5-VL 3B."}
    )
    model_family: Optional[str] = field(
        default='qwen',
        metadata={"help": "Model family for loading the correct model wrapper. Default: 'qwen'"}
    )
    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LoRA (Low-Rank Adaptation) for parameter-efficient training."}
    )
    base_model: Optional[str] = field(
        default='Qwen/Qwen2.5-VL-3B-Instruct',
        metadata={"help": "Base model for LoRA training. Usually same as model parameter."}
    )
    freeze_pretrained: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze pretrained model parameters during training."}
    )
    fallback_model: Optional[str] = field(
        default='Qwen/Qwen2-VL-2B-Instruct',
        metadata={"help": "Fallback model if primary model fails to load. Qwen2.5-VL has 3B/7B; Qwen2-VL has 2B/7B."}
    )
    load_in_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Load model in 8-bit precision for memory efficiency. May affect quality."}
    )
    load_in_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Load model in 4-bit precision for maximum memory efficiency. May affect quality."}
    )
    
    # Device parameters
    device: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to use: 'auto', 'cuda', 'mps', 'cpu'. Auto selects best available."}
    )
    force_fp32: Optional[bool] = field(
        default=False,
        metadata={"help": "Force float32 precision. Used automatically on MPS/CPU for stability (avoids NaN); set manually on CUDA if needed."}
    )
    
    # Paths and logging
    dataset_name: Optional[str] = field(
        default='sb_bench',
        metadata={"help": "Name of the dataset for formatting (e.g., sb_bench, pope)."}
    )
    data_path: Optional[str] = field(
        default='./sb_bench_data/data',
        metadata={"help": "Path to training data directory containing parquet files."}
    )
    log_dir: Optional[str] = field(
        default='./reward_models_sb_bench',
        metadata={"help": "Directory for saving training logs and model checkpoints."}
    )
    cls_embs_path: Optional[str] = field(
        default='./embeddings_output',
        metadata={"help": "Path for saving extracted embeddings during visualization."}
    )
    output_dir: Optional[str] = field(
        default='./outputs',
        metadata={"help": "General output directory for training artifacts."}
    )
    wandb_name: Optional[str] = field(
        default="qwen_vl_reward_sb_bench",
        metadata={"help": "Experiment name for Weights & Biases logging (currently disabled)."}
    )
    
    # Training configuration
    loss_type: Optional[str] = field(
        default='origin',
        metadata={"help": "Loss function type: 'origin', 'margin', 'labelsmooth'. Controls reward model training objective."}
    )
    use_smallset: Optional[bool] = field(
        default=False,
        metadata={"help": "Use small dataset subset for testing. Limits to ~50 samples for quick debugging."}
    )
    save_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Save model checkpoint every N steps."}
    )
    eval_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Run evaluation every N steps."}
    )
    logging_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Log training metrics every N steps."}
    )
    warmup_steps: Optional[int] = field(
        default=50,
        metadata={"help": "Number of warmup steps for learning rate scheduler."}
    )
    debug: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable debug mode with additional logging and error handling."}
    )
    
    # HuggingFace token
    hf_token: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace API token for accessing private models or increasing rate limits."}
    )
    
    # Pipeline splitting
    preprocess_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Only run dataset preprocessing (no model loading or inference). For cost-efficient CPU-only runs."}
    )
    split: Optional[str] = field(
        default="all",
        metadata={"help": "Which split to use: 'train' (80%%), 'test' (20%%), or 'all' (no split). "
                  "For in-domain evaluation, Phase 1 and training use 'train'; evaluation uses 'test'."}
    )
    split_indices_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to split_indices.json. If None, auto-detected next to data_path. "
                  "Generated on first use with seed=42."}
    )
