"""
Main Execution Module

This module coordinates the execution of the reward model training and visualization
pipeline, orchestrating all the modular components.
"""

import os
import sys
import warnings

# Reduce noisy warnings (safe to ignore for this pipeline)
warnings.filterwarnings("ignore", message=".*OpenSSL.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Python 3.9 will be dropped.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*loaded as a fast processor by default.*", category=UserWarning)

import logging
import evaluate
import numpy as np
import torch
from transformers import AutoProcessor, HfArgumentParser, TrainingArguments

# Import modular components
from modules.utils import (
    ScriptArguments,
    DeviceManager,
    ModelLoader,
    DatasetBuilder,
    create_custom_forward,
    RewardDataCollatorWithPadding,
    RewardVisualizer,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment(script_args: ScriptArguments):
    """
    Set up the execution environment with proper configuration.
    
    Input:
        script_args (ScriptArguments): Configuration parameters
    
    Output:
        None (configures environment)
    
    Process:
        1. Set HuggingFace token if provided
        2. Configure logging and warning levels
        3. Set up any additional environment variables
    
    Purpose:
        Prepare the execution environment with proper authentication
        and configuration settings for model loading and training.
    """
    # Set up HuggingFace token
    if script_args.hf_token:
        os.environ["HF_TOKEN"] = script_args.hf_token
    
    # Configure warnings and logging
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)


def create_training_arguments(script_args: ScriptArguments, model_loader: ModelLoader):
    """
    Create training arguments for the reward trainer.
    
    Input:
        script_args (ScriptArguments): Configuration parameters
        model_loader (ModelLoader): Model loader with device information
    
    Output:
        TrainingArguments: Configured training arguments
    
    Process:
        1. Determine output directory based on model and configuration
        2. Set batch sizes and memory management parameters
        3. Configure precision based on device capabilities
        4. Set up logging and evaluation parameters
        5. Configure optimization and learning rate scheduling
        6. Return complete training arguments
    
    Purpose:
        Create comprehensive training configuration that optimizes
        performance for the specific hardware and model combination.
    """
    model_name_split = script_args.model.split("/")[-1]
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_{script_args.learning_rate}"
    
    # Use batch_size for training if not explicitly set
    if script_args.dataloader_batch_size is None:
        script_args.dataloader_batch_size = script_args.batch_size
    
    # eval_strategy is the current name; older transformers used evaluation_strategy
    training_kw = dict(
        output_dir=os.path.join(output_name, 'logs'),
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.dataloader_batch_size,
        per_device_eval_batch_size=script_args.dataloader_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        label_names=[],
        # MPS doesn't support fp16 mixed precision in TrainingArguments
        fp16=(model_loader.dtype == torch.float16 and "mps" not in str(model_loader.device)),
        bf16=model_loader.dtype == torch.bfloat16,
        logging_strategy="steps",
        logging_steps=10,
        warmup_ratio=0.05,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        run_name=script_args.wandb_name,
        max_grad_norm=5.0,
        report_to='none',
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=script_args.dataloader_num_workers,  # Configured via arguments (default 4)
        use_cpu=script_args.device == 'cpu',
    )
    
    training_args = TrainingArguments(**training_kw)
    
    # Add disable_dropout attribute if it doesn't exist
    if not hasattr(training_args, 'disable_dropout'):
        training_args.disable_dropout = False
        
    # Patch for TRL/Transformers compatibility
    if not hasattr(training_args, 'model_init_kwargs'):
        training_args.model_init_kwargs = None
    
    return training_args


def create_metrics():
    """
    Create evaluation metrics for the reward model.
    
    Input:
        None
    
    Output:
        function: Compute metrics function for evaluation
    
    Process:
        1. Load accuracy metric from evaluate library
        2. Define compute_metrics function for reward model evaluation
        3. Return configured metrics function
    
    Purpose:
        Provide evaluation metrics for monitoring training progress
        and model performance during reward model training.
    """
    accuracy = evaluate.load('accuracy')
    
    def compute_metrics(eval_pred):
        """
        Compute metrics for reward model evaluation.
        
        Input:
            eval_pred: Evaluation predictions from trainer
        
        Output:
            Dict[str, float]: Computed metrics
        
        Process:
            1. Extract predictions and convert to class predictions
            2. Create dummy labels (all zeros for reward model)
            3. Compute accuracy using the evaluate library
        
        Purpose:
            Evaluate reward model performance using standard metrics
            appropriate for ranking tasks.
        """
        predictions = eval_pred.predictions
        predictions = np.argmax(predictions, axis=1)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)
    
    return compute_metrics


def main():
    """
    Main execution function that orchestrates the entire pipeline.
    
    Input:
        None (uses command line arguments)
    
    Output:
        pd.DataFrame: Results from visualization process
    
    Process:
        1. Parse command line arguments into configuration
        2. Set up execution environment
        3. Initialize device manager and model loader
        4. Load model and processor with optimal settings
        5. Apply custom forward function for reward model
        6. Build and process dataset
        7. Create training arguments and metrics
        8. Initialize reward trainer with all components
        9. Run visualization to extract embeddings and predictions
        10. Return results dataframe
    
    Purpose:
        Coordinate the entire reward model training and visualization
        pipeline, integrating all modular components into a cohesive
        workflow that processes vision-language data for bias analysis.
    
    Returns:
        pd.DataFrame: Comprehensive results with embeddings, predictions, and metadata
    
    Raises:
        SystemExit: If critical errors occur during execution
    """
    try:
        # Parse arguments
        parser = HfArgumentParser(ScriptArguments)
        script_args = parser.parse_args_into_dataclasses()[0]
        
        # Set up environment
        setup_environment(script_args)
        
        # Initialize dataset builder (needed for both paths)
        dataset_builder = DatasetBuilder(script_args)
        
        # ---- Preprocess-only mode: no GPU / model needed ----
        if script_args.preprocess_only:
            logger.info("Running in preprocess-only mode (no model loading)...")
            # Load just the processor (tokenizer + image processor) — CPU only
            proc_kwargs = {"use_fast": True}
            if getattr(script_args, "max_pixels", 0):
                proc_kwargs["max_pixels"] = script_args.max_pixels
            if getattr(script_args, "min_pixels", 0):
                proc_kwargs["min_pixels"] = script_args.min_pixels
            processor = AutoProcessor.from_pretrained(script_args.model, **proc_kwargs)
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            
            logger.info("Building dataset (preprocess only)...")
            dataset = dataset_builder.build_dataset(script_args.data_path, processor)
            logger.info(f"Preprocessing complete. Dataset has {len(dataset)} samples.")
            return None
        
        # ---- Full inference mode: needs GPU ----
        # Initialize device manager
        device_manager = DeviceManager()
        
        # Initialize model loader
        model_loader = ModelLoader(script_args, device_manager)
        
        # Load model and processor
        logger.info("Loading model and processor...")
        model, processor = model_loader.load_model_and_processor()
        
        # Ensure model is strictly in eval mode for inference
        model.eval()
        
        # Reward models must output a single scalar; tell TRL num_labels=1
        model.config.num_labels = 1

        # Resolve A/B/C token IDs once for token_position={pre,post}_letter.
        # Mirrors PPOVLMController._get_letter_token_ids() — try multiple encodings
        # (bare, leading-space, leading-newline) and keep any single-token match
        # plus the last-token of any multi-token encoding.
        letter_token_ids = None
        if script_args.token_position in ("pre_letter", "post_letter"):
            tok = processor.tokenizer
            ids = set()
            for letter in ["A", "B", "C"]:
                for prefix in ["", " ", "\n"]:
                    enc = tok.encode(f"{prefix}{letter}", add_special_tokens=False)
                    if len(enc) == 1:
                        ids.add(enc[0])
                    elif len(enc) > 1:
                        ids.add(enc[-1])
                for t_id in tok.encode(letter, add_special_tokens=False):
                    ids.add(t_id)
            letter_token_ids = sorted(ids)
            logger.info(
                f"Resolved letter_token_ids for token_position={script_args.token_position!r}: "
                f"{letter_token_ids}"
            )
            if script_args.completion_format != "letter":
                logger.warning(
                    f"token_position={script_args.token_position!r} expects "
                    "completion_format='letter'; with 'free_text' the last-letter search "
                    "may match a letter buried inside the free-text answer."
                )

        # Create custom forward function (Phase 0.6 D3: per-sample slice index;
        # Phase 0.8 A2: configurable layer_idx for multi-layer head sweeps)
        custom_forward_func = create_custom_forward(
            model, model_loader.dtype,
            token_position=script_args.token_position,
            letter_token_ids=letter_token_ids,
            layer_idx=script_args.layer_idx,
        )
        model.forward = custom_forward_func.__get__(model, type(model))
        
        # Auto-suffix the embeddings output path when running with non-default
        # extraction settings, so the legacy free-text/eos artifacts at
        # ./embeddings_output/ are never silently clobbered. Phase 0.8 A2 adds
        # a _L{N} suffix when layer_idx != -2 (penultimate).
        if (script_args.completion_format != "free_text") or (script_args.token_position != "eos"):
            suffix = f"_{script_args.completion_format}_{script_args.token_position}"
            if not script_args.cls_embs_path.rstrip("/").endswith(suffix):
                script_args.cls_embs_path = script_args.cls_embs_path.rstrip("/") + suffix
                logger.info(f"Auto-suffixed cls_embs_path → {script_args.cls_embs_path}")
        if script_args.layer_idx != -2:
            layer_suffix = f"_L{script_args.layer_idx}"
            if not script_args.cls_embs_path.rstrip("/").endswith(layer_suffix):
                script_args.cls_embs_path = script_args.cls_embs_path.rstrip("/") + layer_suffix
                logger.info(f"Auto-suffixed cls_embs_path (layer_idx={script_args.layer_idx}) → {script_args.cls_embs_path}")
        
        # Build dataset
        logger.info("Building dataset...")
        dataset = dataset_builder.build_dataset(script_args.data_path, processor)
        eval_dataset = dataset
        
        # Set up training arguments
        training_args = create_training_arguments(script_args, model_loader)
        
        # Patch for TRL/Transformers compatibility
        # RewardTrainer expects these attributes which are normally in RewardConfig
        
        # 1. Attributes with explicit defaults in RewardConfig
        compatibility_attributes = {
            'model_init_kwargs': None,
            'chat_template_path': None,
            'max_length': script_args.max_length,  # Use script_args value
            'dataset_num_proc': None,
            'pad_to_multiple_of': None,
            'center_rewards_coefficient': None,
            'activation_offloading': False,
            'lr_scheduler_kwargs': None,
            # 'eos_token' and 'pad_token' are handled dynamically below
        }
        
        for attr, default_val in compatibility_attributes.items():
            if not hasattr(training_args, attr):
                setattr(training_args, attr, default_val)
                
        # 2. Dynamic attributes
        if not hasattr(training_args, 'eos_token'):
            try:
                training_args.eos_token = processor.tokenizer.eos_token or processor.tokenizer.eos_token_id
            except:
                training_args.eos_token = None
                
        if not hasattr(training_args, 'pad_token'):
            try:
                training_args.pad_token = processor.tokenizer.pad_token or processor.tokenizer.pad_token_id
            except:
                training_args.pad_token = None
        
        # Define metrics
        compute_metrics = create_metrics()
        
        # Initialize trainer
        try:
            trainer = RewardVisualizer(
                script_args=script_args,
                model=model,
                args=training_args,
                train_dataset=eval_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                data_collator=RewardDataCollatorWithPadding(
                    processor=processor, 
                    max_length=script_args.max_length,
                    batch_size=script_args.batch_size
                ),
            )
        except Exception as e:
            logger.error(f"Error initializing RewardVisualizer: {e}")
            # Try without some arguments that might be problematic
            trainer = RewardVisualizer(
                script_args=script_args,
                model=model,
                args=training_args,
                train_dataset=eval_dataset,
                eval_dataset=eval_dataset,
                data_collator=RewardDataCollatorWithPadding(
                    processor=processor, 
                    max_length=script_args.max_length,
                    batch_size=script_args.batch_size
                ),
            )
        
        # Run visualization
        logger.info("Starting visualization...")
        
        import time
        import json
        
        t0 = time.time()
        df = trainer.visualize_samples(int(1e8), script_args.cls_embs_path, script_args.data_path)
        inference_time = time.time() - t0
        
        try:
            with open("/tmp/inference_metrics.json", "w") as f:
                json.dump({"inference_time_seconds": inference_time}, f)
        except Exception as e:
            logger.warning(f"Could not save inference metrics: {e}")
            
        logger.info(f"Visualization completed! Inference time: {inference_time:.2f}s")
        return df
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import torch
    main()
