"""
Reward Trainer Module

This module provides the custom reward trainer for vision-language models
with specialized loss computation and embedding visualization capabilities.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm
import logging
from collections import defaultdict
from typing import Dict, Any, Tuple
from transformers.trainer_pt_utils import nested_detach
from trl import RewardTrainer
from accelerate import Accelerator
from accelerate.utils import gather_object
from .config import ScriptArguments

logger = logging.getLogger(__name__)


class RewardVisualizer(RewardTrainer):
    """
    Custom reward trainer for visualization and analysis of reward model training.
    
    This class extends the standard RewardTrainer to provide specialized
    functionality for loss computation, prediction, and embedding visualization
    specifically designed for vision-language reward models.
    """
    
    def __init__(self, script_args: ScriptArguments, *args, **kwargs):
        """
        Initialize RewardVisualizer with configuration.
        
        Input:
            script_args (ScriptArguments): Configuration parameters
            *args: Additional arguments passed to parent RewardTrainer
            **kwargs: Additional keyword arguments passed to parent RewardTrainer
        
        Output:
            None (initializes instance)
        
        Process:
            1. Call parent class initialization
            2. Store script arguments for loss computation and visualization
        
        Purpose:
            Set up custom reward trainer with configuration for specialized
            loss computation and visualization capabilities.
        """
        super().__init__(*args, **kwargs)
        self.script_args = script_args
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute reward model loss with proper device handling.
        
        Input:
            model: The reward model
            inputs (Dict[str, Any]): Batch of inputs containing:
                - input_ids: Token IDs for chosen and rejected sequences
                - attention_mask: Attention masks
                - pixel_values: Image pixel values
                - prompt_length: Length of prompt for reward gating
                - margin: Optional margin for margin loss (if used)
            return_outputs (bool): Whether to return additional outputs
        
        Output:
            If return_outputs=False:
                Tuple[torch.Tensor, torch.Tensor]: (loss, embeddings)
            If return_outputs=True:
                Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                    (loss, {"rewards_j": chosen_rewards, "rewards_k": rejected_rewards}, embeddings)
        
        Process:
            1. Forward pass through model with proper error handling
            2. Extract reward scores and embeddings from model output
            3. Split rewards into chosen (j) and rejected (k) pairs
            4. Compute loss based on configured loss type:
                - 'origin': Standard log-sigmoid ranking loss
                - 'margin': Margin-based ranking loss
                - 'labelsmooth': Label-smoothed ranking loss
            5. Return loss and optionally additional outputs
        
        Purpose:
            Compute the ranking loss for reward model training,
            where the model should assign higher rewards to chosen
            responses compared to rejected responses.
        
        Raises:
            NotImplementedError: If unsupported loss type is specified
        """
        try:
            # Prepare extra kwargs for Qwen2-VL
            extra_kwargs = {}
            for k in ["image_grid_thw", "video_grid_thw"]:
                if k in inputs:
                    extra_kwargs[k] = inputs[k]
                    
            res = model(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                pixel_values=inputs["pixel_values"], 
                prompt_length=inputs["prompt_length"],
                **extra_kwargs
            )
        except Exception as e:
            logger.error(f"Error in model forward pass: {e}")
            raise
        
        rewards = res['logits']
        emb = res['hidden_states']
        
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        
        # Compute loss based on type
        if self.script_args.loss_type == 'origin':
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        elif self.script_args.loss_type == 'margin':
            margin = inputs.get("margin", torch.zeros_like(rewards_j))
            loss = -nn.functional.logsigmoid(
                rewards_j - rewards_k - margin.view(-1, 1)
            ).mean()
        elif self.script_args.loss_type == 'labelsmooth':
            loss = (-0.9 * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - 
                   0.1 * nn.functional.logsigmoid(rewards_k - rewards_j).mean())
        else:
            raise NotImplementedError(f"Loss type {self.script_args.loss_type} not implemented")
        
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}, emb
        return loss, emb
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step for evaluation and visualization.
        
        Input:
            model: The reward model
            inputs (Dict[str, Any]): Batch of inputs for prediction
            prediction_loss_only (bool): Whether to return only loss
            ignore_keys (List[str], optional): Keys to ignore in outputs
        
        Output:
            Tuple containing:
                - loss (torch.Tensor): Computed loss
                - logits (torch.Tensor or None): Model predictions
                - labels (torch.Tensor or None): Target labels
                - embeddings (torch.Tensor): Extracted embeddings
        
        Process:
            1. Prepare inputs and set ignore keys if not provided
            2. Perform forward pass with gradient computation disabled
            3. Compute loss and extract outputs using compute_loss method
            4. If prediction_loss_only, return minimal output
            5. Process logits and create dummy labels for evaluation
            6. Stack logits for proper evaluation format
            7. Return structured prediction outputs
        
        Purpose:
            Enable evaluation and visualization by providing predictions,
            losses, and embeddings from the reward model without
            updating model parameters.
        """
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        with torch.no_grad():
            try:
                loss, logits_dict, emb = self.compute_loss(model, inputs, return_outputs=True)
            except Exception as e:
                logger.error(f"Error in prediction step: {e}")
                raise
        
        if prediction_loss_only:
            return (loss, None, None)
        
        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        
        # Stack accepted against rejected
        logits = torch.stack(logits).mean(dim=2).T
        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)
        
        return loss, logits, labels, emb
    
    def visualize_samples(self, num_print_samples: int, cls_embs_path: str, data_path: str):
        """
        Visualize samples and save embeddings with batch processing.
        
        Input:
            num_print_samples (int): Maximum number of samples to process
            cls_embs_path (str): Path to save embedding files
            data_path (str): Path to source data (for metadata)
        
        Output:
            pd.DataFrame: Results table with predictions, embeddings, and metadata
        
        Process:
            1. Get evaluation dataloader and initialize results table
            2. Create output directory for embeddings if needed
            3. Iterate through evaluation batches with progress tracking
            4. For each batch, extract individual samples and process
            5. Skip samples that already have saved embeddings
            6. Perform prediction step to get logits and embeddings
            7. Extract metadata and determine prediction flags
            8. Save embeddings to individual files for each sample
            9. Update results table with all extracted information
            10. Save intermediate results periodically for large datasets
            11. Return final results dataframe with all processed data
        
        Purpose:
            Process the entire evaluation dataset to extract embeddings,
            predictions, and metadata for analysis and visualization.
            Provides comprehensive view of model behavior across the dataset
            with persistent storage of embeddings for downstream analysis.
        """
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        
        if not os.path.exists(cls_embs_path):
            os.makedirs(cls_embs_path)
        
        logger.info(f"Eval dataset length: {len(self.eval_dataset)}")
        logger.info(f"Eval dataloader length: {len(eval_dataloader)}")
        logger.info(f"Processing with batch size: {self.script_args.batch_size}")
        
        processed_samples = 0
        for idx, inputs in tqdm.tqdm(enumerate(eval_dataloader), desc="Processing samples"):
            try:
                data_indices = inputs["data_index"]
                if isinstance(data_indices, torch.Tensor):
                    batch_indices = data_indices.tolist()
                    if not isinstance(batch_indices, list):
                        batch_indices = [batch_indices]
                else:
                    batch_indices = list(data_indices)
                n_pairs = len(batch_indices)

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                # Process the entire batch at once natively!
                # inputs contains `n_pairs` of pairs (chosen/rejected), perfectly collated.
                batch_inputs = dict(inputs)
                
                # Check if we literally already processed all of them
                # If so, we can skip the computation pass entirely.
                all_exist = all(os.path.exists(os.path.join(cls_embs_path, f"emb_{d_idx}.npy")) for d_idx in batch_indices)
                if all_exist:
                    logger.info(f"Skipping cached batch {idx}")
                    continue

                _, batched_logits, _, batched_emb = self.prediction_step(
                    self.model, batch_inputs, prediction_loss_only=False
                )
                
                # Move to CPU for numpy/tolist operations
                batched_logits = batched_logits.detach().cpu()
                batched_emb = batched_emb.detach().cpu()
                # Now we iterate through the RESULTS to save them and update the table.
                
                for batch_idx, data_index in enumerate(batch_indices):
                    fn = os.path.join(cls_embs_path, f"emb_{data_index}.npy")
                    
                    chosen_text = inputs["chosen"][batch_idx] if np.ndim(inputs["chosen"]) > 0 else inputs["chosen"]
                    rejected_text = inputs["rejected"][batch_idx] if np.ndim(inputs["rejected"]) > 0 else inputs["rejected"]
                    prompt = inputs["prompt"][batch_idx] if np.ndim(inputs["prompt"]) > 0 else inputs["prompt"]
                    
                    source = "sb_bench"
                    
                    # Update table
                    table["prompt"].append(prompt)
                    table["chosen_text"].append(chosen_text)
                    table["rejected_text"].append(rejected_text)
                    table["prompt_plus_chosen_response"].append(chosen_text)
                    table["prompt_plus_rejected_response"].append(rejected_text)
                    table["source"].append(source)
                    table["data_index"].append(data_index)
                    
                    processed_samples += 1
                    
                    pair_logits = batched_logits[batch_idx] # shape [2]
                    
                    # Determine flag
                    if pair_logits[0] > pair_logits[1]:
                        table["flag"].extend([1])
                    else:
                        table["flag"].extend([0])
                    
                    # Extract the pair of embeddings (chosen, rejected)
                    # chosen is at 2*batch_idx, rejected is at 2*batch_idx + 1
                    # We output it as shape [1, 2, hidden_dim] to match existing expectations
                    pair_emb = batched_emb[batch_idx * 2 : (batch_idx + 1) * 2, :]
                    cls_emb = pair_emb.float().numpy()
                    cls_emb = cls_emb[None, ...]
                    np.save(fn, cls_emb)
                    
                    # Update table with embeddings and logits
                    if Accelerator().num_processes == 1:
                        table["cls_emb"].extend(gather_object([fn]))
                        table["logits"].extend(gather_object([
                            [round(inner_item, 4) for inner_item in pair_logits.tolist()]
                        ]))
                    else:
                        table["cls_emb"].append(fn)
                        table["logits"].append(
                            [round(inner_item, 4) for inner_item in pair_logits.tolist()]
                        )
                    
                    # Save intermediate results
                    if len(table['chosen_text']) % 1000 == 0:
                        df = pd.DataFrame(table)
                        df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}_interim.csv")
                        logger.info(f"Saved interim results after {len(table['chosen_text'])} samples")
                    
                    if num_print_samples >= 0 and processed_samples >= num_print_samples:
                        break
                        
            except Exception as e:
                logger.exception(f"Error processing batch {idx}: {e}")
                raise
        
        # Save final results
        df = pd.DataFrame(table)
        df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}.csv")
        logger.info(f"Processing completed. Total samples processed: {processed_samples}")
        
        return df
