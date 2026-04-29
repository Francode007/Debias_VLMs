"""
Data Collator Module

This module provides data collation functionality for batch processing
in reward model training with memory management and batch size control.
"""

import torch
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
from transformers import AutoProcessor
from transformers.utils import PaddingStrategy

logger = logging.getLogger(__name__)


@dataclass
class RewardDataCollatorWithPadding:
    """
    Data collator for reward model training with batch size control.
    
    This class handles the collation of individual examples into batches
    for reward model training, with specific attention to memory management
    and proper handling of chosen/rejected pairs.
    
    Attributes:
        processor: HuggingFace processor for tokenization
        padding: Padding strategy for sequences
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad sequences to multiple of this value
        return_tensors: Format for returned tensors
        batch_size: Maximum batch size to prevent memory issues
    """
    
    processor: AutoProcessor
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    batch_size: Optional[int] = field(default=1)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of examples into a batch for reward model training.
        
        Input:
            features (List[Dict[str, Any]]): List of processed examples from dataset
                Each example contains:
                - input_ids_chosen: Tokenized chosen response
                - attention_mask_chosen: Attention mask for chosen response
                - pixel_values_chosen: Processed image for chosen response
                - input_ids_rejected: Tokenized rejected response
                - attention_mask_rejected: Attention mask for rejected response
                - pixel_values_rejected: Processed image for rejected response
                - prompt: Original prompt text
                - chosen: Chosen response text
                - rejected: Rejected response text
                - prompt_plus_chosen_response: Full conversation with chosen response
                - prompt_plus_rejected_response: Full conversation with rejected response
                - prompt_length: Length of prompt for reward gating
                - data_index: Index of example in original dataset
        
        Output:
            Dict[str, Any]: Batched data ready for model input
                Contains:
                - input_ids: Stacked input IDs (chosen + rejected interleaved)
                - attention_mask: Stacked attention masks
                - pixel_values: Stacked image tensors
                - prompt: Prompt text (from first example)
                - chosen: Chosen response text (from first example)
                - rejected: Rejected response text (from first example)
                - prompt_plus_chosen_response: Full chosen conversation
                - prompt_plus_rejected_response: Full rejected conversation
                - prompt_length: Prompt length for reward computation
                - data_index: Example index for tracking
        
        Process:
            1. Limit batch size to prevent memory overflow
            2. Extract chosen and rejected data from each example
            3. Interleave chosen and rejected sequences in the batch
            4. Stack tensors for efficient GPU processing
            5. Preserve metadata from the first example
            6. Return properly formatted batch
        
        Purpose:
            Transform individual processed examples into GPU-ready batches
            with proper memory management and chosen/rejected pair handling
            required for reward model training.
        """
        # Limit batch size to prevent memory issues
        if len(features) > self.batch_size:
            logger.warning(f"Batch size {len(features)} exceeds configured batch_size {self.batch_size}, truncating")
            features = features[:self.batch_size]
        
        merged_features_input_ids = []
        merged_features_attention_mask = []
        merged_features_pixel_values = []
        
        for feature in features:
            # Chosen
            merged_features_input_ids.append(torch.as_tensor(feature["input_ids_chosen"]))
            merged_features_attention_mask.append(torch.as_tensor(feature["attention_mask_chosen"]))
            merged_features_pixel_values.append(torch.as_tensor(feature["pixel_values_chosen"]))
            # Rejected
            merged_features_input_ids.append(torch.as_tensor(feature["input_ids_rejected"]))
            merged_features_attention_mask.append(torch.as_tensor(feature["attention_mask_rejected"]))
            merged_features_pixel_values.append(torch.as_tensor(feature["pixel_values_rejected"]))
            
        # Pad sequences
        max_length = max(len(ids) for ids in merged_features_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for idx in range(len(merged_features_input_ids)):
            seq_len = len(merged_features_input_ids[idx])
            padding_len = max_length - seq_len
            
            pids = torch.cat([
                torch.full((padding_len,), pad_token_id, dtype=merged_features_input_ids[idx].dtype),
                merged_features_input_ids[idx]
            ])
            pmask = torch.cat([
                torch.zeros((padding_len,), dtype=merged_features_attention_mask[idx].dtype),
                merged_features_attention_mask[idx]
            ])
            padded_input_ids.append(pids)
            padded_attention_mask.append(pmask)
            
        # Qwen2-VL pixel_values: concat on dim=0 (batch of images)
        pixel_values = torch.cat(merged_features_pixel_values, dim=0)

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "pixel_values": pixel_values,
            "prompt": [f["prompt"] for f in features],
            "chosen": [f["chosen"] for f in features],
            "rejected": [f["rejected"] for f in features],
            "prompt_plus_chosen_response": [f["prompt_plus_chosen_response"] for f in features],
            "prompt_plus_rejected_response": [f["prompt_plus_rejected_response"] for f in features],
            "prompt_length": [f["prompt_length"] for f in features],
            "data_index": [f["data_index"] for f in features],
        }
        
        for k in ["image_grid_thw", "video_grid_thw"]:
            merged_list = []
            has_key = False
            for feature in features:
                if f"{k}_chosen" in feature and f"{k}_rejected" in feature:
                    has_key = True
                    merged_list.append(torch.as_tensor(feature[f"{k}_chosen"]))
                    merged_list.append(torch.as_tensor(feature[f"{k}_rejected"]))
            if has_key:
                batch[k] = torch.cat(merged_list, dim=0)
        
        return batch
