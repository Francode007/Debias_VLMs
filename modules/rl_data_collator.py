"""
RL Data Collator Module

Collation functions built specifically for passing pure generated prompts
into the PPO generator.
"""

import torch
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
from transformers import AutoProcessor
from transformers.utils import PaddingStrategy

logger = logging.getLogger(__name__)

@dataclass
class RLDataCollatorWithPadding:
    processor: AutoProcessor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Dynamically pads token ids keeping image parameters stacked precisely."""
        
        merged_input_ids = []
        merged_attention_mask = []
        merged_pixel_values = []
        
        for feature in features:
            merged_input_ids.append(torch.as_tensor(feature["input_ids"]))
            merged_attention_mask.append(torch.as_tensor(feature["attention_mask"]))
            merged_pixel_values.append(torch.as_tensor(feature["pixel_values"]))
            
        pixel_values = torch.cat(merged_pixel_values, dim=0)

        # Pad sequences to the longest in the batch
        # We must pad on the LEFT usually if we are doing generation, but standard HF generate handles right padding
        # Wait, for decoder generation, left padding is strongly recommended.
        # However Qwen2.5-VL uses right padding sometimes, but we will pad left manually for safe generation if required.
        # In typical PPO, we pad right but pass attention mask safely.
        
        # Let's pad efficiently
        max_length = max(len(ids) for ids in merged_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        
        padded_input_ids = []
        padded_attention_mask = []
        
        # We pad on the LEFT so generation concatenates easily on the right side without overriding parameters.
        for idx in range(len(merged_input_ids)):
            seq_len = len(merged_input_ids[idx])
            padding_len = max_length - seq_len
            
            pids = torch.cat([
                torch.full((padding_len,), pad_token_id, dtype=merged_input_ids[idx].dtype),
                merged_input_ids[idx]
            ])
            pmask = torch.cat([
                torch.zeros((padding_len,), dtype=merged_attention_mask[idx].dtype),
                merged_attention_mask[idx]
            ])
            padded_input_ids.append(pids)
            padded_attention_mask.append(pmask)
            
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "pixel_values": pixel_values,
        }
        
        for k in ["image_grid_thw", "video_grid_thw"]:
            merged_list = []
            has_key = False
            for feature in features:
                if k in feature:
                    has_key = True
                    merged_list.append(torch.as_tensor(feature[k]))
            if has_key:
                batch[k] = torch.cat(merged_list, dim=0)
        
        return batch
