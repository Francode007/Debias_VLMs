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
            
            if "mm_token_type_ids" in feature:
                if not hasattr(self, "_merged_mm_token_type_ids"):
                    self._merged_mm_token_type_ids = []
                self._merged_mm_token_type_ids.append(torch.as_tensor(feature["mm_token_type_ids"]))
            
            # Robust pixel_values collection
            pv = torch.as_tensor(feature["pixel_values"])
            # Qwen2.5-VL expects pixel_values as (num_patches, flat_dim) or (num_patches, C, T, P, P)
            # If it has a batch dimension of 1, squeeze it.
            if pv.ndim > 2 and pv.shape[0] == 1:
                pv = pv.squeeze(0)
            merged_pixel_values.append(pv)
            
        pixel_values = torch.cat(merged_pixel_values, dim=0)

        # Pad sequences to the longest in the batch
        max_length = max(len(ids) for ids in merged_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        
        padded_input_ids = []
        padded_attention_mask = []
        
        # We pad on the LEFT for generation
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
            
            if hasattr(self, "_merged_mm_token_type_ids"):
                if not hasattr(self, "_padded_mm_token_type_ids"):
                    self._padded_mm_token_type_ids = []
                pmm = torch.cat([
                    torch.zeros((padding_len,), dtype=self._merged_mm_token_type_ids[idx].dtype),
                    self._merged_mm_token_type_ids[idx]
                ])
                self._padded_mm_token_type_ids.append(pmm)
            
        import sys
        
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "pixel_values": pixel_values,
        }
        
        if hasattr(self, "_padded_mm_token_type_ids"):
            batch["mm_token_type_ids"] = torch.stack(self._padded_mm_token_type_ids)
            delattr(self, "_merged_mm_token_type_ids")
            delattr(self, "_padded_mm_token_type_ids")
        
        # Handle grid_thw concatenation
        for k in ["image_grid_thw", "video_grid_thw"]:
            merged_list = []
            has_key = False
            for i, feature in enumerate(features):
                if k in feature and feature[k] is not None:
                    has_key = True
                    t = torch.as_tensor(feature[k])
                    
                    # Ensure t is (num_images_in_sample, 3)
                    if t.ndim == 1:
                        t = t.unsqueeze(0)
                    elif t.ndim == 3 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    merged_list.append(t)
                    
            if has_key:
                batch[k] = torch.cat(merged_list, dim=0).long() # Explicitly cast to long
        
        # Sanity check: total patches must match grid_thw
        if "image_grid_thw" in batch:
            total_patches_from_grid = torch.sum(batch["image_grid_thw"][:, 0] * batch["image_grid_thw"][:, 1] * batch["image_grid_thw"][:, 2])
            if pixel_values.shape[0] != total_patches_from_grid:
                sys.stderr.write(f"CRITICAL WARNING: Patch mismatch! {pixel_values.shape[0]} != {total_patches_from_grid}\n")
                sys.stderr.flush()

        # Pass through gold_label if present (for binary reward mode)
        if "gold_label" in features[0]:
            batch["gold_label"] = torch.tensor(
                [f["gold_label"] for f in features], dtype=torch.long
            )
        
        return batch
