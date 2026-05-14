"""
RL Dataset Processing Module

This module handles dataset creation, processing, and formatting strictly for
generating prompts for the RL phase (PPO generation) of the VLM.
"""

import os
import glob
import ast
import io
import logging
import pandas as pd
from PIL import Image
from datasets import Dataset
from typing import Optional
from .config import ScriptArguments

logger = logging.getLogger(__name__)

class RLDatasetBuilder:
    """Handles dataset creation specifically for generating active responses during PPO."""
    
    def __init__(self, script_args: ScriptArguments):
        self.script_args = script_args
    
    def build_dataset(self, data_path: str, processor, split: str = 'train', size: Optional[int] = None):
        logger.info(f"Loading RL generation dataset from: {data_path}")
        
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")
        
        if self.script_args.use_smallset:
            parquet_files = parquet_files[:2]
        
        dfs = []
        for idx, f in enumerate(parquet_files):
            try:
                df = pd.read_parquet(f, engine="fastparquet")
                if len(df) > 1000 and self.script_args.use_smallset:
                    df = df.head(100)
                dfs.append(df)
                if self.script_args.use_smallset and len(dfs) >= 1:
                    break
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid parquet files could be loaded")
        
        full_df = pd.concat(dfs, ignore_index=True)
        del dfs
        
        if self.script_args.use_smallset:
            full_df = full_df.head(10)
        
        logger.info(f"Full dataset size: {len(full_df)} samples")
        ds = Dataset.from_pandas(full_df)
        del full_df
        
        if size is not None:
            ds = ds.select(range(0, min(size, len(ds))))
        
        # We don't expand into 2 pairs per example. We just need the unique prompt contexts.
        logger.info(f"Loaded {len(ds)} unique prompts for PPO generation.")
        
        # --- Chunked processing with disk persistence & resumability ---
        # Process in small chunks to avoid Arrow finalization hangs on large datasets.
        import hashlib
        import gc
        from datasets import concatenate_datasets, load_from_disk
        
        chunk_size = 200
        total = len(ds)
        num_chunks = (total + chunk_size - 1) // chunk_size
        
        cache_key = hashlib.md5(
            f"{data_path}_{total}_{self.script_args.max_length}_rl".encode()
        ).hexdigest()[:12]
        chunks_dir = os.path.join(os.path.dirname(data_path), f"rl_chunks_{cache_key}")
        os.makedirs(chunks_dir, exist_ok=True)
        
        logger.info(f"Processing {total} examples in {num_chunks} chunks of {chunk_size}")
        
        processed_chunks = []
        for chunk_idx in range(num_chunks):
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_idx:04d}")
            
            # Resumability: skip already-processed chunks
            if os.path.exists(chunk_path):
                logger.info(f"Chunk {chunk_idx+1}/{num_chunks} already exists, loading from disk")
                chunk_ds = load_from_disk(chunk_path)
                processed_chunks.append(chunk_ds)
                continue
            
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total)
            chunk_ds = ds.select(range(start, end))
            
            chunk_ds = chunk_ds.map(
                lambda examples: self._formatting_func_batched(examples, processor),
                batched=True,
                batch_size=32,
                num_proc=1,
                remove_columns=chunk_ds.column_names
            )
            
            chunk_ds.save_to_disk(chunk_path)
            processed_chunks.append(chunk_ds)
            logger.info(f"Chunk {chunk_idx+1}/{num_chunks} done: {len(chunk_ds)} samples")
            gc.collect()
        
        # Concatenate all chunks
        logger.info(f"Concatenating {len(processed_chunks)} chunks...")
        ds = concatenate_datasets(processed_chunks)
        del processed_chunks
        
        logger.info(f"Final RL dataset: {len(ds)} samples (length-filtered inline)")
        
        ds.set_format(type="torch")
        return ds

    def _formatting_func_batched(self, examples, processor):
        """
        Format a batch of examples for RL generation.
        """
        # Pre-declare all output columns for consistent Arrow schema
        _RESULT_KEYS = [
            "pixel_values", "input_ids", "attention_mask",
            "image_grid_thw", "video_grid_thw", "mm_token_type_ids",
            "data_index", "context", "question",
        ]
        results = {k: [] for k in _RESULT_KEYS}
        
        # Determine how many examples in this batch
        batch_size = len(examples[next(iter(examples.keys()))])
        
        all_images = []
        all_prompt_messages = []
        valid_indices = []
        
        for i in range(batch_size):
            # Extract single example from batch
            example = {k: examples[k][i] for k in examples.keys()}
            
            try:
                # Handle flattened file_name structure
                if 'file_name' in example:
                    image_data = example['file_name']
                elif 'file_name.bytes' in example:
                    image_data = {'bytes': example['file_name.bytes']}
                else:
                    image_data = example
                    
                # Load image
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    image_bytes = image_data['bytes']
                elif isinstance(image_data, bytes):
                    image_bytes = image_data
                else:
                    continue
                
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                all_images.append(image)
                
                prompt_text = example['context'] + " " + example['question']
                
                prompt_msg = [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
                ]
                
                all_prompt_messages.append(prompt_msg)
                valid_indices.append(i)
                
            except Exception as e:
                logger.warning(f"Error preparing example {i} in batch: {e}")
                continue
        
        if not valid_indices:
            return results
        
        # Apply chat templates in batch
        prompt_templates = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in all_prompt_messages]
        
        # Process inputs in batch - returning lists (not tensors) to handle variable lengths in ds.map
        kwargs = {
            "padding": False,
            "truncation": False,
            "return_tensors": None,
        }
        
        inputs = processor(text=prompt_templates, images=all_images, **kwargs)
        
        # Free PIL images immediately after processing
        for img in all_images:
            img.close()
        del all_images, all_prompt_messages
        
        # Calculate patch boundaries
        patch_slices = []
        if "pixel_values" in inputs and "image_grid_thw" in inputs:
            grid_thw = inputs["image_grid_thw"]
            import torch
            if isinstance(grid_thw, list):
                grid_thw = torch.tensor(grid_thw)
            start_idx = 0
            for i in range(len(grid_thw)):
                num_patches = int(grid_thw[i, 0] * grid_thw[i, 1] * grid_thw[i, 2])
                patch_slices.append((start_idx, start_idx + num_patches))
                start_idx += num_patches
        
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            # Inline length filter — skip examples exceeding max_length
            if len(inputs["input_ids"][idx_in_valid]) > self.script_args.max_length:
                continue
            
            # Add to results as lists/arrays (datasets will handle them)
            if "pixel_values" in inputs:
                if patch_slices:
                    start_p, end_p = patch_slices[idx_in_valid]
                    results["pixel_values"].append(inputs["pixel_values"][start_p:end_p])
                else:
                    results["pixel_values"].append(inputs["pixel_values"][idx_in_valid])
                    
            results["input_ids"].append(inputs["input_ids"][idx_in_valid])
            results["attention_mask"].append(inputs["attention_mask"][idx_in_valid])
            
            results["image_grid_thw"].append(
                inputs["image_grid_thw"][idx_in_valid] if "image_grid_thw" in inputs else None
            )
            results["video_grid_thw"].append(
                inputs["video_grid_thw"][idx_in_valid] if "video_grid_thw" in inputs else None
            )
            results["mm_token_type_ids"].append(
                inputs["mm_token_type_ids"][idx_in_valid] if "mm_token_type_ids" in inputs else None
            )
            
            # Metadata
            results["data_index"].append(orig_idx)
            results["context"].append(examples["context"][orig_idx])
            results["question"].append(examples["question"][orig_idx])
        
        # Remove columns that are entirely None (model doesn't produce them)
        results = {k: v for k, v in results.items() if not all(x is None for x in v)}
        # Ensure core keys exist for Arrow schema consistency
        for k in ["input_ids", "attention_mask", "pixel_values", "data_index", "context", "question"]:
            if k not in results:
                results[k] = []
        
        return results
    
    def _formatting_func(self, example, processor):
        try:
            if 'file_name' in example:
                image_data = example['file_name']
            elif 'file_name.bytes' in example:
                image_data = {'bytes': example['file_name.bytes']}
            else:
                image_data = example
                
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image_bytes = image_data['bytes']
            elif isinstance(image_data, bytes):
                image_bytes = image_data
                
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            prompt_text = example['context'] + " " + example['question']
            
            prompt_messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
            ]
            
            prompt_template = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            
            kwargs = {
                "padding": False, # Will pad dynamically in collator
                "truncation": False,
                "return_tensors": "pt",
            }
            
            inputs = processor(text=[prompt_template], images=[image], **kwargs)
            
            result = {
                "pixel_values": inputs["pixel_values"],
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
            }
            
            for k in ["image_grid_thw", "video_grid_thw", "mm_token_type_ids"]:
                if k in inputs:
                    result[k] = inputs[k]
            
            return result
        except Exception as e:
            logger.warning(f"Error processing example in RL config: {e}")
            raise
