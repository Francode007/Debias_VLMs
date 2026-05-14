"""
Dataset Processing Module

This module handles dataset creation, processing, and formatting for the reward model
training pipeline, including memory-efficient loading and preprocessing.
"""

import os
import glob
import ast
import io
import hashlib
import logging
import pandas as pd
from PIL import Image
from datasets import Dataset, load_from_disk
from typing import List, Optional
from .config import ScriptArguments

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Handles dataset creation and processing for reward model training.
    
    This class manages the complex process of loading, processing, and formatting
    datasets from parquet files, with specific attention to memory efficiency
    and proper handling of vision-language data.
    """
    
    def __init__(self, script_args: ScriptArguments):
        """
        Initialize DatasetBuilder with configuration.
        
        Input:
            script_args (ScriptArguments): Configuration parameters
        
        Output:
            None (initializes instance)
        
        Process:
            1. Store configuration reference
            2. Define token patterns for different model families
        
        Purpose:
            Set up dataset processing with model-specific token patterns
            for proper sequence gating in reward model training.
        """
        self.script_args = script_args
        self.token_patterns = {
            "qwen": [151644, 46593, 198],  # <|im_start|>assistant\n
        }
    
    def find_token_for_gating(self, lst: List[int], model_family: str) -> int:
        """
        Find the last occurrence of a token pattern in a tokenized sequence.
        
        Input:
            lst (List[int]): List of token IDs
            model_family (str): Model family name ("qwen", etc.)
        
        Output:
            int: Index of the last occurrence of the token pattern
        
        Process:
            1. Get token pattern for the specified model family
            2. Search backwards through the token list
            3. Return index when pattern is found
            4. Raise ValueError if pattern not found
        
        Purpose:
            Identify the position where the assistant response begins
            in the tokenized sequence, which is crucial for reward model
            training to properly gate the loss computation.
        
        Raises:
            ValueError: If token pattern is not found in the sequence
        """
        token_pattern = self.token_patterns[model_family]
        token_pattern_len = len(token_pattern)
        search_end = len(lst)
        
        for j in range(search_end - token_pattern_len, -1, -1):
            if lst[j : j + token_pattern_len] == token_pattern:
                return j
        raise ValueError("Token pattern not found in the list.")
    
    def build_dataset(self, data_path: str, processor, split: str = 'train', size: Optional[int] = None):
        """
        Build dataset from parquet files with memory management.
        
        Input:
            data_path (str): Path to directory containing parquet files
            processor: HuggingFace processor for tokenization and image processing
            split (str): Dataset split name (currently unused, defaults to 'train')
            size (Optional[int]): Maximum dataset size limit
        
        Output:
            Dataset: Processed HuggingFace dataset ready for training
        
        Process:
            1. Load and validate parquet files from directory
            2. Apply memory management constraints (file and row limits)
            3. Concatenate dataframes with memory monitoring
            4. Convert to HuggingFace Dataset format
            5. Apply formatting function to process examples
            6. Filter by sequence length constraints
            7. Set tensor format for PyTorch compatibility
        
        Purpose:
            Create a memory-efficient dataset from raw parquet files
            with proper preprocessing for vision-language reward model training.
        
        Raises:
            FileNotFoundError: If no parquet files found in data_path
            ValueError: If no valid parquet files could be loaded
        """
        logger.info(f"Loading dataset from: {data_path}")
        
        # Load all Parquet files from directory
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # For memory efficiency, limit the number of files we process
        if self.script_args.use_smallset:
            logger.info("Using small dataset for testing - limiting to first 2 files")
            parquet_files = parquet_files[:2]
        elif len(parquet_files) > 10:
            logger.info(f"Large dataset detected ({len(parquet_files)} files). Limiting to first 10 files for memory efficiency.")
            parquet_files = parquet_files[:10]
        
        dfs = []
        max_memory_usage = 0
        
        for idx, f in enumerate(parquet_files):
            try:
                logger.info(f"Loading file {idx+1}/{len(parquet_files)}: {os.path.basename(f)}")
                df = pd.read_parquet(f, engine="fastparquet")
                
                # Limit rows per file for memory management
                if len(df) > 1000 and self.script_args.use_smallset:
                    df = df.head(100)  # Very small for testing
                    logger.info(f"Limited to {len(df)} rows for small set testing")
                elif len(df) > 5000:
                    df = df.head(1000)  # Limit to 1000 rows per file
                    logger.info(f"Limited to {len(df)} rows for memory efficiency")
                
                dfs.append(df)
                
                # Monitor memory usage
                current_memory = sum(df.memory_usage(deep=True).sum() for df in dfs)
                max_memory_usage = max(max_memory_usage, current_memory)
                
                # Break early if we have enough data for testing
                if self.script_args.use_smallset and len(dfs) >= 1:
                    logger.info("Small set mode: stopping after 1 file")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid parquet files could be loaded")
        
        logger.info(f"Concatenating {len(dfs)} dataframes...")
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Free memory from individual dataframes
        del dfs
        
        # Limit dataset size for memory efficiency
        if self.script_args.use_smallset:
            # Very small dataset for testing
            logger.info("Small set mode: limiting to 5 samples")
            full_df = full_df.head(5)
        elif len(full_df) > 10000:
            logger.info(f"Large dataset detected ({len(full_df)} samples). Limiting to 2000 samples for memory efficiency.")
            full_df = full_df.head(2000)
        
        ds = Dataset.from_pandas(full_df)
        
        # Free pandas dataframe memory
        del full_df
        
        if size is not None:
            ds = ds.select(range(0, min(size, len(ds))))
        
        # Add original index column
        new_column = list(range(len(ds)))
        ds = ds.add_column("orig_index", new_column)
        logger.info(f"Original dataset length: {len(ds)}")
        
        # Expand to 2 preference pairs per example (chosen vs rejected_1, chosen vs rejected_2)
        expanded_rows = []
        for i in range(len(ds)):
            row = ds[i]
            row_dict = {k: row[k] for k in row.keys()}
            for pair_idx in range(2):
                new_row = dict(row_dict)
                new_row["pair_idx"] = pair_idx
                new_row["data_index"] = i * 2 + pair_idx  # unique global pair index
                expanded_rows.append(new_row)
        ds = Dataset.from_list(expanded_rows)
        logger.info(f"Expanded to {len(ds)} preference pairs (2 per example)")
        
        # Check for cached preprocessed dataset
        cache_key = hashlib.md5(
            f"{data_path}_{len(ds)}_{self.script_args.max_length}_{self.script_args.use_smallset}".encode()
        ).hexdigest()[:12]
        cache_dir = os.path.join(os.path.dirname(data_path), f"preprocessed_cache_{cache_key}")
        
        if os.path.exists(cache_dir):
            logger.info(f"Loading cached preprocessed dataset from {cache_dir}")
            ds = load_from_disk(cache_dir)
            ds.set_format(type="torch")
            return ds
        
        # Apply formatting in single process to avoid IPC deadlocks.
        # With num_proc>1, large pixel_values arrays returned by workers
        # overflow pipe buffers causing deadlock after 100% completion.
        # Single-process is reliable and fast enough on high-RAM machines.
        map_batch_size = 16  # small batches to cap peak memory
        num_proc = 1
        logger.info(f"Preprocessing dataset with num_proc={num_proc}, batch_size={map_batch_size}")
        ds = ds.map(
            lambda examples: self._formatting_func_batched(examples, processor),
            batched=True,
            batch_size=map_batch_size,
            num_proc=num_proc,
            remove_columns=ds.column_names # Remove raw columns to avoid RewardTrainer auto-processing
        )
        
        # Length filtering is done inline inside _formatting_func_batched
        # to avoid a separate pass over the giant Arrow dataset (which deadlocks
        # due to deserializing huge pixel_values arrays).
        logger.info(f"Dataset after map (length-filtered inline): {len(ds)} samples")
        
        # Skip disk caching — the Arrow table with pixel_values is 200+GB
        # and serializing it causes the same deadlock/hang issues.
        # The volume commit in run_modal.py handles persistence instead.
        
        ds.set_format(type="torch")
        return ds
    
    def _formatting_func_batched(self, examples, processor):
        """
        Format a batch of examples for reward model training.
        """
        # Pre-declare all output columns so Arrow always gets a consistent schema,
        # even when every example in a batch is filtered out.
        _RESULT_KEYS = [
            "prompt_length", "prompt", "chosen", "rejected",
            "prompt_plus_chosen_response", "prompt_plus_rejected_response",
            "input_ids_chosen", "attention_mask_chosen", "pixel_values_chosen",
            "input_ids_rejected", "attention_mask_rejected", "pixel_values_rejected",
            "image_grid_thw_chosen", "image_grid_thw_rejected",
            "video_grid_thw_chosen", "video_grid_thw_rejected",
            "data_index", "pair_idx",
        ]
        results = {k: [] for k in _RESULT_KEYS}
        
        # Determine how many examples in this batch
        batch_size = len(examples[next(iter(examples.keys()))])
        
        all_images = []
        all_chosen_messages = []
        all_rejected_messages = []
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
                    image_data = {
                        'bytes': example['file_name.bytes'],
                        'path': example['file_name.path']
                    }
                else:
                    continue
                    
                # Load image
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    image_bytes = image_data['bytes']
                elif isinstance(image_data, bytes):
                    image_bytes = image_data
                else:
                    continue
                
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                all_images.append(image)
                
                # Extract chosen/rejected
                label = int(example['label'])
                chosen = example[f'ans{label}']
                wrong_indices = [idx for idx in range(3) if idx != label]
                pair_idx = int(example.get('pair_idx', 0))
                rejected = example[f'ans{wrong_indices[pair_idx]}']
                
                prompt_text = example['context'] + " " + example['question']
                
                # User/assistant format
                chosen_msg = [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]},
                    {"role": "assistant", "content": [{"type": "text", "text": chosen}]}
                ]
                rejected_msg = [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]},
                    {"role": "assistant", "content": [{"type": "text", "text": rejected}]}
                ]
                prompt_msg = [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
                ]
                
                all_chosen_messages.append(chosen_msg)
                all_rejected_messages.append(rejected_msg)
                all_prompt_messages.append(prompt_msg)
                valid_indices.append(i)
                
            except Exception as e:
                logger.warning(f"Error preparing example {i} in batch: {e}")
                continue
        
        if not valid_indices:
            return results
            
        # Apply chat templates in batch (tokenize=False returns list of strings)
        chosen_texts = [processor.apply_chat_template(msg, tokenize=False) for msg in all_chosen_messages]
        rejected_texts = [processor.apply_chat_template(msg, tokenize=False) for msg in all_rejected_messages]
        prompt_templates = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in all_prompt_messages]
        
        # Process inputs in batch - returning lists to handle variable lengths in ds.map
        kwargs = {
            "padding": False,
            "truncation": False,
            "return_tensors": None,
        }
        
        # Batch call to processor
        inputs_chosen = processor(text=chosen_texts, images=all_images, **kwargs)
        inputs_rejected = processor(text=rejected_texts, images=all_images, **kwargs)
        
        # Free PIL images immediately after processing to reduce memory
        for img in all_images:
            img.close()
        del all_images, all_chosen_messages, all_rejected_messages, all_prompt_messages
        
        # For Qwen2.5-VL, pixel_values is a flat concatenation of patches from all images.
        # We need to split it per-example using image_grid_thw to determine patch counts.
        import numpy as np
        
        def split_pixel_values(inputs):
            """Split concatenated pixel_values into per-example arrays using image_grid_thw."""
            if "image_grid_thw" not in inputs:
                # Not a Qwen2-VL style model, pixel_values is already per-example
                return inputs["pixel_values"], None
            grid_thw = inputs["image_grid_thw"]
            # Each image contributes t*h*w patches
            if hasattr(grid_thw, 'tolist'):
                grid_list = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
            else:
                grid_list = list(grid_thw)
            patches_per_image = [int(g[0]) * int(g[1]) * int(g[2]) for g in grid_list]
            pv = inputs["pixel_values"]
            if hasattr(pv, 'shape') and len(pv.shape) >= 2:
                # It's a concatenated array/tensor: split along dim 0
                splits = []
                offset = 0
                for n_patches in patches_per_image:
                    splits.append(pv[offset:offset + n_patches])
                    offset += n_patches
                return splits, grid_list
            else:
                # Already a list per example
                return pv, grid_list
        
        pv_chosen_splits, grid_chosen_list = split_pixel_values(inputs_chosen)
        pv_rejected_splits, grid_rejected_list = split_pixel_values(inputs_rejected)
        
        # Calculate prompt lengths
        all_tokens_prompt = processor.tokenizer(prompt_templates, padding=False)["input_ids"]
        
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            example = {k: examples[k][orig_idx] for k in examples.keys()}
            
            # Extract chosen/rejected again for metadata
            label = int(example['label'])
            chosen = example[f'ans{label}']
            wrong_indices = [idx for idx in range(3) if idx != label]
            pair_idx = int(example.get('pair_idx', 0))
            rejected = example[f'ans{wrong_indices[pair_idx]}']
            prompt_text = example['context'] + " " + example['question']
            
            tokens_prompt = all_tokens_prompt[idx_in_valid]
            try:
                prompt_len = self.find_token_for_gating(tokens_prompt, "qwen")
            except:
                prompt_len = len(tokens_prompt) - 1
            
            # Inline length filtering — skip examples that exceed max_length
            # This avoids a separate ds.filter() pass over the huge Arrow table.
            ids_chosen = inputs_chosen["input_ids"][idx_in_valid]
            ids_rejected = inputs_rejected["input_ids"][idx_in_valid]
            if (len(ids_chosen) > self.script_args.max_length or
                len(ids_rejected) > self.script_args.max_length or
                prompt_len >= self.script_args.max_length):
                continue
            
            # Add to results
            results["prompt_length"].append(prompt_len)
            results["prompt"].append(prompt_text)
            results["chosen"].append(chosen)
            results["rejected"].append(rejected)
            results["prompt_plus_chosen_response"].append(chosen_texts[idx_in_valid])
            results["prompt_plus_rejected_response"].append(rejected_texts[idx_in_valid])
            
            results["input_ids_chosen"].append(inputs_chosen["input_ids"][idx_in_valid])
            results["attention_mask_chosen"].append(inputs_chosen["attention_mask"][idx_in_valid])
            results["pixel_values_chosen"].append(pv_chosen_splits[idx_in_valid])
            
            results["input_ids_rejected"].append(inputs_rejected["input_ids"][idx_in_valid])
            results["attention_mask_rejected"].append(inputs_rejected["attention_mask"][idx_in_valid])
            results["pixel_values_rejected"].append(pv_rejected_splits[idx_in_valid])
            
            # Qwen2-VL grid tokens - preserve shape
            for k in ["image_grid_thw", "video_grid_thw"]:
                results[f"{k}_chosen"].append(
                    inputs_chosen[k][idx_in_valid] if k in inputs_chosen else None
                )
                results[f"{k}_rejected"].append(
                    inputs_rejected[k][idx_in_valid] if k in inputs_rejected else None
                )
            
            # Metadata
            results["data_index"].append(example['data_index'])
            results["pair_idx"].append(pair_idx)
            
        # Remove columns that are entirely None (model doesn't produce them)
        results = {k: v for k, v in results.items() if not all(x is None for x in v)}
        # Ensure at least the core keys exist (even if empty) for Arrow schema
        for k in _RESULT_KEYS[:12]:  # core columns that are always produced
            if k not in results:
                results[k] = []
        
        return results



