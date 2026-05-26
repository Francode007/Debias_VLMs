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
from .split import get_or_create_split
from ..data.registry import get_dataset
from ..data.model_registry import get_model_wrapper

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
        self.dataset_adapter = get_dataset(getattr(script_args, 'dataset_name', 'sb_bench'))
        self.model_wrapper = get_model_wrapper(getattr(script_args, 'model_family', 'qwen'))
    
    def find_token_for_gating(self, lst: List[int], model_family: str) -> int:
        return self.model_wrapper.find_token_for_gating(lst)
    
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
        
        # For testing, limit the number of files we process
        if self.script_args.use_smallset:
            logger.info("Using small dataset for testing - limiting to first 2 files")
            parquet_files = parquet_files[:2]
        
        dfs = []
        max_memory_usage = 0
        
        for idx, f in enumerate(parquet_files):
            try:
                logger.info(f"Loading file {idx+1}/{len(parquet_files)}: {os.path.basename(f)}")
                df = pd.read_parquet(f, engine="fastparquet")
                
                # Limit rows per file only for small-set testing
                if len(df) > 1000 and self.script_args.use_smallset:
                    df = df.head(100)  # Very small for testing
                    logger.info(f"Limited to {len(df)} rows for small set testing")
                
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
        
        # Limit dataset size only for testing
        if self.script_args.use_smallset:
            # Very small dataset for testing
            logger.info("Small set mode: limiting to 5 samples")
            full_df = full_df.head(5)
        
        logger.info(f"Full dataset size: {len(full_df)} samples")
        ds = Dataset.from_pandas(full_df)
        
        # Free pandas dataframe memory
        del full_df
        
        if size is not None:
            ds = ds.select(range(0, min(size, len(ds))))
        
        # Add original index column
        new_column = list(range(len(ds)))
        ds = ds.add_column("orig_index", new_column)
        logger.info(f"Original dataset length: {len(ds)}")
        
        # --- Apply train/test split if requested ---
        split_mode = getattr(self.script_args, 'split', 'all')
        if split_mode in ('train', 'test'):
            split_path = getattr(self.script_args, 'split_indices_path', None)
            split_info = get_or_create_split(data_path, split_path)
            selected_indices = split_info['train_indices'] if split_mode == 'train' else split_info['test_indices']
            # Filter to indices that exist in current dataset (handles size limits)
            valid = [i for i in selected_indices if i < len(ds)]
            ds = ds.select(valid)
            logger.info(f"Applied '{split_mode}' split: {len(ds)} samples (from {len(selected_indices)} indices)")
        
        # Expand to 2 preference pairs per example (chosen vs rejected_1, chosen vs rejected_2)
        expanded_rows = []
        for i in range(len(ds)):
            row = ds[i]
            row_dict = {k: row[k] for k in row.keys()}
            orig_idx = row_dict["orig_index"]
            for pair_idx in range(2):
                new_row = dict(row_dict)
                new_row["pair_idx"] = pair_idx
                new_row["data_index"] = orig_idx * 2 + pair_idx  # unique global pair index based on orig position
                expanded_rows.append(new_row)
        ds = Dataset.from_list(expanded_rows)
        logger.info(f"Expanded to {len(ds)} preference pairs (2 per example)")
        
        # --- Chunked processing with disk persistence & resumability ---
        # Instead of one giant ds.map() that builds a 500GB Arrow table (which
        # deadlocks during finalization), process in small chunks, save each to
        # disk immediately, then concatenate from disk at the end.
        
        chunk_size = 200  # examples per chunk
        total = len(ds)
        num_chunks = (total + chunk_size - 1) // chunk_size
        
        cache_key = hashlib.md5(
            f"{data_path}_{total}_{self.script_args.max_length}_{self.script_args.use_smallset}_{split_mode}_"
            f"{getattr(self.script_args, 'completion_format', 'free_text')}".encode()
        ).hexdigest()[:12]
        chunks_dir = os.path.join(os.path.dirname(data_path), f"chunks_{cache_key}")
        os.makedirs(chunks_dir, exist_ok=True)
        
        logger.info(f"Processing {total} examples in {num_chunks} chunks of {chunk_size} (saving to {chunks_dir})")
        
        for chunk_idx in range(num_chunks):
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_idx:04d}")
            
            # Resumability: skip already-processed chunks (but only if the
            # chunk is a complete HF Dataset directory — a Modal worker that
            # died mid-write can leave a half-populated dir without
            # dataset_info.json, which would cause load_from_disk to fail at
            # concat time. Treat such dirs as needing re-processing.)
            marker = os.path.join(chunk_path, "dataset_info.json")
            if os.path.exists(marker):
                logger.info(f"Chunk {chunk_idx+1}/{num_chunks} already exists, skipping")
                continue
            if os.path.isdir(chunk_path):
                logger.warning(
                    f"Chunk {chunk_idx+1}/{num_chunks} at {chunk_path} is incomplete "
                    f"(no dataset_info.json) — removing and re-processing"
                )
                import shutil
                shutil.rmtree(chunk_path, ignore_errors=True)
            
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total)
            chunk_ds = ds.select(range(start, end))
            
            # Process this chunk (single-process, small enough to finalize quickly)
            chunk_ds = chunk_ds.map(
                lambda examples: self._formatting_func_batched(examples, processor),
                batched=True,
                batch_size=32,
                num_proc=1,
                remove_columns=chunk_ds.column_names
            )
            
            # Save chunk to disk immediately (small enough to not hang)
            chunk_ds.save_to_disk(chunk_path)
            logger.info(f"Chunk {chunk_idx+1}/{num_chunks} done: {len(chunk_ds)} samples (saved to disk)")
            
            # Free memory from raw chunk data
            del chunk_ds
            import gc
            gc.collect()
        
        # Concatenate all chunks from disk (load lazily one at a time to limit
        # peak memory — concatenate_datasets memory-maps the underlying Arrow
        # files rather than copying into RAM).
        from datasets import concatenate_datasets
        logger.info(f"Concatenating {num_chunks} chunks from {chunks_dir}...")
        processed_chunks = []
        for chunk_idx in range(num_chunks):
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_idx:04d}")
            marker = os.path.join(chunk_path, "dataset_info.json")
            if os.path.exists(marker):
                processed_chunks.append(load_from_disk(chunk_path))
            else:
                logger.error(
                    f"Chunk {chunk_idx+1}/{num_chunks} at {chunk_path} missing or "
                    f"incomplete at concat time — aborting. Re-run preprocess to rebuild."
                )
                raise FileNotFoundError(f"Incomplete chunk: {chunk_path}")
        ds = concatenate_datasets(processed_chunks)
        del processed_chunks
        
        logger.info(f"Final dataset: {len(ds)} samples (length-filtered inline during processing)")
        
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
                # Load image
                image_bytes = self.dataset_adapter.extract_image_bytes(example)
                if not image_bytes:
                    continue
                
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                all_images.append(image)
                
                # Extract chosen/rejected.
                # Phase 0.6 D3: when completion_format=='letter', condition the head-build
                # forward on a single A/B/C completion (matching PPO generation), not the
                # full free-text answer.
                if getattr(self.script_args, "completion_format", "free_text") == "letter":
                    chosen, rejected = self.dataset_adapter.get_chosen_rejected_letter(example)
                else:
                    chosen, rejected = self.dataset_adapter.get_chosen_rejected(example)
                
                prompt_text = self.dataset_adapter.get_prompt_text(example)
                
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
        
        pv_chosen_splits, grid_chosen_list = self.model_wrapper.extract_pixel_values(inputs_chosen)
        pv_rejected_splits, grid_rejected_list = self.model_wrapper.extract_pixel_values(inputs_rejected)
        
        # Calculate prompt lengths
        all_tokens_prompt = processor.tokenizer(prompt_templates, padding=False)["input_ids"]
        
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            example = {k: examples[k][orig_idx] for k in examples.keys()}
            
            # Extract chosen/rejected again for metadata (must mirror the variant
            # used above for the actual tokenized inputs).
            if getattr(self.script_args, "completion_format", "free_text") == "letter":
                chosen, rejected = self.dataset_adapter.get_chosen_rejected_letter(example)
            else:
                chosen, rejected = self.dataset_adapter.get_chosen_rejected(example)
            prompt_text = self.dataset_adapter.get_prompt_text(example)
            pair_idx = int(example.get('pair_idx', 0))
            
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
            
            # Qwen2-VL grid tokens — use the already-split grid lists (mirrors
            # extract_pixel_values which handles the concatenated-array case).
            results["image_grid_thw_chosen"].append(
                grid_chosen_list[idx_in_valid] if grid_chosen_list else None
            )
            results["image_grid_thw_rejected"].append(
                grid_rejected_list[idx_in_valid] if grid_rejected_list else None
            )
            for k in ["video_grid_thw"]:
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



