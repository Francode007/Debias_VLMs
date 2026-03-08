"""
Dataset Processing Module

This module handles dataset creation, processing, and formatting for the reward model
training pipeline, including memory-efficient loading and preprocessing.
"""

import os
import glob
import ast
import io
import logging
import pandas as pd
from PIL import Image
from datasets import Dataset
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
        
        # Apply formatting
        ds = ds.map(
            lambda example: self._formatting_func(example, processor),
            batched=False,
            num_proc=1
        )
        
        # Filter by length
        ds = ds.filter(
            lambda x: (len(x["input_ids_chosen"]) <= self.script_args.max_length and
                      len(x["input_ids_rejected"]) <= self.script_args.max_length),
            num_proc=1
        )
        len_before_filter = len(ds)
        ds = ds.filter(
            lambda x: x["prompt_length"] < self.script_args.max_length,
            num_proc=1
        )
        len_after_filter = len(ds)
        logger.info(f"Filtered {len_before_filter - len_after_filter} samples due to length")
        
        ds.set_format(type="torch")
        return ds
    
    def _formatting_func(self, example, processor):
        """
        Format individual examples for reward model training.
        
        Input:
            example: Raw example from dataset (dict-like object)
            processor: HuggingFace processor for tokenization and image processing
        
        Output:
            dict: Formatted example with tokenized inputs and metadata
        
        Process:
            1. Extract and validate example structure
            2. Handle flattened file_name structure from parquet
            3. Parse metadata and load image from bytes
            4. Extract chosen/rejected answers based on label
            5. Create conversation messages with image and text
            6. Apply chat templates for tokenization
            7. Process inputs with proper padding and length constraints
            8. Calculate prompt length for reward gating
            9. Return formatted example with all required fields
        
        Purpose:
            Transform raw dataset examples into the format required
            for reward model training, including proper tokenization,
            image processing, and sequence structure.
        
        Raises:
            ValueError: If example structure is invalid
            KeyError: If required keys are missing from example
        """
        try:
            # Check what we actually received - LazyRow behaves like dict but isn't one
            if not hasattr(example, '__getitem__') or not hasattr(example, 'keys'):
                logger.error(f"Expected dict-like object but got {type(example)}: {str(example)[:200]}")
                raise ValueError(f"Expected dict-like object, got {type(example)}")
            
            # Convert LazyRow to dict if needed for easier debugging
            if hasattr(example, '_data'):
                example_dict = {k: example[k] for k in example.keys()}
                logger.debug(f"LazyRow keys: {list(example.keys())}")
            else:
                example_dict = example
                
            # Handle flattened file_name structure
            if 'file_name' in example:
                image_data = example['file_name']
            elif 'file_name.bytes' in example:
                # Handle flattened structure from parquet
                image_data = {
                    'bytes': example['file_name.bytes'],
                    'path': example['file_name.path']
                }
            else:
                logger.error(f"Missing image data keys. Available keys: {list(example.keys())}")
                raise KeyError("No image data found - missing both 'file_name' and 'file_name.bytes'")
                
            # Parse metadata
            additional_metadata = ast.literal_eval(example['additional_metadata'])
            
            # Load image - handle different formats
            if isinstance(image_data, dict) and 'bytes' in image_data:
                # Handle format: {'bytes': b'...'}
                image_bytes = image_data['bytes']
            elif isinstance(image_data, bytes):
                # Handle direct bytes
                image_bytes = image_data
            else:
                raise ValueError(f"Unexpected image data format: {type(image_data)}")
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # SB-Bench: label = correct/non-stereotypical answer index; other two are stereotypical
            label = int(example['label'])
            chosen = example[f'ans{label}']
            wrong_indices = [i for i in range(3) if i != label]
            pair_idx = int(example.get('pair_idx', 0))
            rejected = example[f'ans{wrong_indices[pair_idx]}']
            
            # Prompt: context + question (no answer in user turn)
            prompt_text = example['context'] + " " + example['question']
            
            # User/assistant format so token gating finds assistant response boundary
            chosen_messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": chosen}]}
            ]
            rejected_messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": rejected}]}
            ]
            prompt_messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
            ]
            
            # Apply chat templates
            prompt_plus_chosen = processor.apply_chat_template(chosen_messages, tokenize=False)
            prompt_plus_rejected = processor.apply_chat_template(rejected_messages, tokenize=False)
            prompt_template = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs: truncation=False to avoid breaking image token alignment; long sequences filtered below
            kwargs = {
                "padding": "max_length",
                "truncation": False,
                "max_length": self.script_args.max_length,
                "return_tensors": "pt",
            }
            
            inputs_chosen = processor(text=[prompt_plus_chosen], images=[image], **kwargs)
            inputs_rejected = processor(text=[prompt_plus_rejected], images=[image], **kwargs)
            
            # Calculate prompt length
            tokens_prompt = processor.tokenizer.encode(prompt_template)
            try:
                prompt_len = self.find_token_for_gating(tokens_prompt, "qwen")
            except:
                prompt_len = len(tokens_prompt) - 1
            
            result = {
                "pixel_values_chosen": inputs_chosen["pixel_values"],
                "input_ids_chosen": inputs_chosen["input_ids"][0],
                "attention_mask_chosen": inputs_chosen["attention_mask"][0],
                "pixel_values_rejected": inputs_rejected["pixel_values"],
                "input_ids_rejected": inputs_rejected["input_ids"][0],
                "attention_mask_rejected": inputs_rejected["attention_mask"][0],
                "data_index": example['data_index'],
                "prompt": prompt_text,
                "chosen": chosen,
                "rejected": rejected,
                "prompt_plus_chosen_response": prompt_plus_chosen,
                "prompt_plus_rejected_response": prompt_plus_rejected,
                'prompt_length': prompt_len
            }
            
            # Capture Qwen2-VL specific arguments if present
            for k in ["image_grid_thw", "video_grid_thw"]:
                if k in inputs_chosen:
                    result[f"{k}_chosen"] = inputs_chosen[k]
                if k in inputs_rejected:
                    result[f"{k}_rejected"] = inputs_rejected[k]
            
            return result
        except Exception as e:
            logger.warning(f"Error processing example {example.get('data_index', 'unknown') if isinstance(example, dict) else 'invalid'}: {e}")
            logger.warning(f"Example keys: {list(example.keys()) if isinstance(example, dict) else 'Not a dict'}")
            logger.warning(f"Error type: {type(e).__name__}")
            raise
