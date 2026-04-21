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
        elif len(parquet_files) > 10:
            parquet_files = parquet_files[:10]
        
        dfs = []
        for idx, f in enumerate(parquet_files):
            try:
                df = pd.read_parquet(f, engine="fastparquet")
                if len(df) > 1000 and self.script_args.use_smallset:
                    df = df.head(100)
                elif len(df) > 5000:
                    df = df.head(1000)
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
            full_df = full_df.head(5)
        elif len(full_df) > 10000:
            full_df = full_df.head(2000)
        
        ds = Dataset.from_pandas(full_df)
        del full_df
        
        if size is not None:
            ds = ds.select(range(0, min(size, len(ds))))
        
        # We don't expand into 2 pairs per example. We just need the unique prompt contexts.
        logger.info(f"Loaded {len(ds)} unique prompts for PPO generation.")
        
        ds = ds.map(
            lambda example: self._formatting_func(example, processor),
            batched=False,
            num_proc=1
        )
        
        ds = ds.filter(
            lambda x: len(x["input_ids"]) <= self.script_args.max_length,
            num_proc=1
        )
        
        ds.set_format(type="torch")
        return ds
    
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
            
            for k in ["image_grid_thw", "video_grid_thw"]:
                if k in inputs:
                    result[k] = inputs[k]
            
            return result
        except Exception as e:
            logger.warning(f"Error processing example in RL config: {e}")
            raise
