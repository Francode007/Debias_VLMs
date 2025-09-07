import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

# Core libraries
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
import pandas as pd
import tqdm
from collections import defaultdict

# ML libraries
from accelerate import Accelerator
from accelerate.utils import gather_object
import evaluate
from datasets import Dataset
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import PaddingStrategy
from transformers.trainer_pt_utils import nested_detach
from peft import LoraConfig, TaskType, get_peft_model
from trl import RewardTrainer
from trl.trainer.utils import decode_and_strip_padding, print_rich_table

# Image processing
from PIL import Image
import io
import glob
import ast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device and compatibility utilities
class DeviceManager:
    """Manages device selection and compatibility settings"""
    
    @staticmethod
    def get_optimal_device():
        """Detect and return the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @staticmethod
    def get_optimal_dtype(device: str):
        """Get optimal dtype based on device"""
        if device == "cuda":
            # CUDA supports bfloat16
            return torch.bfloat16
        elif device == "mps":
            # MPS has better support for float16
            return torch.float16
        else:
            # CPU fallback to float32
            return torch.float32
    
    @staticmethod
    def get_attention_implementation(device: str):
        """Get optimal attention implementation based on device"""
        if device == "cuda":
            try:
                import flash_attn
                return "flash_attention_2"
            except ImportError:
                logger.warning("Flash attention not available, falling back to eager attention")
                return "eager"
        else:
            # MPS and CPU don't support flash attention
            return "eager"
    
    @staticmethod
    def configure_torch_backends(device: str):
        """Configure PyTorch backends for optimal performance"""
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif device == "mps":
            # MPS-specific optimizations
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

@dataclass
class ScriptArguments:
    """Configuration arguments for the script"""
    # Training parameters
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=5e-6)
    num_train_epochs: Optional[int] = field(default=1)
    optim: Optional[str] = field(default="adamw_hf")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    max_length: Optional[int] = field(default=1024)
    
    # Memory management
    batch_size: Optional[int] = field(default=1, metadata={"help": "Batch size for data processing (affects memory usage)"})
    dataloader_batch_size: Optional[int] = field(default=None, metadata={"help": "DataLoader batch size (defaults to batch_size if not set)"})
    
    # Model parameters
    use_lora: Optional[bool] = field(default=False)
    base_model: Optional[str] = field(default='Qwen/Qwen2.5-VL-7B-Instruct')
    freeze_pretrained: Optional[bool] = field(default=False)
    
    # Device parameters
    device: Optional[str] = field(default="auto")  # auto, cuda, mps, cpu
    force_fp32: Optional[bool] = field(default=False)  # Force float32 for compatibility
    
    # Paths and logging
    data_path: Optional[str] = field(default='./sb_bench_data/data')
    log_dir: Optional[str] = field(default='./reward_models_sb_bench')
    cls_embs_path: Optional[str] = field(default='./embeddings_output')
    wandb_name: Optional[str] = field(default="qwen_vl_reward_sb_bench")
    
    # Training configuration
    loss_type: Optional[str] = field(default='origin')
    use_smallset: Optional[bool] = field(default=False)
    save_steps: Optional[int] = field(default=100)
    debug: Optional[bool] = field(default=False)
    
    # HuggingFace token
    hf_token: Optional[str] = field(default=None)

class ModelLoader:
    """Handles model loading with device compatibility"""
    
    def __init__(self, script_args: ScriptArguments, device_manager: DeviceManager):
        self.script_args = script_args
        self.device_manager = device_manager
        self.device = self._setup_device()
        self.dtype = self._setup_dtype()
        self.attention_impl = self._setup_attention()
        
    def _setup_device(self):
        """Setup device based on arguments"""
        if self.script_args.device == "auto":
            device = self.device_manager.get_optimal_device()
        else:
            device = self.script_args.device
            
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_dtype(self):
        """Setup data type based on device and arguments"""
        if self.script_args.force_fp32:
            dtype = torch.float32
        else:
            dtype = self.device_manager.get_optimal_dtype(self.device)
        
        logger.info(f"Using dtype: {dtype}")
        return dtype
    
    def _setup_attention(self):
        """Setup attention implementation"""
        attention_impl = self.device_manager.get_attention_implementation(self.device)
        logger.info(f"Using attention implementation: {attention_impl}")
        return attention_impl
    
    def load_model_and_processor(self):
        """Load model and processor with optimal settings"""
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError:
            logger.error("Qwen2VL model not found. Please install the required transformers version.")
            raise
        
        # Configure backends
        self.device_manager.configure_torch_backends(self.device)
        
        # Load processor
        processor = AutoProcessor.from_pretrained(self.script_args.base_model)
        
        # Prepare model loading arguments
        model_kwargs = {
            "num_labels": 1,
            "torch_dtype": self.dtype,
        }
        
        # Add attention implementation if supported
        if self.attention_impl != "eager":
            model_kwargs["attn_implementation"] = self.attention_impl
        
        # Device mapping
        if self.device == "cuda":
            # Use accelerate for CUDA
            accelerator = Accelerator()
            device_map = accelerator.local_process_index
            model_kwargs["device_map"] = device_map
        else:
            # For MPS and CPU, load to specific device
            model_kwargs["device_map"] = None
        
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.script_args.base_model,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if model_kwargs["device_map"] is None:
                model = model.to(self.device)
                
        except Exception as e:
            logger.warning(f"Failed to load model with optimal settings: {e}")
            logger.info("Falling back to CPU with float32...")
            
            # Fallback to CPU with basic settings
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.script_args.base_model,
                num_labels=1,
                torch_dtype=torch.float32,
                device_map=None
            )
            model = model.to("cpu")
            self.device = "cpu"
            self.dtype = torch.float32
        
        # Add score head
        model.score = nn.Linear(model.config.hidden_size, 1, bias=False)
        
        return model, processor

class DatasetBuilder:
    """Handles dataset creation and processing"""
    
    def __init__(self, script_args: ScriptArguments):
        self.script_args = script_args
        self.token_patterns = {
            "qwen": [151644, 46593, 198],  # <|im_start|>assistant\n
        }
    
    def find_token_for_gating(self, lst: List[int], model_family: str) -> int:
        """Find the last occurrence of a token_pattern in a list"""
        token_pattern = self.token_patterns[model_family]
        token_pattern_len = len(token_pattern)
        search_end = len(lst)
        
        for j in range(search_end - token_pattern_len, -1, -1):
            if lst[j : j + token_pattern_len] == token_pattern:
                return j
        raise ValueError("Token pattern not found in the list.")
    
    def build_dataset(self, data_path: str, processor, split: str = 'train', size: Optional[int] = None):
        """Build dataset from parquet files"""
        logger.info(f"Loading dataset from: {data_path}")
        
        # Load all Parquet files from directory
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        dfs = []
        for f in parquet_files:
            try:
                df = pd.read_parquet(f, engine="fastparquet")
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid parquet files could be loaded")
        
        full_df = pd.concat(dfs, ignore_index=True)
        ds = Dataset.from_pandas(full_df)
        
        if size is not None:
            ds = ds.select(range(0, min(size, len(ds))))
        
        # Add index column
        new_column = list(range(len(ds)))
        ds = ds.add_column("data_index", new_column)
        logger.info(f"Dataset length: {len(ds)}")
        
        # Apply formatting with controlled batch processing
        batch_size_for_map = min(self.script_args.batch_size, 100)  # Limit to prevent memory issues
        ds = ds.map(
            lambda example: self._formatting_func(example, processor),
            batched=False,
            num_proc=4,
            batch_size=batch_size_for_map
        )
        
        # Filter by length
        ds = ds.filter(
            lambda x: (len(x["input_ids_chosen"]) <= self.script_args.max_length and 
                      len(x["input_ids_rejected"]) <= self.script_args.max_length),
            num_proc=4
        )
        
        len_before_filter = len(ds)
        ds = ds.filter(
            lambda x: x["prompt_length"] < self.script_args.max_length,
            num_proc=4
        )
        len_after_filter = len(ds)
        logger.info(f"Filtered {len_before_filter - len_after_filter} samples due to length")
        
        ds.set_format(type="torch")
        return ds
    
    def _formatting_func(self, example, processor):
        """Format individual examples"""
        try:
            # Parse metadata
            additional_metadata = ast.literal_eval(example['additional_metadata'])
            
            # Load image
            image_data = example['file_name']
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Get answers
            label = example['label']
            chosen = example[f'ans{label}']
            rejected_idx = 0 if label != 0 else 1
            rejected = example[f'ans{rejected_idx}']
            
            # Create prompt
            prompt_text = example['context'] + " " + example['question']
            
            # Create messages
            chosen_messages = [{
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": prompt_text + " Answer: " + chosen}
                ]
            }]
            rejected_messages = [{
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": prompt_text + " Answer: " + rejected}
                ]
            }]
            prompt_messages = [{
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": prompt_text}
                ]
            }]
            
            # Apply chat templates
            prompt_plus_chosen = processor.apply_chat_template(chosen_messages, tokenize=False)
            prompt_plus_rejected = processor.apply_chat_template(rejected_messages, tokenize=False)
            prompt_template = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            kwargs = {
                "padding": "max_length", 
                "truncation": True, 
                "max_length": self.script_args.max_length, 
                "return_tensors": "pt"
            }
            
            inputs_chosen = processor(text=[prompt_plus_chosen], images=[image], **kwargs)
            inputs_rejected = processor(text=[prompt_plus_rejected], images=[image], **kwargs)
            
            # Calculate prompt length
            tokens_prompt = processor.tokenizer.encode(prompt_template)
            try:
                prompt_len = self.find_token_for_gating(tokens_prompt, "qwen")
            except:
                prompt_len = len(tokens_prompt) - 1
            
            return {
                "pixel_values_chosen": inputs_chosen["pixel_values"][0],
                "input_ids_chosen": inputs_chosen["input_ids"][0],
                "attention_mask_chosen": inputs_chosen["attention_mask"][0],
                "pixel_values_rejected": inputs_rejected["pixel_values"][0],
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
        except Exception as e:
            logger.warning(f"Error processing example {example.get('data_index', 'unknown')}: {e}")
            raise

def create_custom_forward(model, dtype):
    """Create custom forward function for the model"""
    def custom_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prompt_length: Optional[torch.Tensor] = None
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Ensure inputs are on the correct device and dtype
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device, dtype=dtype)
        
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs.hidden_states[-1]
        logits = self.score(hidden_states)
        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
        
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # Extract embeddings
        chose_emb = hidden_states[0, sequence_lengths[0], :]
        rej_emb = hidden_states[1, sequence_lengths[1], :]
        prompt_emb = hidden_states[0, (prompt_length-1):(prompt_length+1), :]
        
        emb = torch.cat([chose_emb[None,...], rej_emb[None,...], prompt_emb], 0)
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=emb,
            attentions=transformer_outputs.attentions,
        )
    
    return custom_forward

@dataclass
class RewardDataCollatorWithPadding:
    """Data collator for reward model training with batch size control"""
    processor: AutoProcessor
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    batch_size: Optional[int] = field(default=1)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Limit batch size to prevent memory issues
        if len(features) > self.batch_size:
            logger.warning(f"Batch size {len(features)} exceeds configured batch_size {self.batch_size}, truncating")
            features = features[:self.batch_size]
        
        merged_features_input_ids = []
        merged_features_attention_mask = []
        merged_features_pixel_values = []
        
        for feature in features:
            # Chosen
            merged_features_input_ids.append(feature["input_ids_chosen"])
            merged_features_attention_mask.append(feature["attention_mask_chosen"])
            merged_features_pixel_values.append(feature["pixel_values_chosen"])
            # Rejected
            merged_features_input_ids.append(feature["input_ids_rejected"])
            merged_features_attention_mask.append(feature["attention_mask_rejected"])
            merged_features_pixel_values.append(feature["pixel_values_rejected"])
        
        # Stack tensors
        batch = {
            "input_ids": torch.stack(merged_features_input_ids),
            "attention_mask": torch.stack(merged_features_attention_mask),
            "pixel_values": torch.stack(merged_features_pixel_values),
            "prompt": features[0]["prompt"],
            "chosen": features[0]["chosen"],
            "rejected": features[0]["rejected"],
            "prompt_plus_chosen_response": features[0]["prompt_plus_chosen_response"],
            "prompt_plus_rejected_response": features[0]["prompt_plus_rejected_response"],
            "prompt_length": features[0]["prompt_length"],
            "data_index": features[0]["data_index"],
        }
        
        return batch

class RewardVisualizer(RewardTrainer):
    """Custom reward trainer for visualization"""
    
    def __init__(self, script_args: ScriptArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.script_args = script_args
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with device handling"""
        try:
            res = model(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                pixel_values=inputs["pixel_values"], 
                prompt_length=inputs["prompt_length"]
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
        """Prediction step with error handling"""
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
        """Visualize samples and save embeddings with batch processing"""
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
                # Handle batch processing - process each item in the batch
                if isinstance(inputs["data_index"], torch.Tensor):
                    batch_indices = inputs["data_index"].tolist()
                    if not isinstance(batch_indices, list):
                        batch_indices = [batch_indices]
                else:
                    batch_indices = [inputs["data_index"]]
                
                for batch_idx, data_index in enumerate(batch_indices):
                    fn = os.path.join(cls_embs_path, f"emb_{data_index}.npy")
                    
                    if os.path.exists(fn):
                        continue
                    
                    # Extract single sample from batch if needed
                    if len(batch_indices) > 1:
                        # Handle batched input
                        single_input = {
                            "input_ids": inputs["input_ids"][batch_idx*2:(batch_idx+1)*2],  # chosen + rejected
                            "attention_mask": inputs["attention_mask"][batch_idx*2:(batch_idx+1)*2],
                            "pixel_values": inputs["pixel_values"][batch_idx*2:(batch_idx+1)*2],
                            "prompt_length": inputs["prompt_length"] if isinstance(inputs["prompt_length"], int) else inputs["prompt_length"][batch_idx],
                            "data_index": data_index
                        }
                    else:
                        single_input = inputs
                    
                    _, logits, _, emb = self.prediction_step(
                        self.model, single_input, prediction_loss_only=False
                    )
                    
                    # Extract data - handle both single and batch cases
                    if isinstance(inputs["chosen"], list):
                        chosen_text = inputs["chosen"][batch_idx] if len(batch_indices) > 1 else inputs["chosen"][0]
                        rejected_text = inputs["rejected"][batch_idx] if len(batch_indices) > 1 else inputs["rejected"][0]
                        prompt = inputs["prompt"][batch_idx] if len(batch_indices) > 1 else inputs["prompt"][0]
                    else:
                        chosen_text = inputs["chosen"]
                        rejected_text = inputs["rejected"]
                        prompt = inputs["prompt"]
                    
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
                    if num_print_samples >= 0 and processed_samples >= num_print_samples:
                        break
                    
                    # Determine flag
                    if logits[0][0] > logits[0][1]:
                        table["flag"].extend([1])
                    else:
                        table["flag"].extend([0])
                    
                    # Save embeddings
                    cls_emb = emb.float().cpu().numpy()
                    cls_emb = cls_emb[None, ...]
                    np.save(fn, cls_emb)
                    
                    # Update table with embeddings and logits
                    if Accelerator().num_processes == 1:
                        table["cls_emb"].extend(gather_object([fn]))
                        table["logits"].extend(gather_object([
                            [round(inner_item, 4) for inner_item in item] 
                            for item in logits.tolist()
                        ]))
                    else:
                        table["cls_emb"].append(fn)
                        table["logits"].extend([
                            [round(inner_item, 4) for inner_item in item] 
                            for item in logits.tolist()
                        ])
                    
                    # Save intermediate results
                    if len(table['chosen_text']) % 1000 == 0:
                        df = pd.DataFrame(table)
                        df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}_interim.csv")
                        logger.info(f"Saved interim results after {len(table['chosen_text'])} samples")
                
                if num_print_samples >= 0 and processed_samples >= num_print_samples:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing batch {idx}: {e}")
                continue
        
        # Save final results
        df = pd.DataFrame(table)
        df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}.csv")
        logger.info(f"Processing completed. Total samples processed: {processed_samples}")
        
        return df

def main():
    """Main execution function"""
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Set up HuggingFace token
    if script_args.hf_token:
        os.environ["HF_TOKEN"] = script_args.hf_token
    
    # Initialize device manager
    device_manager = DeviceManager()
    
    # Initialize model loader
    model_loader = ModelLoader(script_args, device_manager)
    
    # Load model and processor
    logger.info("Loading model and processor...")
    model, processor = model_loader.load_model_and_processor()
    
    # Create custom forward function
    custom_forward_func = create_custom_forward(model, model_loader.dtype)
    model.forward = custom_forward_func.__get__(model, type(model))
    
    # Initialize dataset builder
    dataset_builder = DatasetBuilder(script_args)
    
    # Build dataset
    logger.info("Building dataset...")
    dataset = dataset_builder.build_dataset(script_args.data_path, processor)
    eval_dataset = dataset
    
    # Set up training arguments
    model_name_split = script_args.base_model.split("/")[-1]
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_{script_args.learning_rate}"
    
    # Use batch_size for training if not explicitly set
    if script_args.dataloader_batch_size is None:
        script_args.dataloader_batch_size = script_args.batch_size
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_name, 'logs'),
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.dataloader_batch_size,
        per_device_eval_batch_size=script_args.dataloader_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        label_names=[],
        fp16=model_loader.dtype == torch.float16,
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
        dataloader_num_workers=0,  # Reduce for memory constraints
    )
    
    # Define metrics
    accuracy = evaluate.load('accuracy')
    
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        predictions = np.argmax(predictions, axis=1)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)
    
    # Initialize trainer
    trainer = RewardVisualizer(
        script_args=script_args,
        model=model,
        args=training_args,
        tokenizer=processor.tokenizer,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            processor=processor, 
            max_length=script_args.max_length,
            batch_size=script_args.batch_size
        ),
    )
    
    # Run visualization
    logger.info("Starting visualization...")
    df = trainer.visualize_samples(int(1e8), script_args.cls_embs_path, script_args.data_path)
    
    logger.info("Visualization completed successfully!")
    return df

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)
