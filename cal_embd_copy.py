import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
import gc

# Core libraries
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
import pandas as pd
import tqdm
from collections import defaultdict
import glob
import ast
import io

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
            return torch.bfloat16
        elif device == "mps":
            return torch.float16
        else:
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
            return "eager"
    
    @staticmethod
    def configure_torch_backends(device: str):
        """Configure PyTorch backends for optimal performance"""
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

@dataclass
class ScriptArguments:
    """Configuration arguments for the script"""
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=1e-6)
    num_train_epochs: Optional[int] = field(default=1)
    optim: Optional[str] = field(default="adamw_torch")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    max_length: Optional[int] = field(default=512)
    
    batch_size: Optional[int] = field(default=1, metadata={"help": "Batch size for data processing"})
    dataloader_batch_size: Optional[int] = field(default=None, metadata={"help": "DataLoader batch size"})
    
    model: Optional[str] = field(default='./qwen2-vl-2b', metadata={"help": "Model path"})
    use_lora: Optional[bool] = field(default=False)
    base_model: Optional[str] = field(default='./qwen2-vl-2b')
    freeze_pretrained: Optional[bool] = field(default=False)
    fallback_model: Optional[str] = field(default='./qwen2-vl-2b')
    load_in_8bit: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)
    
    device: Optional[str] = field(default="auto")
    force_fp32: Optional[bool] = field(default=False)
    
    data_path: Optional[str] = field(default='./Sb-Bench_Dataset/Real')
    log_dir: Optional[str] = field(default='./reward_models_sb_bench')
    cls_embs_path: Optional[str] = field(default='./embeddings_output')
    output_dir: Optional[str] = field(default='./outputs')
    wandb_name: Optional[str] = field(default="qwen_vl_reward_sb_bench")
    
    loss_type: Optional[str] = field(default='origin')
    use_smallset: Optional[bool] = field(default=True)
    save_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=50)
    logging_steps: Optional[int] = field(default=10)
    warmup_steps: Optional[int] = field(default=10)
    debug: Optional[bool] = field(default=True)
    
    hf_token: Optional[str] = field(default=None)

class ModelLoader:
    """Handles model loading with device compatibility"""
    
    def __init__(self, script_args: ScriptArguments, device_manager: DeviceManager):
        self.script_args = script_args
        self.device_manager = device_manager
        self.device = self._setup_device()
        self.dtype = self._setup_dtype()
        self.attention_impl = self._setup_attention()
        
        self.get_local_model_path = lambda x: x
        self.get_recommended_local_model = lambda: "./qwen2-vl-2b"
        self.list_local_models = lambda: print("Using local model path")
        
    def _setup_device(self):
        if self.script_args.device == "auto":
            device = self.device_manager.get_optimal_device()
        else:
            device = self.script_args.device
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_dtype(self):
        if self.script_args.force_fp32:
            dtype = torch.float32
        else:
            dtype = self.device_manager.get_optimal_dtype(self.device)
        logger.info(f"Using dtype: {dtype}")
        return dtype
    
    def _setup_attention(self):
        attention_impl = self.device_manager.get_attention_implementation(self.device)
        logger.info(f"Using attention implementation: {attention_impl}")
        return attention_impl
    
    def load_model_and_processor(self):
        from transformers import Qwen2VLForConditionalGeneration
        self.device_manager.configure_torch_backends(self.device)
        processor = AutoProcessor.from_pretrained(self.script_args.model)
        
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
            logger.info("Set pad_token_id to eos_token_id in processor")
        
        model_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
        }
        if self.attention_impl != "eager":
            model_kwargs["attn_implementation"] = self.attention_impl
        model_kwargs["device_map"] = None if self.device != "cuda" else "auto"
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(self.script_args.model, **model_kwargs)
        
        if model.config.pad_token_id is None:
            model.config.pad_token_id = processor.tokenizer.pad_token_id
            logger.info("Set pad_token_id in model config")
        
        if model_kwargs["device_map"] is None:
            model = model.to(self.device)
        
        if not hasattr(model, 'score'):
            hidden_size = getattr(model.config, 'hidden_size', 1280)
            model.score = nn.Linear(hidden_size, 1, bias=False)
            logger.info(f"Added score head with hidden_size: {hidden_size}")
        
        logger.info(f"Successfully loaded model: {self.script_args.model}")
        return model, processor

class DatasetBuilder:
    """Handles dataset creation and processing"""
    
    def __init__(self, script_args: ScriptArguments):
        self.script_args = script_args
        self.token_patterns = {
            "qwen": [151644, 46593, 198],  # <|im_start|>assistant\n
        }
    
    def find_token_for_gating(self, lst: List[int], model_family: str) -> int:
        token_pattern = self.token_patterns[model_family]
        token_pattern_len = len(token_pattern)
        search_end = len(lst)
        for j in range(search_end - token_pattern_len, -1, -1):
            if lst[j:j + token_pattern_len] == token_pattern:
                return j
        return len(lst) - 1
    
    def build_dataset(self, data_path: str, processor, split: str = 'train', size: Optional[int] = None):
        logger.info(f"Loading dataset from: {data_path}")
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        if self.script_args.use_smallset:
            parquet_files = parquet_files[:1]
            logger.info("Using small dataset: limited to 1 file")
        
        dfs = []
        for idx, f in enumerate(parquet_files):
            try:
                logger.info(f"Loading file {idx+1}/{len(parquet_files)}: {os.path.basename(f)}")
                df = pd.read_parquet(f, engine="fastparquet")
                if self.script_args.use_smallset:
                    df = df.head(50)
                    logger.info(f"Limited to {len(df)} rows for small set")
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid parquet files could be loaded")
        
        full_df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        
        if self.script_args.use_smallset:
            full_df = full_df.head(50)
            logger.info("Small set mode: limited to 50 samples")
        
        ds = Dataset.from_pandas(full_df)
        del full_df
        gc.collect()
        
        if size is not None:
            ds = ds.select(range(0, min(size, len(ds))))
        
        new_column = list(range(len(ds)))
        ds = ds.add_column("data_index", new_column)
        logger.info(f"Dataset length: {len(ds)}")
        
        ds = ds.map(
            lambda example: self._formatting_func(example, processor),
            batched=False,
            num_proc=1
        )
        
        len_before = len(ds)
        ds = ds.filter(lambda x: x is not None and all(
            key in x and x[key] is not None for key in [
                "pixel_values_chosen", "pixel_values_rejected",
                "input_ids_chosen", "input_ids_rejected"
            ]
        ))
        logger.info(f"Filtered {len_before - len(ds)} invalid samples")
        
        ds = ds.filter(
            lambda x: len(x["input_ids_chosen"]) <= self.script_args.max_length and 
                     len(x["input_ids_rejected"]) <= self.script_args.max_length,
            num_proc=1
        )
        ds = ds.filter(
            lambda x: x["prompt_length"] < self.script_args.max_length,
            num_proc=1
        )
        logger.info(f"Final dataset length: {len(ds)}")
        
        ds.set_format(type="torch")
        return ds
    
    def _formatting_func(self, example, processor):
        try:
            if not hasattr(example, '__getitem__') or not hasattr(example, 'keys'):
                logger.error(f"Invalid example type: {type(example)}")
                return None
            
            if 'file_name' in example:
                image_data = example['file_name']
            elif 'file_name.bytes' in example:
                image_data = {'bytes': example['file_name.bytes'], 'path': example['file_name.path']}
            else:
                logger.error(f"Missing image data for example {example.get('data_index', 'unknown')}")
                return None
            
            try:
                additional_metadata = ast.literal_eval(example['additional_metadata'])
            except:
                logger.warning(f"Invalid metadata for example {example.get('data_index', 'unknown')}")
                return None
            
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image_bytes = image_data['bytes']
            elif isinstance(image_data, bytes):
                image_bytes = image_data
            else:
                logger.error(f"Invalid image data format for example {example.get('data_index', 'unknown')}")
                return None
            
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image for example {example.get('data_index', 'unknown')}: {e}")
                return None
            
            label = example['label']
            chosen = example[f'ans{label}']
            rejected_idx = 0 if label != 0 else 1
            rejected = example[f'ans{rejected_idx}']
            prompt_text = example['context'] + " " + example['question']
            
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
            
            prompt_plus_chosen = processor.apply_chat_template(chosen_messages, tokenize=False)
            prompt_plus_rejected = processor.apply_chat_template(rejected_messages, tokenize=False)
            prompt_template = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            
            kwargs = {
                "padding": "max_length",
                "truncation": False,
                "max_length": self.script_args.max_length,
                "return_tensors": "pt"
            }
            
            inputs_chosen = processor(text=[prompt_plus_chosen], images=[image], **kwargs)
            inputs_rejected = processor(text=[prompt_plus_rejected], images=[image], **kwargs)
            
            if inputs_chosen["pixel_values"] is None or inputs_rejected["pixel_values"] is None:
                logger.warning(f"Skipping example {example.get('data_index', 'unknown')} due to missing pixel_values")
                return None
            
            tokens_prompt = processor.tokenizer.encode(prompt_template)
            prompt_len = self.find_token_for_gating(tokens_prompt, "qwen")
            
            result = {
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
            
            if self.script_args.debug:
                logger.debug(f"Processed example {example['data_index']}: {list(result.keys())}")
            return result
        except Exception as e:
            logger.warning(f"Error processing example {example.get('data_index', 'unknown')}: {e}")
            return None

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
        
        if input_ids is None or attention_mask is None or pixel_values is None:
            logger.error("Missing required inputs: input_ids, attention_mask, or pixel_values is None")
            raise ValueError("Invalid inputs: input_ids, attention_mask, or pixel_values is None")
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device, dtype=dtype)
        
        model_to_call = getattr(self, 'model', self)
        
        try:
            transformer_outputs = model_to_call(
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
        except Exception as e:
            logger.error(f"Error in model forward pass: {e}")
            raise
        
        hidden_states = transformer_outputs.hidden_states[-1]
        logits = self.score(hidden_states)
        
        batch_size = input_ids.shape[0]
        if self.config.pad_token_id is None:
            raise ValueError("pad_token_id must be defined in model config")
        
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)
        
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                self.config.problem_type = "regression"
            
            loss_fct = MSELoss()
            loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        try:
            chose_emb = hidden_states[0, sequence_lengths[0], :]
            rej_emb = hidden_states[1, sequence_lengths[1], :] if batch_size > 1 else chose_emb
            prompt_emb = hidden_states[0, max(0, prompt_length-1):prompt_length+1, :] if prompt_length is not None else chose_emb
            emb = torch.cat([chose_emb[None,...], rej_emb[None,...], prompt_emb], 0)
        except Exception as e:
            logger.warning(f"Error extracting embeddings: {e}, using fallback")
            emb = pooled_logits.unsqueeze(0).repeat(3, 1, 1)
        
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
    """Data collator for reward model training"""
    processor: AutoProcessor
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    batch_size: Optional[int] = field(default=1)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features or len(features) == 0:
            logger.warning("Empty features list received by data collator")
            return None
        
        feature = features[0]
        required_keys = [
            "input_ids_chosen", "attention_mask_chosen", "pixel_values_chosen",
            "input_ids_rejected", "attention_mask_rejected", "pixel_values_rejected"
        ]
        
        if not all(key in feature and feature[key] is not None for key in required_keys):
            logger.warning(f"Skipping invalid sample: missing or None values in {feature.get('data_index', 'unknown')}")
            return None
        
        batch = {
            "input_ids": torch.stack([feature["input_ids_chosen"], feature["input_ids_rejected"]]),
            "attention_mask": torch.stack([feature["attention_mask_chosen"], feature["attention_mask_rejected"]]),
            "pixel_values": torch.stack([feature["pixel_values_chosen"], feature["pixel_values_rejected"]]),
            "prompt_length": feature["prompt_length"],
            "prompt": feature["prompt"],
            "chosen": feature["chosen"],
            "rejected": feature["rejected"],
            "prompt_plus_chosen_response": feature["prompt_plus_chosen_response"],
            "prompt_plus_rejected_response": feature["prompt_plus_rejected_response"],
            "data_index": feature["data_index"],
        }
        
        return batch

class RewardVisualizer(RewardTrainer):
    """Custom reward trainer for visualization"""
    
    def __init__(self, script_args: ScriptArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.script_args = script_args
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if inputs is None:
            logger.error("Received None inputs in compute_loss")
            raise ValueError("Inputs cannot be None")
        
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
        if bsz < 2:
            logger.error(f"Invalid batch size {bsz} in compute_loss")
            raise ValueError(f"Expected batch size >= 2, got {bsz}")
        
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}, emb
        return loss, emb
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if inputs is None:
            logger.error("Received None inputs in prediction_step")
            raise ValueError("Inputs cannot be None")
        
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
        
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
        logits = torch.stack(logits).mean(dim=2).T
        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)
        
        return loss, logits, labels, emb
    
    def visualize_samples(self, num_print_samples: int, cls_embs_path: str, data_path: str):
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        
        if not os.path.exists(cls_embs_path):
            os.makedirs(cls_embs_path)
        
        logger.info(f"Eval dataset length: {len(self.eval_dataset)}")
        logger.info(f"Eval dataloader length: {len(eval_dataloader)}")
        
        processed_samples = 0
        for idx, inputs in tqdm.tqdm(enumerate(eval_dataloader), desc="Processing samples"):
            if inputs is None:
                logger.warning(f"Skipping batch {idx} due to None inputs")
                continue
            
            try:
                data_index = inputs["data_index"].item() if isinstance(inputs["data_index"], torch.Tensor) else inputs["data_index"]
                fn = os.path.join(cls_embs_path, f"emb_{data_index}.npy")
                
                if os.path.exists(fn):
                    continue
                
                with torch.no_grad():
                    _, logits, _, emb = self.prediction_step(
                        self.model, inputs, prediction_loss_only=False
                    )
                
                prompt = inputs["prompt"]
                chosen_text = inputs["chosen"]
                rejected_text = inputs["rejected"]
                source = "sb_bench"
                
                table["prompt"].append(prompt)
                table["chosen_text"].append(chosen_text)
                table["rejected_text"].append(rejected_text)
                table["prompt_plus_chosen_response"].append(chosen_text)
                table["prompt_plus_rejected_response"].append(rejected_text)
                table["source"].append(source)
                table["data_index"].append(data_index)
                
                table["flag"].extend([1 if logits[0][0] > logits[0][1] else 0])
                
                cls_emb = emb.float().cpu().numpy()
                cls_emb = cls_emb[None, ...]
                np.save(fn, cls_emb)
                
                table["cls_emb"].append(fn)
                table["logits"].extend([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()])
                
                processed_samples += 1
                if num_print_samples >= 0 and processed_samples >= num_print_samples:
                    break
                
                if processed_samples % 10 == 0:
                    df = pd.DataFrame(table)
                    df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}_interim.csv")
                    logger.info(f"Saved interim results after {processed_samples} samples")
                
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing sample {data_index}: {e}")
                continue
        
        df = pd.DataFrame(table)
        df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}.csv")
        logger.info(f"Processing completed. Total samples processed: {processed_samples}")
        
        return df

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    if script_args.hf_token:
        os.environ["HF_TOKEN"] = script_args.hf_token
    
    device_manager = DeviceManager()
    model_loader = ModelLoader(script_args, device_manager)
    
    logger.info("Loading model and processor...")
    model, processor = model_loader.load_model_and_processor()
    
    custom_forward_func = create_custom_forward(model, model_loader.dtype)
    model.forward = custom_forward_func.__get__(model, type(model))
    
    dataset_builder = DatasetBuilder(script_args)
    logger.info("Building dataset...")
    dataset = dataset_builder.build_dataset(script_args.data_path, processor)
    
    eval_dataset = dataset
    if script_args.dataloader_batch_size is None:
        script_args.dataloader_batch_size = script_args.batch_size
    
    training_args = TrainingArguments(
        output_dir=os.path.join(script_args.output_dir, 'logs'),
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.dataloader_batch_size,
        per_device_eval_batch_size=script_args.dataloader_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        eval_steps=script_args.eval_steps,
        save_strategy="steps",
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        label_names=[],
        fp16=model_loader.dtype == torch.float16,
        bf16=model_loader.dtype == torch.bfloat16,
        logging_strategy="steps",
        logging_steps=script_args.logging_steps,
        warmup_ratio=0.05,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        run_name=script_args.wandb_name,
        max_grad_norm=5.0,
        report_to='none',
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
    )
    
    # Add disable_dropout attribute if it doesn't exist
    if not hasattr(training_args, 'disable_dropout'):
        training_args.disable_dropout = False
        
    accuracy = evaluate.load('accuracy')
    
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        predictions = np.argmax(predictions, axis=1)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)
    
    trainer = RewardVisualizer(
        script_args=script_args,
        model=model,
        args=training_args,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            processor=processor,
            max_length=script_args.max_length,
            batch_size=script_args.batch_size
        ),
    )
    
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