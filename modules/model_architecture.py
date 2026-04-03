"""
Model Architecture Module

This module contains custom forward functions and model modifications
for reward model training with vision-language models.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Tuple, List
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

logger = logging.getLogger(__name__)


def create_custom_forward(model, dtype):
    """
    Create custom forward function for reward model training.
    
    Input:
        model: The base vision-language model
        dtype: Target data type for computations
    
    Output:
        function: Custom forward function bound to the model
    
    Process:
        1. Define custom forward function with reward model modifications
        2. Handle device and dtype conversions for inputs
        3. Process through base model with error handling
        4. Compute reward scores using score head
        5. Calculate sequence lengths for proper pooling
        6. Extract embeddings for visualization
        7. Compute loss if labels provided
        8. Return structured output with logits, loss, and embeddings
    
    Purpose:
        Modify the standard vision-language model forward pass
        to support reward model training with proper loss computation,
        embedding extraction, and sequence-aware pooling.
    """
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
        prompt_length: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        Custom forward pass for reward model with vision-language input.
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Ensure inputs are on the correct device and dtype
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device, dtype=dtype)
        
        # Handle kwargs tensors (e.g. image_grid_thw)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(self.device)
        
        if pixel_values is not None and "mm_token_type_ids" not in kwargs and input_ids is not None:
            # Qwen2-VL specific: create mm_token_type_ids for M-RoPE
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = getattr(self.config, "image_token_id", 151655)
            video_token_id = getattr(self.config, "video_token_id", 151652)
            
            # Map vision tokens to type 1
            mm_token_type_ids[input_ids == image_token_id] = 1
            mm_token_type_ids[input_ids == video_token_id] = 1
            
            kwargs["mm_token_type_ids"] = mm_token_type_ids
        
        # Handle different model architectures
        if hasattr(self, 'model'):
            # For models with a separate transformer component
            model_to_call = self.model
        else:
            # For models where the main class is the transformer
            model_to_call = self
        
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
                output_hidden_states=False,
                return_dict=return_dict,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in model forward pass: {e}")
            if pixel_values is not None:
                logger.warning("Retrying without pixel_values")
                
                # Strip out multimodal kwargs safely
                safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["image_grid_thw", "video_grid_thw", "mm_token_type_ids"]}
                
                transformer_outputs = model_to_call(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=False,
                    return_dict=return_dict,
                    **safe_kwargs
                )
            else:
                raise

        # Use last_hidden_state only (no full hidden_states stack) to save memory
        hidden_states = transformer_outputs[0] if isinstance(transformer_outputs, tuple) else transformer_outputs.last_hidden_state

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
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        # Phase 1: no score head; use dummy logits so trainer interface still works
        pooled_logits = torch.zeros(batch_size, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        
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
        
        # Extract embeddings safely
        try:
            chose_emb = hidden_states[0, sequence_lengths[0], :]
            rej_emb = hidden_states[1, sequence_lengths[1], :]
            if prompt_length is not None and prompt_length > 0:
                prompt_emb = hidden_states[0, max(0, prompt_length-1):prompt_length+1, :]
            else:
                prompt_emb = hidden_states[0, max(0, sequence_lengths[0]-1):sequence_lengths[0]+1, :]
            
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
