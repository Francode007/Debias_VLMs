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


def create_custom_forward(model, dtype, token_position: str = "eos", letter_token_ids=None):
    """
    Create custom forward function for reward model training.
    
    Input:
        model: The base vision-language model
        dtype: Target data type for computations
        token_position (str): Which token's penultimate-layer hidden state to return.
            'eos'         — last non-pad token (legacy / backcompat).
            'post_letter' — the A/B/C letter token itself.
            'pre_letter'  — the token immediately BEFORE the letter; its logit is
                            what predicts the letter, matching PPO's binary-mode
                            `ans_pos` exactly (Phase 0.6 D3 fix).
        letter_token_ids: Iterable of token IDs that represent the answer letters
            A, B, or C under the model's tokenizer (any encoding variant — see
            extract.py for the resolution logic). Required when token_position is
            'post_letter' or 'pre_letter'; ignored for 'eos'.
    
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
    if token_position not in ("eos", "post_letter", "pre_letter"):
        raise ValueError(
            f"create_custom_forward: token_position={token_position!r} not supported. "
            "Must be one of 'eos', 'post_letter', 'pre_letter'. "
            "('all_letters_mean' is reserved for a future implementation.)"
        )
    if token_position in ("post_letter", "pre_letter"):
        if not letter_token_ids:
            raise ValueError(
                f"create_custom_forward: token_position={token_position!r} requires "
                "letter_token_ids (the tokenizer IDs for the A/B/C answer letters). "
                "Resolve them once at model-creation time and pass them in."
            )
        # Resolve to a single 1D long tensor for vectorised lookup inside the forward.
        _letter_token_ids_tensor = torch.tensor(sorted(set(int(t) for t in letter_token_ids)), dtype=torch.long)
    else:
        _letter_token_ids_tensor = None

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
            model_dtype = getattr(self.model, "dtype", torch.bfloat16)
            pixel_values = pixel_values.to(self.device, dtype=model_dtype)
        
        # Handle kwargs tensors (e.g. image_grid_thw)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(self.device)
        
        # Handle different model architectures
        if hasattr(self, 'model'):
            # For models with a separate transformer component
            model_to_call = self.model
        else:
            # For models where the main class is the transformer
            model_to_call = self

        # Filter kwargs to only those accepted by the inner model's forward()
        import inspect
        valid_params = set(inspect.signature(model_to_call.forward).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
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
                **filtered_kwargs
            )
        except Exception as e:
            logger.error(f"Error in model forward pass: {e}")
            if pixel_values is not None:
                logger.warning("Retrying without pixel_values")
                
                # Strip out multimodal kwargs safely
                safe_kwargs = {k: v for k, v in filtered_kwargs.items() if k not in ["image_grid_thw", "video_grid_thw"]}
                
                transformer_outputs = model_to_call(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                    **safe_kwargs
                )
            else:
                raise

        # Extract penultimate layer (-2) from full hidden states
        if hasattr(transformer_outputs, 'hidden_states') and transformer_outputs.hidden_states is not None:
            all_hidden_states = transformer_outputs.hidden_states
        elif isinstance(transformer_outputs, tuple) and len(transformer_outputs) > 2:
            all_hidden_states = transformer_outputs[2]
        else:
            # Debug: what is actually in transformer_outputs?
            out_keys = getattr(transformer_outputs, "keys", lambda: "No keys")()
            out_type = type(transformer_outputs)
            logger.error(f"Missing hidden_states. Output type: {out_type}, Keys: {out_keys}")
            raise ValueError(f"hidden_states not found in model output ({out_type}). Make sure output_hidden_states=True is effective.")
        
        # Extract the penultimate layer (3D Tensor: [Batch, Seq, Dim])
        penultimate_layer = all_hidden_states[-2]

        # Slice to get only the final token's embedding (2D Tensor: [Batch, Dim])
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
                # Find the index of the last non-padding token
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(penultimate_layer.device)
            else:
                sequence_lengths = -1

        # Choose the per-sample index into the penultimate layer.
        #
        # Phase 0.6 D3: when the chosen/rejected completion is a single A/B/C
        # letter, the head-build basis must be aligned with the position that
        # PPO scores at training time. PPO places the binary reward at
        # `ans_pos = gen_starts_shifted + k`, where `k` is the offset of the
        # parsed letter token within the generated suffix. In full-sequence
        # coordinates this is `(letter_pos - 1)` (the predictor of the letter)
        # for the typical k=0 case. We mirror that selection here.
        if token_position == "eos" or (isinstance(sequence_lengths, int) and sequence_lengths == -1):
            slice_indices = sequence_lengths  # int sentinel handled below
        elif input_ids is None:
            # Defensive: cannot find letter positions without input_ids.
            slice_indices = sequence_lengths
        else:
            device = penultimate_layer.device
            letter_ids_dev = _letter_token_ids_tensor.to(device)
            # (B, T) bool: True where token is one of the answer letters.
            is_letter = torch.isin(input_ids.to(device), letter_ids_dev)
            # Find the LAST occurrence of any letter token per row.
            # argmax on a reversed mask gives offset-from-end of the first True.
            T = input_ids.shape[-1]
            rev = torch.flip(is_letter.int(), dims=[-1])
            any_letter = is_letter.any(dim=-1)
            offset_from_end = rev.argmax(dim=-1)  # 0 if no True (also when row is all False)
            letter_pos = (T - 1 - offset_from_end).to(device)
            # Samples with no letter token fall back to the EOS position to
            # avoid silent indexing errors; they will be visible as outliers in
            # the embedding distribution and can be filtered downstream.
            letter_pos = torch.where(any_letter, letter_pos, sequence_lengths.to(letter_pos.dtype))
            if token_position == "post_letter":
                slice_indices = letter_pos
            else:  # pre_letter
                slice_indices = (letter_pos - 1).clamp(min=0)

        # Extract the hidden states at the chosen per-sample index.
        if isinstance(slice_indices, int) and slice_indices == -1:
            hidden_states = penultimate_layer[:, -1, :]
        else:
            hidden_states = penultimate_layer[
                torch.arange(batch_size, device=penultimate_layer.device), slice_indices
            ]

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
        
        # Extract embeddings safely across arbitrary batch size
        try:
            # hidden_states is already sliced to [Batch, Dim] representing the last token
            emb = hidden_states
        except Exception as e:
            logger.warning(f"Error extracting embeddings: {e}, using fallback")
            emb = pooled_logits.repeat(1, 1).view(batch_size, -1)
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=emb,
            attentions=transformer_outputs.attentions,
        )
    
    return custom_forward
