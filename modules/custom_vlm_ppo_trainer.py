import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import PreTrainedModel
import logging

from .fast_rl import FastRLNode
from .caa_feedback import compute_caa_weight

logger = logging.getLogger(__name__)

class PPOVLMController:
    """
    Decoupled PPO Trainer for Vision-Language Models.
    Integrates Phase 2 (FastRL) and Phase 3 (CAA Feedback) strictly decoupled.
    """
    def __init__(
        self,
        active_policy: PreTrainedModel,
        reward_heads_weight: torch.Tensor,
        accelerator: Accelerator,
        fast_rl_node: FastRLNode,
        kl_beta: float = 0.1,
        ppo_clip_range: float = 0.2,
        gamma: float = 1.0,
        lam: float = 0.95,
        vf_coef: float = 0.1
    ):
        """
        Args:
            active_policy: The 3B model (with LoRA attached and active).
                           We assume disabling the LoRA adapter yields the reference policy.
            reward_heads_weight: Tensor of shape (K, hidden_size) of the 100 PCA orthogonal directions.
            accelerator: Hugging Face Accelerate instance.
            fast_rl_node: Instance of FastRLNode for dynamic mirror descent updates.
        """
        self.policy = active_policy
        
        # Value head for PPO (scalar output mapping the hidden state of the active policy)
        if hasattr(self.policy.config, "hidden_size"):
            hidden_size = self.policy.config.hidden_size
        elif hasattr(self.policy.config, "text_config") and hasattr(self.policy.config.text_config, "hidden_size"):
            hidden_size = self.policy.config.text_config.hidden_size
        else:
            hidden_size = self.policy.get_input_embeddings().weight.shape[-1]
        self.value_head = nn.Linear(hidden_size, 1, bias=False).to(accelerator.device, dtype=torch.bfloat16)
        
        self.accelerator = accelerator
        self.fast_rl = fast_rl_node
        self.kl_beta = kl_beta
        self.ppo_clip_range = ppo_clip_range
        self.gamma = gamma
        self.lam = lam
        self.vf_coef = vf_coef
        
        self.reward_heads_weight = reward_heads_weight.to(accelerator.device, dtype=torch.bfloat16)

    def extract_logits_and_values(self, model, value_head, input_ids, attention_mask, pixel_values, kwargs, extract_embedding=False):
        """Forward pass to extract logits and values, and optionally the penultimate layer embedding."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            **kwargs
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1] # final layer
        values = value_head(hidden_states).squeeze(-1)
        
        if extract_embedding:
            penultimate = outputs.hidden_states[-2] 
            # Assuming padding is handled, grab the last token
            pad_token_id = getattr(model.config, "pad_token_id", None)
            if pad_token_id is not None:
                seq_lens = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
                seq_lens = seq_lens % input_ids.shape[-1]
                e_token = penultimate[torch.arange(penultimate.shape[0]), seq_lens, :]
            else:
                e_token = penultimate[:, -1, :]
            return logits, values, e_token
            
        return logits, values

    def extract_preference_embeddings(self, model, input_ids, attention_mask, pixel_values, kwargs):
        """Extract the <EOS> token embedding from the penultimate layer of the extractor."""
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                **kwargs
            )
            # Penultimate layer
            penultimate = outputs.hidden_states[-2] 
            # Assuming padding is handled, grab the last token
            pad_token_id = getattr(model.config, "pad_token_id", None)
            if pad_token_id is not None:
                seq_lens = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
                seq_lens = seq_lens % input_ids.shape[-1]
                e_token = penultimate[torch.arange(penultimate.shape[0]), seq_lens, :]
            else:
                e_token = penultimate[:, -1, :]
            return e_token

    def compute_logprobs(self, logits, labels):
        """Standard logprob extraction."""
        logprobs = torch.log_softmax(logits, dim=-1)
        # Shift logits and labels internally if decoding
        labels = labels[:, 1:].clone()
        logprobs = logprobs[:, :-1, :]
        
        loss_mask = (labels != -100)
        labels[labels == -100] = 0
        per_token_logprobs = torch.gather(logprobs, 2, labels.unsqueeze(2)).squeeze(2)
        return per_token_logprobs, loss_mask

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, loss_mask: torch.Tensor):
        """Calculate generalized advantage estimate."""
        batch_size, seq_len = values.shape
        advantages = torch.zeros_like(values)
        lastgaelam = 0
        
        # rewards are sequence-level (sparsely populated at the last valid token of each sequence)
        # We will map the scalar R_final to the end of the sequence.
        seq_rewards = torch.zeros_like(values)
        end_indices = loss_mask.int().sum(dim=1) - 1 # Last valid token
        
        for i in range(batch_size):
            if end_indices[i] >= 0:
                seq_rewards[i, end_indices[i]] = rewards[i]
                
        for t in reversed(range(seq_len)):
            nextvalues = values[:, t + 1] if t < seq_len - 1 else 0.0
            delta = seq_rewards[:, t] + self.gamma * nextvalues * loss_mask[:, t] - values[:, t]
            advantages[:, t] = lastgaelam = delta + self.gamma * self.lam * lastgaelam * loss_mask[:, t]
            
        returns = advantages + values
        return advantages, returns

    def step(self, batch, optimizer_policy, optimizer_value):
        """
        Executes a single custom PPO Step with decoupled CAA weighting.
        """
        prompt_input_ids = batch["input_ids"]
        prompt_attention_mask = batch["attention_mask"]
        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)
        video_grid_thw = batch.get("video_grid_thw", None)
        mm_token_type_ids = batch.get("mm_token_type_ids", None)
        
        kwargs = {}
        if image_grid_thw is not None: kwargs["image_grid_thw"] = image_grid_thw
        if video_grid_thw is not None: kwargs["video_grid_thw"] = video_grid_thw
        if mm_token_type_ids is not None: kwargs["mm_token_type_ids"] = mm_token_type_ids
        
        # 1. TRUE DUAL GENERATION
        self.policy.eval()
        with torch.no_grad():
            # Generate $y_{curr}$ from Active Policy
            unwrapped_policy = self.accelerator.unwrap_model(self.policy)
            
            curr_outputs = unwrapped_policy.generate(
                prompt_input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=pixel_values,
                **kwargs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True
            )
            
            # Generate $y_{init}$ from Reference Policy
            with self.policy.disable_adapter():
                init_outputs = unwrapped_policy.generate(
                    prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    pixel_values=pixel_values,
                    **kwargs,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True
                )
                
        # Reconstruct dynamic attention masks natively
        pad_token_id = getattr(self.policy.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 151643 # Qwen2 default pad token, or fallback to 0
        
        # Re-pad generated sequences to the LEFT for Flash Attention compatibility.
        # generate() produces right-padded outputs (content + pad tokens at end).
        # Qwen2.5-VL with Flash Attention requires left-padding.
        def _left_pad_generated(sequences, pad_id):
            """Convert right-padded generated sequences to left-padded."""
            attention_mask = (sequences != pad_id).long()
            batch_size, seq_len = sequences.shape
            # Count actual content length per sequence
            content_lens = attention_mask.sum(dim=1)
            # If all sequences have the same length (no padding), skip
            if (content_lens == seq_len).all():
                return sequences, attention_mask
            # Re-arrange: move padding to the left
            new_sequences = torch.full_like(sequences, pad_id)
            new_mask = torch.zeros_like(attention_mask)
            for i in range(batch_size):
                clen = content_lens[i].item()
                # Content is at the start of the original (right-padded) sequence
                new_sequences[i, seq_len - clen:] = sequences[i, :clen]
                new_mask[i, seq_len - clen:] = 1
            return new_sequences, new_mask
        
        curr_outputs, curr_attention_mask = _left_pad_generated(curr_outputs, pad_token_id)
        init_outputs, init_attention_mask = _left_pad_generated(init_outputs, pad_token_id)

        # Build post-generation kwargs for scoring forward passes.
        # Key differences from generation kwargs:
        #   1. pixel_values=None: images were already consumed during generate() and
        #      embedded into the KV cache. The generated input_ids still contain image
        #      placeholder tokens, but passing raw pixel_values again would cause the
        #      model to try to re-embed them, leading to a token count mismatch.
        #   2. mm_token_type_ids must be extended: generated sequences are longer than
        #      prompts by max_new_tokens. New tokens are text (type=0), so we pad with 0s.
        #   3. image_grid_thw / video_grid_thw are kept as-is (they describe the image
        #      patches, not the sequence length).
        def _build_scoring_kwargs(gen_outputs, gen_attention_mask, original_kwargs):
            """Build kwargs suitable for a full scoring forward pass on generated sequences.
            pixel_values are NOT passed for scoring (images consumed during generate()),
            so image_grid_thw/video_grid_thw are also excluded."""
            scoring_kw = {}
            
            if "mm_token_type_ids" in original_kwargs:
                orig_mm = original_kwargs["mm_token_type_ids"]  # (batch, prompt_len)
                gen_len = gen_outputs.shape[1]
                extra = gen_len - orig_mm.shape[1]
                if extra > 0:
                    # Extend with zeros (text token type) to match generated length
                    ext = torch.zeros(
                        (orig_mm.shape[0], extra),
                        dtype=orig_mm.dtype,
                        device=orig_mm.device
                    )
                    scoring_kw["mm_token_type_ids"] = torch.cat([orig_mm, ext], dim=1)
                else:
                    scoring_kw["mm_token_type_ids"] = orig_mm
            return scoring_kw
        
        curr_scoring_kwargs = _build_scoring_kwargs(curr_outputs, curr_attention_mask, kwargs)
        init_scoring_kwargs = _build_scoring_kwargs(init_outputs, init_attention_mask, kwargs)

        # 2. Extract Active Policy Logprobs and Values over y_{curr}
        # NOTE: pixel_values=None for scoring passes. Images were consumed during generate()
        # and are encoded in the generated input_ids as placeholder tokens. Re-passing
        # pixel_values would cause a token/feature count mismatch.
        self.policy.train()
        self.value_head.train()
        
        curr_logits, curr_values = self.extract_logits_and_values(
            self.policy, self.value_head, curr_outputs, curr_attention_mask, None, curr_scoring_kwargs
        )
        curr_logprobs, loss_mask = self.compute_logprobs(curr_logits, curr_outputs)
        
        # 3. Extract Reference Policy Logprobs + Embeddings (batched single forward pass)
        with torch.no_grad():
            self.policy.eval()
            with self.policy.disable_adapter():
                bs = curr_outputs.shape[0]
                curr_len = curr_outputs.shape[1]
                init_len = init_outputs.shape[1]
                max_len = max(curr_len, init_len)
                
                # Left-pad shorter sequences to match for batching
                def _left_pad_to(seq, mask, target_len, pad_id):
                    if seq.shape[1] == target_len:
                        return seq, mask
                    extra = target_len - seq.shape[1]
                    padded_seq = torch.full((seq.shape[0], target_len), pad_id, dtype=seq.dtype, device=seq.device)
                    padded_mask = torch.zeros((seq.shape[0], target_len), dtype=mask.dtype, device=mask.device)
                    padded_seq[:, extra:] = seq
                    padded_mask[:, extra:] = mask
                    return padded_seq, padded_mask
                
                curr_padded, curr_mask_padded = _left_pad_to(curr_outputs, curr_attention_mask, max_len, pad_token_id)
                init_padded, init_mask_padded = _left_pad_to(init_outputs, init_attention_mask, max_len, pad_token_id)
                
                combined_ids = torch.cat([curr_padded, init_padded], dim=0)
                combined_mask = torch.cat([curr_mask_padded, init_mask_padded], dim=0)
                
                combined_kw = {}
                if "mm_token_type_ids" in curr_scoring_kwargs:
                    c_mm = curr_scoring_kwargs["mm_token_type_ids"]
                    i_mm = init_scoring_kwargs["mm_token_type_ids"]
                    if c_mm.shape[1] < max_len:
                        c_mm = torch.nn.functional.pad(c_mm, (max_len - c_mm.shape[1], 0), value=0)
                    if i_mm.shape[1] < max_len:
                        i_mm = torch.nn.functional.pad(i_mm, (max_len - i_mm.shape[1], 0), value=0)
                    combined_kw["mm_token_type_ids"] = torch.cat([c_mm, i_mm], dim=0)
                
                combined_logits, _, combined_emb = self.extract_logits_and_values(
                    self.policy, self.value_head, combined_ids, combined_mask, None, combined_kw, extract_embedding=True
                )
                
                ref_logits_curr = combined_logits[:bs]
                e_curr = combined_emb[:bs]
                e_init = combined_emb[bs:]
                
                # Crop logits back to original curr length if padded
                if curr_len < max_len:
                    ref_logits_curr = ref_logits_curr[:, (max_len - curr_len):, :]
                
                init_logprobs, _ = self.compute_logprobs(ref_logits_curr, curr_outputs)
                
            self.policy.train()
                
        # 4. KL Divergence Penalty (Token level)
        kl_divs = curr_logprobs - init_logprobs
        seq_kl = (kl_divs * loss_mask).sum(dim=1) # (Batch,)
        
        # 6. Orthogonal Scoring (Phase 1 application)
        # w_k dot e_curr => (Batch, K)
        # Cast embeddings to match reward heads dtype (accelerate may upcast to fp32)
        e_curr = e_curr.to(self.reward_heads_weight.dtype)
        e_init = e_init.to(self.reward_heads_weight.dtype)
        r_curr_k = torch.matmul(e_curr, self.reward_heads_weight.T)
        r_init_k = torch.matmul(e_init, self.reward_heads_weight.T)
        
        # 7. Phase 3 - CAA Feedback Weight
        w_hat = compute_caa_weight(r_init_k, r_curr_k)
        
        # 8. Phase 2 - Fast-RL Dynamic Balancing
        R_curr = self.fast_rl.update(r_curr_k)
        
        # 9. KL Regularization
        R_final = R_curr - self.kl_beta * seq_kl
        
        # 10. Loss Optimization (GAE via R_final)
        # Align values with loss_mask: values has shape (batch, seq_len) but
        # logprobs/loss_mask are shifted by 1 position, so shape is (batch, seq_len-1).
        # Use values[:, :-1] as current token values for GAE computation.
        curr_values_aligned = curr_values[:, :-1]
        advantages, returns = self.compute_gae(R_final.detach(), curr_values_aligned.detach(), loss_mask)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Surrogate Loss
        ratio = torch.exp(curr_logprobs - init_logprobs.detach())
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_range)
        
        # Per-token pg loss -> Per-sequence pg loss
        seq_pg_loss = torch.max(pg_loss1, pg_loss2) * loss_mask
        seq_pg_loss = seq_pg_loss.sum(dim=1) / loss_mask.sum(dim=1)
        
        # Apply Causal Feedback Weight! \hat{w} * L_PPO
        scaled_pg_loss = (w_hat * seq_pg_loss).mean()
        
        # Value Loss
        v_loss = 0.5 * ((curr_values_aligned - returns) ** 2) * loss_mask
        v_loss = v_loss.sum(dim=1) / loss_mask.sum(dim=1)
        v_loss = v_loss.mean()
        
        loss = scaled_pg_loss + self.vf_coef * v_loss
        
        # Backprop
        self.accelerator.backward(loss)
        
        optimizer_policy.step()
        optimizer_value.step()
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        
        return {
            "loss": loss.item(),
            "pg_loss": scaled_pg_loss.item(),
            "v_loss": v_loss.item(),
            "reward": R_curr.mean().item(),
            "kl": seq_kl.mean().item(),
            "w_hat_mean": w_hat.mean().item()
        }
