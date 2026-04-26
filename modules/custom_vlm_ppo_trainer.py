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
                top_p=0.9
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
                    top_p=0.9
                )
                
        # Reconstruct dynamic attention masks natively
        pad_token_id = getattr(self.policy.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 151643 # Qwen2 default pad token, or fallback to 0
        curr_attention_mask = (curr_outputs != pad_token_id).long()
        init_attention_mask = (init_outputs != pad_token_id).long()

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
            """Build kwargs suitable for a full scoring forward pass on generated sequences."""
            scoring_kw = {}
            for k in ["image_grid_thw", "video_grid_thw"]:
                if k in original_kwargs:
                    scoring_kw[k] = original_kwargs[k]
            
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
        self.policy.train()
        self.value_head.train()
        
        curr_logits, curr_values = self.extract_logits_and_values(
            self.policy, self.value_head, curr_outputs, curr_attention_mask, pixel_values, curr_scoring_kwargs
        )
        curr_logprobs, loss_mask = self.compute_logprobs(curr_logits, curr_outputs)
        
        # 3. Extract Reference Policy Logprobs over y_{curr} AND e_curr simultaneously
        with torch.no_grad():
            self.policy.eval()
            with self.policy.disable_adapter():
                init_logits_curr_traj, _, e_curr = self.extract_logits_and_values(
                    self.policy, self.value_head, curr_outputs, curr_attention_mask, pixel_values, curr_scoring_kwargs, extract_embedding=True
                )
                init_logprobs, _ = self.compute_logprobs(init_logits_curr_traj, curr_outputs)
                
                # 5. Extract Embeddings for y_{init}
                e_init = self.extract_preference_embeddings(
                    self.policy, init_outputs, init_attention_mask, pixel_values, init_scoring_kwargs
                )
                
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
