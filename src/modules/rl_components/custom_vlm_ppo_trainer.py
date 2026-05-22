import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import PreTrainedModel
import logging

from .fast_rl import FastRLNode
from .caa_feedback import compute_causal_reward_penalty, compute_dispersive_loss

logger = logging.getLogger(__name__)

class PPOVLMController:
    """
    Token-Level Dense Reward PPO Trainer for Vision-Language Models.

    Key architectural changes (v2):
      - Dense token-level reward: projects hidden states at EVERY generated token
        onto reward heads, providing per-token bias signal to GAE.
      - Logit-grounded reward: adds a component measuring actual token probability
        shift, preventing null-space exploitation (representation hacking).
      - Token-level KL folded directly into per-token reward for native GAE absorption.
      - Causal penalty uses mean-token embedding drift (not EOS-only).
      - Dispersive loss on mean-pooled embeddings (O(B^2), not O((B*T)^2)).
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
        vf_coef: float = 0.1,
        lambda_causal: float = 0.5,
        delta_margin: float = 1.0,
        lambda_dispersive: float = 0.01,
        logit_reward_coef: float = 0.1,
        max_gen_tokens: int = 256,
    ):
        """
        Args:
            active_policy: The 3B model (with LoRA attached and active).
                           Disabling the LoRA adapter yields the reference policy.
            reward_heads_weight: Tensor of shape (K, hidden_size) — reward direction vectors.
            accelerator: Hugging Face Accelerate instance.
            fast_rl_node: Instance of FastRLNode for dynamic reward head balancing.
            lambda_causal: Causal deviation penalty strength (in reward).
            delta_margin: Tolerance margin for embedding drift before penalty activates.
            lambda_dispersive: Weight of dispersive regularization loss.
            logit_reward_coef: Weight of the logit-grounded reward component.
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
        self.lambda_causal = lambda_causal
        self.delta_margin = delta_margin
        self.lambda_dispersive = lambda_dispersive
        self.logit_reward_coef = logit_reward_coef
        
        self.reward_heads_weight = reward_heads_weight.to(accelerator.device, dtype=torch.bfloat16)
        self.max_gen_tokens = max_gen_tokens

        # Running normalization for task reward (keeps r_task in same scale as KL/logit components)
        self._r_task_running_mean = torch.tensor(0.0, device=accelerator.device)
        self._r_task_running_var = torch.tensor(1.0, device=accelerator.device)
        self._r_task_initialized = False
        self._r_task_ema_decay = 0.99

    def extract_logits_values_and_hidden(self, model, value_head, input_ids, attention_mask, pixel_values, kwargs):
        """Forward pass returning logits, per-token values, and full penultimate hidden states."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            **kwargs
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # final layer for values
        values = value_head(hidden_states).squeeze(-1)
        penultimate = outputs.hidden_states[-2]  # penultimate layer for reward projection
        return logits, values, penultimate

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

    def compute_dense_gae(self, dense_rewards: torch.Tensor, values: torch.Tensor, loss_mask: torch.Tensor):
        """
        Compute GAE with DENSE per-token rewards.

        Args:
            dense_rewards: (batch, seq_len) — reward signal at every token position.
            values: (batch, seq_len) — value estimates per token.
            loss_mask: (batch, seq_len) — which tokens are valid.
        Returns:
            advantages: (batch, seq_len)
            returns: (batch, seq_len)
        """
        batch_size, seq_len = values.shape
        advantages = torch.zeros_like(values)
        lastgaelam = torch.zeros(batch_size, device=values.device)

        for t in reversed(range(seq_len)):
            nextvalues = values[:, t + 1] if t < seq_len - 1 else torch.zeros(batch_size, device=values.device)
            delta = dense_rewards[:, t] + self.gamma * nextvalues * loss_mask[:, t] - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam * loss_mask[:, t]
            advantages[:, t] = lastgaelam

        returns = advantages + values
        return advantages, returns

    def step(self, batch, optimizer_policy, optimizer_value):
        """
        Executes a single token-level dense reward PPO step.

        Changes from v1 (sequence-level):
          - Reward projected at EVERY token position, not just EOS
          - KL penalty integrated per-token into dense reward
          - Logit-grounded reward prevents null-space exploitation
          - Causal penalty from mean-token drift (not EOS-only)
          - Dispersive loss on mean-pooled embeddings
          - No separate reference generation pass for embeddings
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
        
        # 1. GENERATION — Active policy only (no separate reference generation needed)
        self.policy.eval()
        with torch.inference_mode():
            unwrapped_policy = self.accelerator.unwrap_model(self.policy)
            
            curr_outputs = unwrapped_policy.generate(
                prompt_input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=pixel_values,
                **kwargs,
                max_new_tokens=self.max_gen_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True
            )
                
        # Reconstruct dynamic attention masks natively
        pad_token_id = getattr(self.policy.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 151643  # Qwen2 default pad token
        
        def _left_pad_generated(sequences, pad_id):
            """Convert right-padded generated sequences to left-padded."""
            attention_mask = (sequences != pad_id).long()
            batch_size, seq_len = sequences.shape
            content_lens = attention_mask.sum(dim=1)
            if (content_lens == seq_len).all():
                return sequences, attention_mask
            new_sequences = torch.full_like(sequences, pad_id)
            new_mask = torch.zeros_like(attention_mask)
            for i in range(batch_size):
                clen = content_lens[i].item()
                new_sequences[i, seq_len - clen:] = sequences[i, :clen]
                new_mask[i, seq_len - clen:] = 1
            return new_sequences, new_mask
        
        curr_outputs, curr_attention_mask = _left_pad_generated(curr_outputs, pad_token_id)

        # Build scoring kwargs — pass through vision metadata for correct M-RoPE and embeddings
        def _build_scoring_kwargs(gen_outputs, original_kwargs):
            scoring_kw = {}
            # image_grid_thw / video_grid_thw: unchanged (describe images, not sequence length)
            if "image_grid_thw" in original_kwargs:
                scoring_kw["image_grid_thw"] = original_kwargs["image_grid_thw"]
            if "video_grid_thw" in original_kwargs:
                scoring_kw["video_grid_thw"] = original_kwargs["video_grid_thw"]
            # mm_token_type_ids: extend to match generated sequence length
            if "mm_token_type_ids" in original_kwargs:
                orig_mm = original_kwargs["mm_token_type_ids"]
                gen_len = gen_outputs.shape[1]
                extra = gen_len - orig_mm.shape[1]
                if extra > 0:
                    ext = torch.zeros(
                        (orig_mm.shape[0], extra),
                        dtype=orig_mm.dtype,
                        device=orig_mm.device
                    )
                    scoring_kw["mm_token_type_ids"] = torch.cat([orig_mm, ext], dim=1)
                else:
                    scoring_kw["mm_token_type_ids"] = orig_mm
            return scoring_kw
        
        curr_scoring_kwargs = _build_scoring_kwargs(curr_outputs, kwargs)

        # 2. ACTIVE POLICY FORWARD — extract logits, values, and ALL hidden states
        #    Pass pixel_values so vision tokens get proper ViT embeddings (not generic token embeddings)
        #    and image_grid_thw is in curr_scoring_kwargs for correct M-RoPE position computation.
        self.policy.train()
        self.value_head.train()
        
        curr_logits, curr_values, curr_penultimate = self.extract_logits_values_and_hidden(
            self.policy, self.value_head, curr_outputs, curr_attention_mask, pixel_values, curr_scoring_kwargs
        )
        curr_logprobs, _ = self.compute_logprobs(curr_logits, curr_outputs)
        
        # FIX: loss_mask must only include GENERATED tokens (exclude prompt + padding)
        # After left-padding: content is right-aligned, prompt is first part of content
        total_seq_len = curr_outputs.shape[1]
        content_lens = curr_attention_mask.sum(dim=1)  # (B,) actual content length per sample
        prompt_lens = prompt_attention_mask.sum(dim=1)  # (B,) actual prompt length per sample
        # Generation starts at: (total_seq_len - content_len) + prompt_len in the padded sequence
        gen_starts = (total_seq_len - content_lens + prompt_lens).long()  # (B,)
        # In shifted space (logprobs are for positions 0..seq_len-2 predicting 1..seq_len-1)
        gen_starts_shifted = (gen_starts - 1).clamp(min=0)  # (B,)
        shifted_len = total_seq_len - 1  # length of shifted logprobs/labels
        positions = torch.arange(shifted_len, device=curr_outputs.device).unsqueeze(0)  # (1, T)
        loss_mask = (positions >= gen_starts_shifted.unsqueeze(1)) & curr_attention_mask[:, 1:].bool()
        loss_mask = loss_mask.float()
        
        # 3. REFERENCE POLICY FORWARD — logits and hidden states (single pass)
        #    Same pixel_values for correct vision embeddings (ViT has no adapters, output is identical)
        with torch.no_grad():
            self.policy.eval()
            with self.policy.disable_adapter():
                ref_logits, _, ref_penultimate = self.extract_logits_values_and_hidden(
                    self.policy, self.value_head, curr_outputs, curr_attention_mask, pixel_values, curr_scoring_kwargs
                )
                ref_logprobs, _ = self.compute_logprobs(ref_logits, curr_outputs)
            self.policy.train()

        # 4. TOKEN-LEVEL DENSE REWARD COMPUTATION
        # Align penultimate hidden states with the shifted logprob positions:
        # logprobs/loss_mask are (batch, seq_len-1) due to shift, so use penultimate[:, 1:, :]
        h_active = curr_penultimate[:, 1:, :].to(self.reward_heads_weight.dtype)   # (B, T, D)
        h_ref = ref_penultimate[:, 1:, :].to(self.reward_heads_weight.dtype)       # (B, T, D)
        
        # Project ALL token hidden states onto reward heads: (B, T, D) @ (D, K) → (B, T, K)
        r_token_k = torch.matmul(h_active, self.reward_heads_weight.T)  # (B, T, K)
        
        # Mean-pool over sequence for FastRL alpha update (global head importance)
        # Only pool over valid (non-padding) positions
        mask_expanded = loss_mask.unsqueeze(-1).float()  # (B, T, 1)
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
        r_mean_k = (r_token_k * mask_expanded).sum(dim=1) / valid_counts  # (B, K)
        
        # Update FastRL alpha weights (returns scalar composite — we use alpha directly for dense)
        _ = self.fast_rl.update(r_mean_k)
        alpha = self.fast_rl.alpha  # (K,) — current head weights
        
        # Dense task reward: weighted sum across heads at each token
        r_task_dense_raw = torch.matmul(r_token_k, alpha.to(r_token_k.dtype))  # (B, T)
        
        # Normalize r_task to unit variance using running statistics
        # This ensures KL/logit reward components (~0.01-0.3) are not dwarfed by
        # raw SVM projections (~10-30) which would make KL regularization ineffective.
        with torch.no_grad():
            valid_vals = r_task_dense_raw[loss_mask.bool()]
            batch_mean = valid_vals.mean()
            batch_var = valid_vals.var().clamp(min=1e-8)
            if not self._r_task_initialized:
                self._r_task_running_mean = batch_mean
                self._r_task_running_var = batch_var
                self._r_task_initialized = True
            else:
                self._r_task_running_mean = self._r_task_ema_decay * self._r_task_running_mean + (1 - self._r_task_ema_decay) * batch_mean
                self._r_task_running_var = self._r_task_ema_decay * self._r_task_running_var + (1 - self._r_task_ema_decay) * batch_var
        r_task_std = torch.sqrt(self._r_task_running_var + 1e-8)
        r_task_dense = (r_task_dense_raw - self._r_task_running_mean) / r_task_std  # ~ N(0,1)
        
        # 5. LOGIT-GROUNDED REWARD (Phase 2 — prevents null-space hacking)
        # Only reward INCREASED confidence (ReLU) — selective positive divergence signal
        logprob_diff = curr_logprobs - ref_logprobs.detach()  # (B, T)
        logit_reward = torch.clamp(logprob_diff, min=0) * self.logit_reward_coef  # (B, T)
        
        # 6. TOKEN-LEVEL KL PENALTY (Schulman k3 estimator — always non-negative)
        # k3 = exp(Δ) - 1 - Δ, where Δ = log(π/π_ref)
        # This avoids cancellation with logit_reward (which uses raw Δ) because:
        #   - logit_reward = max(0, Δ) * coef   (linear in positive Δ)
        #   - kl_penalty = (exp(Δ)-1-Δ) * beta  (quadratic near 0, exponential for large Δ)
        # Net effect: small positive divergence is rewarded, large divergence is penalized.
        token_kl = torch.exp(logprob_diff) - 1.0 - logprob_diff  # (B, T) — always ≥ 0
        token_kl = token_kl.clamp(max=10.0)  # Prevent explosion for large logprob_diff
        
        # 7. CAUSAL PENALTY (mean-token embedding drift)
        # Use mean embedding across valid positions (not EOS-only)
        e_active_mean = (h_active.float() * mask_expanded).sum(dim=1) / valid_counts  # (B, D)
        e_ref_mean = (h_ref.float() * mask_expanded).sum(dim=1) / valid_counts  # (B, D)
        
        causal_penalty = compute_causal_reward_penalty(
            e_ref_mean, e_active_mean,
            lambda_causal=self.lambda_causal,
            delta_margin=self.delta_margin,
        )  # (B,) — scalar per sequence, broadcast to all tokens
        
        # Distribute causal penalty evenly across valid tokens
        seq_lens = loss_mask.sum(dim=1).clamp(min=1).float()  # (B,)
        causal_per_token = (causal_penalty / seq_lens).unsqueeze(1).expand_as(r_task_dense)  # (B, T)
        
        # 8. COMPOSE DENSE REWARD
        # r_t = r_task_t + logit_reward_t + causal_per_token_t - beta * kl_t
        dense_rewards = (r_task_dense + logit_reward + causal_per_token - self.kl_beta * token_kl) * loss_mask
        
        # 9. DENSE GAE
        curr_values_aligned = curr_values[:, :-1]  # align with shifted positions
        advantages, returns = self.compute_dense_gae(
            dense_rewards.detach(), curr_values_aligned.detach(), loss_mask
        )
        # Normalize advantages ONLY over valid (generated) positions
        # Prompt positions have garbage advantages (-values[t]) that would dilute the signal
        valid_advs = advantages[loss_mask.bool()]
        adv_mean = valid_advs.mean()
        adv_std = valid_advs.std().clamp(min=1e-8)
        advantages = (advantages - adv_mean) / adv_std
        
        # 10. PPO SURROGATE LOSS (UNSCALED — trust region preserved)
        ratio = torch.exp(curr_logprobs - ref_logprobs.detach())
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_range)
        
        seq_pg_loss = (torch.max(pg_loss1, pg_loss2) * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)
        pg_loss = seq_pg_loss.mean()
        
        # 11. VALUE LOSS
        v_loss = 0.5 * ((curr_values_aligned - returns) ** 2) * loss_mask
        v_loss = v_loss.sum(dim=1) / loss_mask.sum(dim=1)
        v_loss = v_loss.mean()
        
        # 12. DISPERSIVE REGULARIZATION (mean-pooled embeddings, O(B^2))
        dispersive_loss = compute_dispersive_loss(e_active_mean)
        
        # 13. TOTAL LOSS
        loss = pg_loss + self.vf_coef * v_loss + self.lambda_dispersive * dispersive_loss
        
        # Backprop
        self.accelerator.backward(loss)
        
        optimizer_policy.step()
        optimizer_value.step()
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        
        return {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "dispersive_loss": dispersive_loss.item(),
            "causal_penalty": causal_penalty.mean().item(),
            "reward_dense_mean": dense_rewards.sum(dim=1).mean().item(),
            "reward_task": r_task_dense[loss_mask.bool()].mean().item(),
            "logit_reward": logit_reward[loss_mask.bool()].mean().item(),
            "kl": token_kl[loss_mask.bool()].mean().item(),
        }
