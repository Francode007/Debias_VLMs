"""
Phase 0.8 PPO trainer (probe-as-head reward).

Variant of `custom_vlm_ppo_trainer.PPOVLMController` specialised for the
`bias_aligned` probe-head deployment validated in
[Phase0.8/Multi_layer_head_analysis.md](../../../Phase0.8/Multi_layer_head_analysis.md)
and the cross-dataset workflow in
[Phase0.8/Debiasing_Workflow.md](../../../Phase0.8/Debiasing_Workflow.md).

Differences vs the legacy trainer:

1. **Configurable reward-head layer.** The projection layer is exposed via
   `reward_head_layer` (default = L13, the lead layer from §4½.10 of the
   analysis). The legacy trainer hard-codes the penultimate layer
   (`outputs.hidden_states[-2]`), which would throw away the multi-layer
   finding for any L != N-2.
2. **`bias_aligned` reward mode.** Composes
        r_total = w_corr  * 1[pred==gold]
                + w_bias  * -(w_biasA · h_L[post-letter])      (negate: high = biased)
                + w_ambig * 1[gold==C  AND pred==C]            (§4½.11 ambig-preservation)
                - kl_beta * token_kl
   as a *sparse, answer-position-anchored* token reward, mirroring the
   `binary` reward placement in the legacy trainer.
3. **Mandatory frozen-feature projection.** The reward head was fit on
   the un-adapted base model's hidden states, so `use_frozen_phi=True` is
   enforced (a misuse guard, not just a default).

Legacy `svm` / `binary` reward modes are preserved verbatim for backward
compatibility with Phase 0.5/0.6 reproductions, but the intended Phase 0.8
entry point is `reward_mode='bias_aligned'`.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import logging


class FirstTokenAllowlistProcessor(LogitsProcessor):
    """Restricts the FIRST generated token to a fixed allowlist of token IDs.

    Used during PPO rollouts to match the constrained-decoding eval setup:
    the model may only emit A / B / C as its first generated token. Subsequent
    tokens are unrestricted (the model will typically emit <eos> immediately).

    Without this, stochastic sampling over the full 152k vocab produces a
    train-time accuracy floor far below the model's actual constrained
    capability (we measured 0.40 train vs 0.62 vanilla constrained eval).
    """

    def __init__(self, allowed_token_ids, prompt_length: int):
        super().__init__()
        self.allowed = list(allowed_token_ids)
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        # input_ids: (B, T_so_far). T_so_far == prompt_length on the first
        # generation step, then grows. We only mask on the very first step.
        if input_ids.shape[1] != self.prompt_length:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed] = 0.0
        return scores + mask

from .fast_rl import FastRLNode
from .caa_feedback import compute_causal_reward_penalty, compute_dispersive_loss

logger = logging.getLogger(__name__)

class Phase08PPOController:
    """
    Phase 0.8 PPO trainer with probe-as-head (bias_aligned) reward.

    See module docstring for the design rationale. The class signature is
    a strict superset of `PPOVLMController`: every constructor argument from
    the legacy trainer is accepted unchanged, plus three new ones
    (`reward_head_layer`, `bias_aligned_coef`, `ambig_preservation_coef`).

    The `step()` method routes on `reward_mode`:
      * `'binary'`   — legacy +1/-1 task reward (Phase 0.5/0.6).
      * `'svm'`      — legacy dense SVM-projection reward (Phase 0.5/0.6).
      * `'bias_aligned'` — Phase 0.8 reward: binary correctness + negated
        probe projection at `reward_head_layer` + ambig-preservation bonus,
        all sparse at the answer-letter token; KL folded in as before.
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
        max_gen_tokens: int = 8,
        max_grad_norm: float = 1.0,
        reward_mode: str = "svm",
        target_kl: float = 0.0,
        kl_adapt_rate: float = 0.1,
        kl_beta_min: float = 0.05,
        kl_beta_max: float = 5.0,
        value_clip_range: float = 0.0,
        use_frozen_phi: bool = False,
        # ── PHASE 0.8 additions ────────────────────────────────────────────
        reward_head_layer: int = 13,
        bias_aligned_coef: float = 1.0,
        ambig_preservation_coef: float = 0.5,
        correctness_coef: float = 1.0,
        # ───────────────────────────────────────────────────────────────────
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
        self.max_grad_norm = max_grad_norm
        self.reward_mode = reward_mode  # "svm" or "binary"

        # Phase 0 collapse-mitigation knobs.
        self.target_kl = target_kl              # >0 enables adaptive KL
        self.kl_adapt_rate = kl_adapt_rate
        self.kl_beta_min = kl_beta_min
        self.kl_beta_max = kl_beta_max
        self.value_clip_range = value_clip_range  # >0 enables value clipping

        # Phase 0.6 D1: project reward heads against the FROZEN (LoRA-off)
        # penultimate representation φ_ref instead of the trainable φ_active.
        # The DRM SVM/PCA heads were fit on φ from the un-adapted base model;
        # projecting against the moving φ_active lets PPO "hack" the reward
        # by drifting φ itself rather than changing the policy's answer
        # distribution. With use_frozen_phi=True the reward signal is grounded
        # in a fixed feature space, and policy improvement is forced through
        # the logit / sampling path.
        self.use_frozen_phi = use_frozen_phi

        # ── PHASE 0.8 knobs ───────────────────────────────────────────────
        # `reward_head_layer` selects which entry of `outputs.hidden_states`
        # to project the reward head against. Default 13 follows the lead
        # layer chosen in §4½.10 / §4½.13 of Multi_layer_head_analysis.md.
        # Use -2 to reproduce the legacy penultimate-layer behaviour.
        self.reward_head_layer = int(reward_head_layer)
        self.bias_aligned_coef = float(bias_aligned_coef)
        self.ambig_preservation_coef = float(ambig_preservation_coef)
        self.correctness_coef = float(correctness_coef)

        # Mis-use guard: the bias_aligned probe was fit on un-adapted base
        # hidden states, so projecting against the trainable φ would let PPO
        # "hack" the reward by drifting representations (§4½.9 + §2.6.1
        # diagnosis in the analysis doc).
        if self.reward_mode == "bias_aligned" and not self.use_frozen_phi:
            logger.warning(
                "reward_mode='bias_aligned' requested without use_frozen_phi=True; "
                "forcing use_frozen_phi=True. The bias_aligned head was fit on "
                "frozen base-model hidden states and must be projected against "
                "the reference (LoRA-off) representation to remain valid."
            )
            self.use_frozen_phi = True
        # ──────────────────────────────────────────────────────────────────

        # Running normalization for task reward (keeps r_task in same scale as KL/logit components)
        self._r_task_running_mean = torch.tensor(0.0, device=accelerator.device)
        self._r_task_running_var = torch.tensor(1.0, device=accelerator.device)
        self._r_task_initialized = False
        self._r_task_ema_decay = 0.99
        self._letter_token_ids = None  # Cached for binary reward mode

    def _get_letter_token_ids(self):
        """Resolve token IDs for A, B, C letters. Cached after first call.
        
        Tries multiple encodings: ' A', 'A', '\\nA' to find single-token representations.
        Also builds a reverse lookup for ALL possible token IDs that could represent A/B/C.
        """
        if self._letter_token_ids is not None:
            return self._letter_token_ids
        unwrapped = self.accelerator.unwrap_model(self.policy)
        if hasattr(unwrapped, "tokenizer"):
            tok = unwrapped.tokenizer
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(unwrapped.config._name_or_path, trust_remote_code=True)
        
        letter_ids = {}  # {label_idx: set of matching token IDs}
        for idx, letter in enumerate(["A", "B", "C"]):
            candidates = set()
            # Try multiple possible encodings
            for prefix in ["", " ", "\n"]:
                encoded = tok.encode(f"{prefix}{letter}", add_special_tokens=False)
                if len(encoded) == 1:
                    candidates.add(encoded[0])
                elif len(encoded) > 1:
                    # The last token might be the letter
                    candidates.add(encoded[-1])
            # Also try the letter alone
            encoded_bare = tok.encode(letter, add_special_tokens=False)
            for t_id in encoded_bare:
                candidates.add(t_id)
            letter_ids[idx] = candidates
        
        # Convert to a flat lookup: {token_id: label_idx}
        # If a token_id maps to multiple labels, prefer exact single-token encoding
        flat_lookup = {}
        for idx, candidates in letter_ids.items():
            for t_id in candidates:
                if t_id not in flat_lookup:
                    flat_lookup[t_id] = idx
        
        logger.info(f"Binary reward: letter candidates = {letter_ids}, flat_lookup = {flat_lookup}")
        self._letter_token_ids = flat_lookup
        return self._letter_token_ids

    def extract_logits_values_and_hidden(self, model, value_head, input_ids, attention_mask, pixel_values, kwargs):
        """Forward pass returning logits, per-token values, and the reward-projection hidden states.

        Phase 0.8: the third return is `hidden_states[self.reward_head_layer]`
        (not always the penultimate layer). For Qwen2.5-VL the hidden_states
        tuple is length N+1 (embedding + N transformer outputs); index 13
        therefore corresponds to the output of transformer block 13, which is
        the same convention the probe-fit pipeline uses.
        """
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
        # Reward-projection layer. `self.reward_head_layer` is an int; negative
        # values index from the end (e.g. -2 = penultimate, the legacy default).
        reward_hidden = outputs.hidden_states[self.reward_head_layer]
        return logits, values, reward_hidden

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

    @torch.no_grad()
    def evaluate_subset(self, eval_dataloader) -> dict:
        """Greedy constrained generation on a held-out subset → accuracy.

        Mirrors the eval-sweep decoding (first-token allowlist over A/B/C,
        argmax, no sampling) so the mid-training signal is directly
        comparable to the post-hoc eval sweep numbers — unlike training-batch
        `binary_accuracy`, which is sampled at temperature=0.3 with batch=8
        and is therefore far too noisy to use as an early-stop signal.

        Returns:
            dict with keys: midtrain_eval_acc, midtrain_eval_n,
            midtrain_eval_parse_rate, midtrain_eval_pred_dist (A/B/C frac).
        """
        if eval_dataloader is None:
            return {}

        unwrapped_policy = self.accelerator.unwrap_model(self.policy)
        self.policy.eval()
        letter_lookup = self._get_letter_token_ids()  # {tok_id: label_idx}
        allowed_ids = list(letter_lookup.keys())

        n_total = 0
        n_correct = 0
        n_parsed = 0
        pred_counts = [0, 0, 0]  # A, B, C

        for batch in eval_dataloader:
            prompt_input_ids = batch["input_ids"].to(self.accelerator.device)
            prompt_attention_mask = batch["attention_mask"].to(self.accelerator.device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.accelerator.device)
            kwargs = {}
            for k in ("image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
                v = batch.get(k)
                if v is not None:
                    kwargs[k] = v.to(self.accelerator.device)

            logits_processor = LogitsProcessorList([
                FirstTokenAllowlistProcessor(
                    allowed_token_ids=allowed_ids,
                    prompt_length=prompt_input_ids.shape[1],
                )
            ])

            gen_out = unwrapped_policy.generate(
                prompt_input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=pixel_values,
                **kwargs,
                max_new_tokens=self.max_gen_tokens,
                do_sample=False,  # greedy — deterministic eval
                use_cache=True,
                logits_processor=logits_processor,
            )

            prompt_len = prompt_input_ids.shape[1]
            gold = batch.get("gold_label")
            bsz = gen_out.shape[0]
            for b in range(bsz):
                gen_tokens = gen_out[b, prompt_len:]
                pred_label = -1
                for tok in gen_tokens[:5]:
                    tok_id = tok.item()
                    if tok_id in letter_lookup:
                        pred_label = letter_lookup[tok_id]
                        break
                if pred_label >= 0:
                    n_parsed += 1
                    pred_counts[pred_label] += 1
                    g = gold[b].item() if gold is not None else -1
                    if g >= 0 and pred_label == g:
                        n_correct += 1
                n_total += 1

        acc = (n_correct / n_total) if n_total > 0 else 0.0
        parse_rate = (n_parsed / n_total) if n_total > 0 else 0.0
        denom = max(1, sum(pred_counts))
        return {
            "midtrain_eval_acc": acc,
            "midtrain_eval_n": n_total,
            "midtrain_eval_parse_rate": parse_rate,
            "midtrain_eval_pred_A": pred_counts[0] / denom,
            "midtrain_eval_pred_B": pred_counts[1] / denom,
            "midtrain_eval_pred_C": pred_counts[2] / denom,
        }

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
            
            # Constrained sampling: restrict the first generated token to
            # {A, B, C} so train-time decoding matches eval-time decoding.
            # PPO still gets exploration via temperature over the 3 letters.
            logits_processor = None
            if self.reward_mode in ("binary", "bias_aligned"):
                letter_lookup = self._get_letter_token_ids()  # {tok_id: label}
                allowed_ids = list(letter_lookup.keys())
                logits_processor = LogitsProcessorList([
                    FirstTokenAllowlistProcessor(
                        allowed_token_ids=allowed_ids,
                        prompt_length=prompt_input_ids.shape[1],
                    )
                ])

            # NOTE: temperature was lowered from 0.7 → 0.3 to reduce sampling
            # variance over the 3-way {A,B,C} action space. At 0.7, even a model
            # peaked on the correct letter sampled the wrong letter often enough
            # to keep mean train acc at ~0.40 (vs vanilla constrained-argmax 0.62).
            # At 0.3, sampling stays close to argmax while preserving enough
            # entropy for PPO exploration.
            curr_outputs = unwrapped_policy.generate(
                prompt_input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=pixel_values,
                **kwargs,
                max_new_tokens=self.max_gen_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                use_cache=True,
                logits_processor=logits_processor,
            )
                
        # Reconstruct dynamic attention masks natively
        pad_token_id = getattr(self.policy.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 151643  # Qwen2 default pad token
        
        def _left_pad_generated(sequences, pad_id):
            """Convert generated sequences (which may have both left AND right padding) to left-padded.
            
            HF generate() with left-padded inputs preserves left-padding and adds right-padding:
            Input:  [PAD PAD prompt_tokens]
            Output: [PAD PAD prompt_tokens gen_tokens PAD PAD]
            
            We need to extract the contiguous content block and right-align it.
            """
            attention_mask = (sequences != pad_id).long()
            batch_size, seq_len = sequences.shape
            content_lens = attention_mask.sum(dim=1)
            if (content_lens == seq_len).all():
                return sequences, attention_mask
            new_sequences = torch.full_like(sequences, pad_id)
            new_mask = torch.zeros_like(attention_mask)
            for i in range(batch_size):
                clen = content_lens[i].item()
                if clen == 0:
                    continue
                # Find first non-pad position (content start)
                first_content = attention_mask[i].argmax().item()
                # Copy the contiguous content block, right-aligned
                new_sequences[i, seq_len - clen:] = sequences[i, first_content:first_content + clen]
                new_mask[i, seq_len - clen:] = 1
            return new_sequences, new_mask
        
        curr_outputs, curr_attention_mask = _left_pad_generated(curr_outputs, pad_token_id)

        # ── PHASE 1 FIX ──────────────────────────────────────────────────────
        # Capture the OLD policy log-probs (= log π_θ at rollout time).
        # This is the policy that actually GENERATED `curr_outputs`. Because no
        # optimizer step has occurred in this batch yet, π_old == π_current
        # *weights-wise*, but we must snapshot the log-probs under no-grad here
        # and use them — not π_ref — in the PPO surrogate ratio.
        # ─────────────────────────────────────────────────────────────────────
        # (Computed after we build `curr_scoring_kwargs` just below — see below.)

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

        # PHASE 1: snapshot OLD-policy log-probs of the sampled tokens (no grad).
        # This is the correct π_old for the PPO surrogate ratio. We deliberately
        # use the SAME active policy (LoRA enabled) — π_old is the policy at
        # rollout time, not π_ref.
        with torch.no_grad():
            self.policy.eval()
            old_logits, _, _ = self.extract_logits_values_and_hidden(
                self.policy, self.value_head, curr_outputs, curr_attention_mask,
                pixel_values, curr_scoring_kwargs,
            )
            old_logprobs, _ = self.compute_logprobs(old_logits, curr_outputs)
            old_logprobs = old_logprobs.detach()

        # 2. ACTIVE POLICY FORWARD — extract logits, values, and ALL hidden states
        #    Pass pixel_values so vision tokens get proper ViT embeddings (not generic token embeddings)
        #    and image_grid_thw is in curr_scoring_kwargs for correct M-RoPE position computation.
        self.policy.train()
        self.value_head.train()
        
        curr_logits, curr_values, curr_reward_h = self.extract_logits_values_and_hidden(
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
                ref_logits, _, ref_reward_h = self.extract_logits_values_and_hidden(
                    self.policy, self.value_head, curr_outputs, curr_attention_mask, pixel_values, curr_scoring_kwargs
                )
                ref_logprobs, _ = self.compute_logprobs(ref_logits, curr_outputs)
            self.policy.train()

        # 4. TOKEN-LEVEL DENSE REWARD COMPUTATION (Phase 3 off-by-one fix)
        # logprobs/labels are shifted: positions 1..T-1 are predicted FROM contexts 0..T-2.
        # The hidden state that justified predicting token t+1 is the state AT position t,
        # so we use penultimate[:, :-1, :], not penultimate[:, 1:, :].
        h_active = curr_reward_h[:, :-1, :].to(self.reward_heads_weight.dtype)  # (B, T, D)
        h_ref = ref_reward_h[:, :-1, :].to(self.reward_heads_weight.dtype)      # (B, T, D)

        if self.reward_mode == "binary":
            # ─── BINARY REWARD MODE ──────────────────────────────────────────
            # Reward = +1 if generated letter == gold, -1 otherwise.
            # Placed as a sparse terminal reward at the LAST valid token position.
            gold_labels = batch.get("gold_label")  # (B,) tensor of 0/1/2
            if gold_labels is None:
                logger.warning("BINARY REWARD: gold_label NOT found in batch! Reward will be all -1.")
            # Generated portion starts at gen_starts in the full sequence
            batch_size = curr_outputs.shape[0]
            binary_reward_per_sample = torch.zeros(batch_size, device=curr_outputs.device)
            letter_token_ids = self._get_letter_token_ids()  # {0: id_A, 1: id_B, 2: id_C}

            num_correct = 0
            num_unparsed = 0
            # Track WHERE in the generated sequence the letter token was found.
            # -1 means "no letter token found in the first 5 generated tokens".
            # We use this below to place the reward at the position whose logit
            # *actually* predicted the letter — not on the (often non-letter)
            # first generated token (e.g. " The" / " Answer:" preambles).
            pred_offset_per_sample = [-1] * batch_size
            for b_idx in range(batch_size):
                gen_start = gen_starts[b_idx].item()
                # Get generated token ids (only the generated portion)
                gen_tokens = curr_outputs[b_idx, gen_start:]
                # Look for A/B/C token in the first few generated positions
                pred_label = -1
                for offset, tok in enumerate(gen_tokens[:5]):
                    tok_id = tok.item()
                    if tok_id in letter_token_ids:
                        pred_label = letter_token_ids[tok_id]
                        pred_offset_per_sample[b_idx] = offset
                        break

                gold = gold_labels[b_idx].item() if gold_labels is not None else -1

                # CRITICAL: if we can't parse prediction OR don't have gold, reward = -1
                if pred_label < 0 or gold < 0:
                    binary_reward_per_sample[b_idx] = -1.0
                    num_unparsed += 1
                elif pred_label == gold:
                    binary_reward_per_sample[b_idx] = 1.0
                    num_correct += 1
                else:
                    binary_reward_per_sample[b_idx] = -1.0

            # Stash for metrics (see below). Parsed-offset stats let us see if the
            # model is emitting the letter at position 0 (good) or buried after a
            # preamble (bad — base capability is hidden behind verbose generation).
            parsed = [o for o in pred_offset_per_sample if o >= 0]
            self._last_parse_success_rate = (len(parsed) / batch_size) if batch_size > 0 else 0.0
            self._last_pred_letter_offset_mean = (
                (sum(parsed) / len(parsed)) if parsed else float("nan")
            )

            # Log diagnostics on first few batches
            if not hasattr(self, '_binary_log_count'):
                self._binary_log_count = 0
            if self._binary_log_count < 3:
                self._binary_log_count += 1
                # Decode first sample's generated tokens for debugging
                gen_start_0 = gen_starts[0].item()
                gen_tok_ids = curr_outputs[0, gen_start_0:gen_start_0+5].tolist()
                gold_vals = gold_labels.tolist() if gold_labels is not None else "NONE"
                logger.info(
                    f"BINARY REWARD DEBUG (batch {self._binary_log_count}): "
                    f"letter_token_ids={letter_token_ids}, "
                    f"first_gen_tokens={gen_tok_ids}, "
                    f"gold_labels={gold_vals}, "
                    f"correct={num_correct}/{batch_size}, unparsed={num_unparsed}/{batch_size}"
                )

            # Construct sparse reward: place at the position whose logit ACTUALLY
            # predicted the answer letter, not the first generated token.
            #
            # Why: chat-tuned VLMs often emit a preamble (" The", " Answer:",
            # "<thinking>") before the letter even when prompted to answer with
            # only A/B/C. Crediting the first-generated-token's predictor with
            # the letter's reward pushes the model AWAY from whatever produced
            # the letter — actively destroying baseline capability (we saw the
            # vanilla 0.619 baseline degrade to ~0.40 under PPO before this fix).
            #
            # Position math: if the letter appeared at curr_outputs[b, gen_start+k],
            # the logit that predicted it sits at full-seq position (gen_start+k-1),
            # which equals (gen_starts_shifted[b] + k) in shifted/logprob coords.
            # When no letter is found in the first 5 tokens (k = -1), fall back to
            # gen_starts_shifted (the old behaviour) — these samples get r=-1 anyway
            # and we have no better position to place it.
            dense_rewards = torch.zeros_like(loss_mask)
            for b_idx in range(batch_size):
                k = pred_offset_per_sample[b_idx]
                if k >= 0:
                    ans_pos = gen_starts_shifted[b_idx].item() + k
                else:
                    ans_pos = gen_starts_shifted[b_idx].item()
                # Clamp to valid range (defensive — should always be inside loss_mask)
                if 0 <= ans_pos < loss_mask.shape[1] and loss_mask[b_idx, ans_pos] > 0:
                    dense_rewards[b_idx, ans_pos] = binary_reward_per_sample[b_idx]
            # NB: do NOT multiply by loss_mask again — answer-position is already inside it.
            # Keep the unmodified per-sample binary reward around for honest accuracy logging,
            # since `dense_rewards` will shortly be mixed with the KL penalty below.
            self._last_binary_reward_per_sample = binary_reward_per_sample.detach()

            # Token-level KL penalty (folded into reward, the standard PPO-with-KL recipe).
            # With kl_beta>0 this pulls the policy back toward π_ref and prevents
            # unbounded drift / mode collapse. In the previous code path this was
            # computed but discarded — that is why the policy degraded toward chance.
            logprob_diff = curr_logprobs - ref_logprobs.detach()
            token_kl = torch.exp(logprob_diff) - 1.0 - logprob_diff
            token_kl = token_kl.clamp(max=10.0)
            dense_rewards = (dense_rewards - self.kl_beta * token_kl) * loss_mask

            # Diagnostics (no SVM, no causal penalty in binary mode)
            logit_reward = torch.zeros_like(logprob_diff)
            r_task_dense = dense_rewards  # for logging
            causal_penalty = torch.zeros(batch_size, device=curr_outputs.device)
            dispersive_loss = torch.tensor(0.0, device=curr_outputs.device)
            # ─────────────────────────────────────────────────────────────────

        elif self.reward_mode == "bias_aligned":
            # ─── PHASE 0.8 BIAS-ALIGNED REWARD ───────────────────────────────
            # r_total(b, ans_pos) =
            #     w_corr  * 1[pred==gold]
            #   + w_bias  * -(w_biasA · h_L[b, ans_pos])      (negate: high = biased)
            #   + w_ambig * 1[gold==C AND pred==C]            (§4½.11)
            #   - kl_beta * token_kl                          (dense, per-token)
            #
            # Implementation mirrors the binary branch (sparse placement at the
            # logit position that *actually* predicted the answer letter), with
            # the additional probe-projection term computed at the SAME
            # ans_pos. The probe head is reward_heads_weight[0] (shape (1, D)
            # for a single bias_aligned direction). If multiple heads were
            # passed, we use FastRL alpha as the mixing coefficient over them.
            assert self.use_frozen_phi, "bias_aligned mode requires use_frozen_phi=True"

            gold_labels = batch.get("gold_label")
            conditions = batch.get("condition")   # list[str] or tensor of ints; optional
            if gold_labels is None:
                logger.warning("BIAS_ALIGNED: gold_label missing from batch; correctness term forced to -1.")

            batch_size = curr_outputs.shape[0]
            letter_token_ids = self._get_letter_token_ids()  # {tok_id: label_idx}

            # ── 1. Parse the predicted letter and its position offset.
            pred_label_per_sample = [-1] * batch_size
            pred_offset_per_sample = [-1] * batch_size
            for b_idx in range(batch_size):
                gen_start = gen_starts[b_idx].item()
                gen_tokens = curr_outputs[b_idx, gen_start:]
                for offset, tok in enumerate(gen_tokens[:5]):
                    tok_id = tok.item()
                    if tok_id in letter_token_ids:
                        pred_label_per_sample[b_idx] = letter_token_ids[tok_id]
                        pred_offset_per_sample[b_idx] = offset
                        break

            # ── 2. Correctness term (binary ±1, masked to parseable rows).
            corr_per_sample = torch.zeros(batch_size, device=curr_outputs.device)
            ambig_per_sample = torch.zeros(batch_size, device=curr_outputs.device)
            num_correct = 0
            num_unparsed = 0
            for b_idx in range(batch_size):
                pred = pred_label_per_sample[b_idx]
                gold = gold_labels[b_idx].item() if gold_labels is not None else -1
                if pred < 0 or gold < 0:
                    corr_per_sample[b_idx] = -1.0
                    num_unparsed += 1
                    continue
                if pred == gold:
                    corr_per_sample[b_idx] = 1.0
                    num_correct += 1
                else:
                    corr_per_sample[b_idx] = -1.0
                # Ambig-preservation bonus: when the gold answer is C
                # (gold == 2 in our label space) and the model picked C, add
                # an explicit positive reward so PPO does not destroy ambig
                # abstention to chase disambig s_d shrinkage (§4½.11).
                if gold == 2 and pred == 2:
                    ambig_per_sample[b_idx] = 1.0

            # ── 3. Bias-projection term at the answer-letter position.
            #     Note: we project against the FROZEN reference hidden state
            #     (h_ref) because the bias_aligned head was fit on un-adapted
            #     base activations. Negate to make "high score = biased" map
            #     to "low reward = bad".
            bias_w = self.reward_heads_weight  # (K, D); for bias_aligned, K=1
            if bias_w.shape[0] > 1:
                # If multiple heads passed, weight by FastRL alpha (legacy compat).
                alpha = self.fast_rl.alpha.to(bias_w.dtype)  # (K,)
                bias_dir = (alpha[:, None] * bias_w).sum(dim=0, keepdim=True)  # (1, D)
            else:
                bias_dir = bias_w  # (1, D)

            bias_per_sample = torch.zeros(batch_size, device=curr_outputs.device, dtype=h_ref.dtype)
            for b_idx in range(batch_size):
                k = pred_offset_per_sample[b_idx]
                if k < 0:
                    continue  # no letter parsed → no bias signal for this sample
                ans_pos = gen_starts_shifted[b_idx].item() + k
                if 0 <= ans_pos < h_ref.shape[1]:
                    # h_ref is shifted-aligned: row index = full-seq position
                    proj = torch.dot(bias_dir[0], h_ref[b_idx, ans_pos])
                    bias_per_sample[b_idx] = -proj  # negate: high projection = biased
            bias_per_sample = bias_per_sample.float()

            # ── 4. Compose per-sample sparse reward at ans_pos.
            total_per_sample = (
                self.correctness_coef * corr_per_sample
                + self.bias_aligned_coef * bias_per_sample
                + self.ambig_preservation_coef * ambig_per_sample
            )

            dense_rewards = torch.zeros_like(loss_mask)
            for b_idx in range(batch_size):
                k = pred_offset_per_sample[b_idx]
                ans_pos = (
                    gen_starts_shifted[b_idx].item() + (k if k >= 0 else 0)
                )
                if 0 <= ans_pos < loss_mask.shape[1] and loss_mask[b_idx, ans_pos] > 0:
                    dense_rewards[b_idx, ans_pos] = total_per_sample[b_idx]

            # ── 5. Stash per-component metrics (before KL folding).
            self._last_binary_reward_per_sample = corr_per_sample.detach()
            self._last_bias_term_per_sample = bias_per_sample.detach()
            self._last_ambig_bonus_per_sample = ambig_per_sample.detach()
            parsed = [o for o in pred_offset_per_sample if o >= 0]
            self._last_parse_success_rate = (len(parsed) / batch_size) if batch_size > 0 else 0.0
            self._last_pred_letter_offset_mean = (
                (sum(parsed) / len(parsed)) if parsed else float("nan")
            )

            # ── 6. Fold token-level KL into the dense reward (same recipe
            #     as the binary branch).
            logprob_diff = curr_logprobs - ref_logprobs.detach()
            token_kl = torch.exp(logprob_diff) - 1.0 - logprob_diff
            token_kl = token_kl.clamp(max=10.0)
            dense_rewards = (dense_rewards - self.kl_beta * token_kl) * loss_mask

            # ── 7. Zero-out optional terms for downstream metrics consistency.
            logit_reward = torch.zeros_like(logprob_diff)
            r_task_dense = dense_rewards
            causal_penalty = torch.zeros(batch_size, device=curr_outputs.device)
            dispersive_loss = torch.tensor(0.0, device=curr_outputs.device)
            # ─────────────────────────────────────────────────────────────────

        else:
            # ─── SVM REWARD MODE (original dense reward) ─────────────────────
            # Phase 0.6 D1: optionally use frozen φ (ref_penultimate) for the
            # reward projection so the heads stay grounded in the feature
            # space they were trained on. Causal penalty below still compares
            # h_active vs h_ref — we want to detect drift, not feed it back.
            h_reward = h_ref if self.use_frozen_phi else h_active
            # Project ALL token hidden states onto reward heads: (B, T, D) @ (D, K) → (B, T, K)
            r_token_k = torch.matmul(h_reward, self.reward_heads_weight.T)  # (B, T, K)
            
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
            
            # Phase 3 fix: SCALE-ONLY normalization (preserve sign / directional info).
            with torch.no_grad():
                valid_vals = r_task_dense_raw[loss_mask.bool()]
                batch_var = valid_vals.var().clamp(min=1e-8)
                if not self._r_task_initialized:
                    self._r_task_running_var = batch_var
                    self._r_task_initialized = True
                else:
                    self._r_task_running_var = (
                        self._r_task_ema_decay * self._r_task_running_var
                        + (1 - self._r_task_ema_decay) * batch_var
                    )
            r_task_std = torch.sqrt(self._r_task_running_var + 1e-8)
            r_task_dense = r_task_dense_raw / r_task_std  # sign-preserving
            
            # 5. LOGIT-GROUNDED REWARD (DISABLED in Phase 1 fix)
            logprob_diff = curr_logprobs - ref_logprobs.detach()
            logit_reward = torch.zeros_like(logprob_diff)

            # 6. TOKEN-LEVEL KL PENALTY (DISABLED in Phase 1 fix)
            token_kl = torch.exp(logprob_diff) - 1.0 - logprob_diff
            token_kl = token_kl.clamp(max=10.0)
            
            # 7. CAUSAL PENALTY (mean-token embedding drift) — METRIC ONLY
            # Phase 0.6 analysis: with D1/D2/D3 fixes the SVM reward signal is
            # real, so the policy legitimately changes e_active. The unrectified
            # causal penalty interpreted this as drift and grew to O(-100),
            # dominating r_task (~O(1)) and blowing up the value head (v_loss
            # > 4000 by step 40 of the Phase 0.6 SVM run). Dropped from
            # dense_rewards composition; will be reintroduced in a rectified
            # form in a later phase. We still COMPUTE it for visibility in
            # metrics.jsonl so the drift magnitude is observable.
            e_active_mean = (h_active.float() * mask_expanded).sum(dim=1) / valid_counts  # (B, D)
            e_ref_mean = (h_ref.float() * mask_expanded).sum(dim=1) / valid_counts  # (B, D)

            causal_penalty = compute_causal_reward_penalty(
                e_ref_mean, e_active_mean,
                lambda_causal=self.lambda_causal,
                delta_margin=self.delta_margin,
            )

            # 8. COMPOSE DENSE REWARD (Phase 0.6: causal_per_token dropped;
            # KL & logit_reward already removed in Phase 1).
            dense_rewards = r_task_dense * loss_mask

            # 12. DISPERSIVE REGULARIZATION (mean-pooled embeddings, O(B^2))
            e_active_mean_disp = (h_active.float() * mask_expanded).sum(dim=1) / valid_counts
            dispersive_loss = compute_dispersive_loss(e_active_mean_disp)
            # ─────────────────────────────────────────────────────────────────
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
        
        # 10. PPO SURROGATE LOSS (Phase 1 fix: ratio uses π_old, not π_ref)
        ratio = torch.exp(curr_logprobs - old_logprobs)  # old_logprobs is already detached
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_range)
        
        seq_pg_loss = (torch.max(pg_loss1, pg_loss2) * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)
        pg_loss = seq_pg_loss.mean()
        
        # 11. VALUE LOSS (optionally clipped per Schulman PPO)
        if self.value_clip_range > 0.0:
            # Snapshot of v_old at rollout time. In single-epoch on-policy PPO
            # the value head has not been updated yet within this batch, so the
            # detached current prediction IS v_old. The clip then prevents the
            # value head from chasing transient reward spikes inside the
            # current minibatch, which is the standard precursor to policy
            # collapse (cf. CRITICAL_REVIEW_v4.md, B2/C1).
            v_old = curr_values_aligned.detach()
            v_clipped = v_old + (curr_values_aligned - v_old).clamp(
                -self.value_clip_range, self.value_clip_range
            )
            v_loss_unclipped = (curr_values_aligned - returns) ** 2
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped) * loss_mask
        else:
            v_loss = 0.5 * ((curr_values_aligned - returns) ** 2) * loss_mask
        v_loss = v_loss.sum(dim=1) / loss_mask.sum(dim=1)
        v_loss = v_loss.mean()
        
        # 13. TOTAL LOSS
        loss = pg_loss + self.vf_coef * v_loss + self.lambda_dispersive * dispersive_loss
        
        # Backprop
        self.accelerator.backward(loss)

        # SANITY: on the very first backward call, verify that gradients
        # actually reached the trainable (LoRA) parameters. Fail loudly
        # rather than waiting for a 23-minute training run to discover
        # the policy never moved.
        if not getattr(self, "_grad_flow_checked", False):
            n_trainable = 0
            n_with_grad = 0
            sample_zero_param = None
            for name, p in self.policy.named_parameters():
                if p.requires_grad:
                    n_trainable += 1
                    if p.grad is not None and p.grad.abs().sum().item() > 0:
                        n_with_grad += 1
                    elif sample_zero_param is None:
                        sample_zero_param = name
            logger.info(
                f"GRAD FLOW CHECK: {n_with_grad}/{n_trainable} trainable policy "
                f"params received non-zero gradients on first backward. "
                f"loss={loss.item():.4f}, pg_loss={pg_loss.item():.4f}"
            )
            if n_with_grad == 0 and n_trainable > 0:
                raise RuntimeError(
                    f"Zero gradient flow to ALL {n_trainable} trainable policy "
                    f"params. Example param with no grad: {sample_zero_param}. "
                    f"Check: (1) use_reentrant=False on gradient checkpointing, "
                    f"(2) ppo_controller.policy is the prepared model, "
                    f"(3) no torch.no_grad() wraps curr_logprobs forward, "
                    f"(4) curr_logprobs is connected to loss (not detached)."
                )
            self._grad_flow_checked = True

        # Phase 3 hygiene: gradient clipping (only on accumulation boundaries),
        # but capture pre-clip gradient norms on EVERY micro-batch so the
        # metric is meaningful in the per-step log. (Previously the norm was
        # only read inside `if sync_gradients:` while metrics.jsonl writes
        # every micro-batch — so 3/4 logged rows showed 0.0 and the
        # remaining row could also show 0.0 if the optimizer step had
        # already cleared the grads. This was the cause of the "pg_gn=0.0
        # across all 250 steps" telemetry blind-spot in Phase 0.)
        def _total_grad_norm(params):
            grads = [p.grad.detach() for p in params if p.grad is not None]
            if not grads:
                return 0.0
            norms = torch.stack([g.norm(2) for g in grads])
            return float(torch.norm(norms, 2).item())

        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        policy_grad_norm = _total_grad_norm(policy_params)
        value_grad_norm = _total_grad_norm(list(self.value_head.parameters()))

        if self.accelerator.sync_gradients and self.max_grad_norm is not None:
            # Only the sync-boundary call actually performs the clip in-place;
            # the per-step norm above is already captured pre-clip.
            self.accelerator.clip_grad_norm_(policy_params, self.max_grad_norm)
            self.accelerator.clip_grad_norm_(
                self.value_head.parameters(), self.max_grad_norm,
            )

        optimizer_policy.step()
        optimizer_value.step()
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()

        # ── Adaptive KL controller (Schulman / Ouyang 2022) ─────────────────
        # After the optimizer step we observe the *post-update* KL. If it is
        # above the per-token target we tighten kl_beta; if below, we loosen.
        # This is the standard recipe and removes the manual kl_beta tuning
        # that was the proximate cause of the ep1-end collapse documented in
        # the research report (§2.1).
        kl_observed = (
            token_kl[loss_mask.bool()].mean().item() if loss_mask.any() else 0.0
        )
        if self.target_kl > 0.0 and self.accelerator.sync_gradients:
            err = kl_observed / self.target_kl - 1.0
            step_factor = float(
                max(0.5, min(2.0, 1.0 + self.kl_adapt_rate * err))
            )
            new_beta = self.kl_beta * step_factor
            new_beta = max(self.kl_beta_min, min(self.kl_beta_max, new_beta))
            self.kl_beta = new_beta

        # Honest signal-magnitude metric: how big are the advantages at the
        # reward-bearing positions (i.e. the answer-letter logit positions)?
        # This is what the LoRA gradient is actually proportional to.
        adv_abs_mean = (
            advantages[loss_mask.bool()].abs().mean().item() if loss_mask.any() else 0.0
        )

        metrics = {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "dispersive_loss": dispersive_loss.item(),
            "causal_penalty": causal_penalty.mean().item(),
            "reward_dense_mean": dense_rewards.sum(dim=1).mean().item(),
            "reward_task": r_task_dense[loss_mask.bool()].mean().item() if loss_mask.any() else 0.0,
            # Phase 0.6 smoke-test gate: per-token reward_task variance across the
            # valid (generated) positions in this batch. std > 0.01 confirms the
            # DRM heads produce a non-degenerate token-level reward signal.
            "reward_task_std": r_task_dense[loss_mask.bool()].std().item() if loss_mask.sum() > 1 else 0.0,
            "logit_reward": logit_reward[loss_mask.bool()].mean().item() if loss_mask.any() else 0.0,
            "kl": token_kl[loss_mask.bool()].mean().item() if loss_mask.any() else 0.0,
            "mean_abs_logprob_diff": logprob_diff[loss_mask.bool()].abs().mean().item() if loss_mask.any() else 0.0,
            "mean_abs_old_curr_diff": (curr_logprobs - old_logprobs)[loss_mask.bool()].abs().mean().item() if loss_mask.any() else 0.0,
            "ratio_mean": ratio[loss_mask.bool()].mean().item() if loss_mask.any() else 1.0,
            "adv_abs_mean": adv_abs_mean,
            "policy_grad_norm": policy_grad_norm,
            "value_grad_norm": value_grad_norm,
            "kl_beta": float(self.kl_beta),
        }
        if self.reward_mode == "binary":
            # Track accuracy from the *pre-KL* per-sample reward (dense_rewards now
            # has the KL penalty folded in, so its sign no longer tracks correctness).
            bps = getattr(self, "_last_binary_reward_per_sample", None)
            if bps is not None:
                accuracy = (bps > 0).float().mean().item()
                metrics["binary_accuracy"] = accuracy
                metrics["reward_binary_mean"] = bps.mean().item()
            # Parse diagnostics: did the model actually emit a letter token in
            # the first 5 generated positions, and at what offset?
            psr = getattr(self, "_last_parse_success_rate", None)
            if psr is not None:
                metrics["parse_success_rate"] = psr
            plom = getattr(self, "_last_pred_letter_offset_mean", None)
            if plom is not None and plom == plom:  # nan-safe
                metrics["pred_letter_offset_mean"] = plom
        elif self.reward_mode == "bias_aligned":
            bps = getattr(self, "_last_binary_reward_per_sample", None)
            if bps is not None:
                metrics["binary_accuracy"] = (bps > 0).float().mean().item()
                metrics["reward_correctness_mean"] = bps.mean().item()
            bias_t = getattr(self, "_last_bias_term_per_sample", None)
            if bias_t is not None:
                metrics["reward_bias_aligned_mean"] = bias_t.float().mean().item()
                metrics["reward_bias_aligned_abs_mean"] = bias_t.float().abs().mean().item()
            ambig_b = getattr(self, "_last_ambig_bonus_per_sample", None)
            if ambig_b is not None:
                metrics["ambig_preserve_rate"] = ambig_b.float().mean().item()
            psr = getattr(self, "_last_parse_success_rate", None)
            if psr is not None:
                metrics["parse_success_rate"] = psr
            plom = getattr(self, "_last_pred_letter_offset_mean", None)
            if plom is not None and plom == plom:
                metrics["pred_letter_offset_mean"] = plom
            metrics["reward_head_layer"] = self.reward_head_layer
        return metrics
