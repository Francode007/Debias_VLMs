import torch
import torch.nn as nn
import torch.nn.functional as F


class FastRLNode:
    """
    Fast-RL Node with running z-score normalization and entropy-regularized
    deficit-based simplex updates. Prevents one-hot collapse and ensures
    KL penalty remains meaningful relative to normalized reward scales.
    """
    def __init__(self, num_heads: int, tau: float = 1.0, ema_decay: float = 0.99, device: str = "cpu"):
        """
        Args:
            num_heads: Number of orthogonal reward heads (K).
            tau: Temperature for entropy-regularized softmax (prevents collapse).
                 Higher = more uniform, lower = more focused on worst heads.
            ema_decay: Exponential moving average decay for running stats.
            device: Tensor device.
        """
        self.num_heads = num_heads
        self.tau = tau
        self.ema_decay = ema_decay
        self.device = device

        # Uniform initial weights on the simplex
        self.alpha = torch.ones(num_heads, device=device, dtype=torch.float32) / num_heads

        # Running statistics for z-score normalization (per head)
        self.running_mean = torch.zeros(num_heads, device=device, dtype=torch.float32)
        self.running_var = torch.ones(num_heads, device=device, dtype=torch.float32)
        self.initialized = False

    def _update_running_stats(self, r_k_mean: torch.Tensor):
        """Update running mean/variance via EMA."""
        if not self.initialized:
            self.running_mean = r_k_mean.clone()
            self.running_var = torch.ones_like(r_k_mean)
            self.initialized = True
        else:
            self.running_mean = self.ema_decay * self.running_mean + (1 - self.ema_decay) * r_k_mean
            # Variance from squared deviation of batch mean from running mean
            self.running_var = self.ema_decay * self.running_var + (1 - self.ema_decay) * (r_k_mean - self.running_mean) ** 2

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Z-score normalize rewards using running statistics. Shape: (batch, K) -> (batch, K)."""
        std = torch.sqrt(self.running_var + 1e-8)
        return (rewards - self.running_mean.unsqueeze(0)) / std.unsqueeze(0)

    def update(self, rewards_curr: torch.Tensor) -> torch.Tensor:
        """
        Update alpha via deficit-based entropy-regularized softmax,
        then compute composite reward from normalized scores.

        Args:
            rewards_curr: (batch_size, num_heads) raw reward projections.
        Returns:
            composite_reward: (batch_size,) weighted normalized reward.
        """
        rewards_curr = rewards_curr.float()
        r_k_mean = rewards_curr.mean(dim=0)  # (num_heads,)

        # Update running statistics
        self._update_running_stats(r_k_mean)

        # Z-score normalize the batch rewards
        rewards_norm = self.normalize(rewards_curr)  # (batch, K)

        # Deficit-based alpha update:
        # Focus on heads that are UNDERPERFORMING (below zero after normalization).
        # deficit_k = max(0, -r_k_normalized_mean) — higher deficit = more weight needed
        r_norm_mean = rewards_norm.mean(dim=0)  # (K,)
        deficit = torch.clamp(-r_norm_mean, min=0.0)

        # Entropy-regularized softmax with temperature tau
        # High tau → uniform, low tau → focused on worst-performing heads
        self.alpha = F.softmax(deficit / self.tau, dim=-1).detach()

        # Composite reward from NORMALIZED rewards (centered ~0, std ~1)
        composite_reward = torch.matmul(rewards_norm, self.alpha)

        return composite_reward
