import torch
import torch.nn as nn
import torch.nn.functional as F

class FastRLNode:
    """
    Fast-RL Node that dynamically balances Phase 1 multi-dimensional 
    orthogonal rewards during PPO training.
    """
    def __init__(self, num_heads: int, strategy: str = "exponentiated", eta: float = 0.01, device: str = "cpu"):
        """
        Args:
            num_heads: The number of orthogonal reward heads (K).
            strategy: "exponentiated", "projected", or "adam".
            eta: Learning rate for the Fast-RL update.
            device: Tensor device.
        """
        self.num_heads = num_heads
        self.strategy = strategy.lower()
        self.eta = eta
        self.device = device
        
        # Initialize Uniform Weights
        self.alpha = torch.ones(num_heads, device=device, dtype=torch.float32) / num_heads
        
        if self.strategy == "adam":
            # For Adam-style, keep logits and momentum buffers over those logits
            self.logits = torch.zeros(num_heads, device=device, dtype=torch.float32)
            self.m = torch.zeros_like(self.logits)
            self.v = torch.zeros_like(self.logits)
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0
            # Alpha is just softmax(logits), initially uniform since logits=0.

    def update(self, rewards_curr: torch.Tensor) -> torch.Tensor:
        """
        Update the weights alpha based on the current policy rewards matrix.
        
        Args:
            rewards_curr: Tensor of shape (batch_size, num_heads) - The rewards r_k
        Returns:
            composite_reward: Tensor of shape (batch_size,) - The weighted sum R_curr
        """
        batch_size = rewards_curr.shape[0]
        # Ensure float32 for numerical stability (rewards may arrive in bf16)
        rewards_curr = rewards_curr.float()
        # Average the rewards across the batch to get the expected gradient performance
        r_k_mean = rewards_curr.mean(dim=0)  # Shape (num_heads,)
        
        # 1. Update Strategy
        if self.strategy == "exponentiated":
            # Multiplicative Weights (Exponentiated Gradient) on Simplex
            # alpha(t+1) propto alpha(t) * exp(eta * r)
            # Implemented safely in log space to avoid overflow, then softmax.
            log_alpha = torch.log(self.alpha + 1e-12)
            self.alpha = F.softmax(log_alpha + self.eta * r_k_mean, dim=-1)

        elif self.strategy == "projected":
            # Projected Gradient Ascent on the probability simplex
            new_alpha = self.alpha + self.eta * r_k_mean
            self.alpha = self._project_simplex(new_alpha)
            
        elif self.strategy == "adam":
            # Adam-style update on Logits, then map through softmax.
            # Here, the 'gradient' to ascend is r_k_mean.
            self.t += 1
            grad = -r_k_mean  # Adam minimizes, we want to maximize r_k -> substitute -r_k
            
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            
            self.logits = self.logits - self.eta * m_hat / (torch.sqrt(v_hat) + self.epsilon)
            self.alpha = F.softmax(self.logits, dim=-1)
        else:
            raise ValueError(f"Unknown Fast-RL strategy: {self.strategy}")
            
        # Ensure it's detached from graph for the RL loop and properly typed
        self.alpha = self.alpha.detach()
        
        # 2. Composite Reward Calculation
        # rewards_curr: (Batch, K)
        # alpha: (K,)
        # composite_reward: (Batch,)
        composite_reward = torch.matmul(rewards_curr, self.alpha)
        
        return composite_reward

    def _project_simplex(self, v: torch.Tensor) -> torch.Tensor:
        """
        Projects a vector 'v' onto the probability simplex: sum(x) = 1, x >= 0.
        O(N log N) projection algorithm.
        """
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0) - 1.0
        indices = torch.arange(1, len(v) + 1, device=v.device, dtype=v.dtype)
        cond = v_sorted - cssv / indices > 0
        rho = indices[cond][-1].int().item()
        theta = cssv[rho - 1] / float(rho)
        w = torch.clamp(v - theta, min=0.0)
        return w
