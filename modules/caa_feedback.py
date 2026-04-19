import torch

def compute_caa_weight(rewards_init: torch.Tensor, rewards_curr: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Phase 3: Causality-Aware Alignment (CAA).
    Calculate an instrumented causal scalar weight for each sample in the PPO batch.
    
    The function calculates the orthogonal deviation of the active policy from the 
    reference policy inside the phase 1 feature space.
    
    Args:
        rewards_init: Tensor of shape (batch_size, num_heads) - r_init^(k) 
                      computed using the frozen base reference policy.
        rewards_curr: Tensor of shape (batch_size, num_heads) - r_curr^(k)
                      computed using the active training PPO policy.
        epsilon: Small scalar to avoid division by zero during Min-Max normalization.
        
    Returns:
        w_hat: Tensor of shape (batch_size,) 
               The min-max normalized L2 norm of the causal differences.
    """
    
    # 1. Delta rewards (Absolute causal difference across all axes)
    # Shape: (batch_size, num_heads)
    delta_r_vec = torch.abs(rewards_init - rewards_curr)
    
    # 2. L2 Norm per sample
    # Shape: (batch_size,)
    w_feedback = torch.linalg.norm(delta_r_vec, ord=2, dim=-1)
    
    # 3. Batch-wise Min-Max Normalization
    # Note: If batch size is 1, min == max, w_hat will be completely flat/zeroed out.
    # We should handle it properly if w_max == w_min. 
    w_min = torch.min(w_feedback)
    w_max = torch.max(w_feedback)
    
    w_hat = (w_feedback - w_min) / (w_max - w_min + epsilon)
    
    # If the batch difference min/max are virtually identical, default w_hat to 1.0
    # to prevent erasing the loss.
    if torch.allclose(w_min, w_max, atol=1e-6):
        w_hat = torch.ones_like(w_feedback)
        
    return w_hat.detach()
