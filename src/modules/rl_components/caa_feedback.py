import torch


def compute_causal_reward_penalty(
    e_init: torch.Tensor,
    e_curr: torch.Tensor,
    lambda_causal: float = 0.5,
    delta_margin: float = 1.0,
) -> torch.Tensor:
    """
    Compute a causal alignment penalty to be ADDED to the reward signal
    (not used as a loss multiplier).

    Penalizes embedding drift beyond a tolerance margin, integrated into the
    reward function so that GAE naturally absorbs the signal without violating
    PPO trust region geometry.

    Formula:
        penalty_i = -lambda_causal * max(0, ||e_curr_i - e_init_i||_2 - delta_margin)

    Args:
        e_init: (batch_size, hidden_dim) — reference policy embeddings.
        e_curr: (batch_size, hidden_dim) — active policy embeddings.
        lambda_causal: Strength of the causal deviation penalty.
        delta_margin: Tolerance margin — small embedding shifts are not penalized.

    Returns:
        causal_penalty: (batch_size,) — negative penalty to add to reward.
            Zero when drift is within margin; increasingly negative beyond it.
    """
    # L2 distance in embedding space
    drift = torch.linalg.norm(e_curr - e_init, ord=2, dim=-1)  # (batch,)

    # Hinge penalty: only activate beyond margin
    excess_drift = torch.clamp(drift - delta_margin, min=0.0)

    # Return as negative reward contribution
    causal_penalty = -lambda_causal * excess_drift

    return causal_penalty.detach()


def compute_dispersive_loss(hidden_states: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Bounded dispersive regularization (Phase 3 fix).

    Replaces the previous `-mean(log(d + eps))` form, which is unbounded as
    d -> 0 and produced unstable gradients during training.

    New formulation (Gaussian RBF in [0, 1]):

        L_disp = mean_{i != j}  exp(-||h_i - h_j||_2^2 / (2 * sigma^2))

    - Always in [0, 1]; gradients are bounded.
    - Minimizing this still drives embeddings apart, but with diminishing
      pressure as they separate (rather than infinite pressure when close).

    Args:
        hidden_states: (B, D)
        sigma:         RBF bandwidth.

    Returns:
        Scalar loss in [0, 1].
    """
    batch_size = hidden_states.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

    diffs = hidden_states.unsqueeze(0) - hidden_states.unsqueeze(1)  # (B, B, D)
    sq_dists = (diffs * diffs).sum(dim=-1)                            # (B, B)

    mask = ~torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
    pairwise_sq = sq_dists[mask]
    return torch.exp(-pairwise_sq / (2.0 * sigma * sigma + 1e-8)).mean()
