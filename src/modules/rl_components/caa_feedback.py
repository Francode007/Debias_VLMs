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
    Dispersive regularization loss that prevents representation collapse by
    penalizing pairwise proximity of embeddings in the batch.

    Formula:
        L_disp = -1/(B*(B-1)) * sum_{i!=j} log(||h_i - h_j||_2 + epsilon)

    This forces embeddings of different inputs to remain spread apart,
    preserving full-rank representational capacity.

    Args:
        hidden_states: (batch_size, hidden_dim) — penultimate layer embeddings.
        sigma: Scale parameter (unused in log formulation, kept for API stability).

    Returns:
        Scalar dispersive loss (to be minimized — drives embeddings apart).
    """
    batch_size = hidden_states.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

    # Pairwise L2 distances: (B, B)
    diffs = hidden_states.unsqueeze(0) - hidden_states.unsqueeze(1)  # (B, B, D)
    dists = torch.linalg.norm(diffs, ord=2, dim=-1)  # (B, B)

    # Exclude diagonal (self-pairs)
    mask = ~torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
    pairwise_dists = dists[mask]  # (B*(B-1),)

    # Log-distance repulsion: minimize this → push embeddings apart
    log_dists = torch.log(pairwise_dists + 1e-8)

    # Negative mean log-distance: lower distance → higher loss
    dispersive_loss = -log_dists.mean()

    return dispersive_loss
