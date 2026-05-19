"""
Multiple DRM heads for scoring embeddings.

Loads PCA component .pth files (from generate_drm_heads.py) and applies them
to (chosen, rejected) embeddings to produce per-head rewards. Used by
evaluate_drm_heads.py for SB-Bench evaluation.
"""

import os
import re
import glob
from typing import Optional
import torch
import torch.nn as nn


def extract_component_number(filepath: str) -> int:
    """Extract component index from filename like sb_bench-PCA-component42.pth"""
    basename = os.path.basename(filepath)
    match = re.search(r"component(\d+)\.pth$", basename)
    if match:
        return int(match.group(1))
    return 0


class MultipleHead(nn.Module):
    """
    Multi-head reward module: one linear layer per PCA component.
    Forward takes embeddings (N, 2, hidden_size) and returns
    (rewards_chosen, rewards_rejected) each of shape (N, num_heads).
    """

    def __init__(
        self,
        hidden_size: int,
        score_head_weight: str,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_heads: Optional[int] = None,
    ):
        super().__init__()
        self.device = device
        self.score_head = nn.Linear(hidden_size, 1, dtype=dtype, bias=False)

        if score_head_weight and os.path.isdir(score_head_weight):
            self._load_ckpts(score_head_weight, num_heads)
        else:
            raise ValueError("score_head_weight must be a directory of .pth files")

    def _load_ckpts(self, weight_dir: str, num_heads: Optional[int]):
        pth_files = glob.glob(os.path.join(weight_dir, "*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files in {weight_dir}")
        pth_files = sorted(pth_files, key=extract_component_number)
        if num_heads is not None:
            pth_files = pth_files[:num_heads]
        weights = []
        for p in pth_files:
            state = torch.load(p, map_location="cpu", weights_only=True)
            w = state.get("weight", state)
            if isinstance(w, dict):
                w = w.get("weight")
            weights.append(w)
        combined = torch.cat(weights, dim=0)
        self.score_head = nn.Linear(
            combined.shape[1], combined.shape[0], dtype=combined.dtype, bias=False
        )
        self.score_head.weight.data = combined.to(self.device)

    def forward(self, embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        embs: (N, 2, hidden_size) — [chosen_emb, rejected_emb] per sample.
        Returns:
            rewards_chosen: (N, num_heads)
            rewards_rejected: (N, num_heads)
        """
        n = embs.shape[0]
        chosen = embs[:, 0, :]
        rejected = embs[:, 1, :]
        stacked = torch.cat([chosen, rejected], dim=0).to(
            device=self.device, dtype=self.score_head.weight.dtype
        )
        scores = self.score_head(stacked)
        rewards_chosen = scores[:n]
        rewards_rejected = scores[n:]
        return rewards_chosen, rewards_rejected
