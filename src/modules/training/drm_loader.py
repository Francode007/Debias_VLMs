"""
DRM (Debiased Reward Model) Head Loading Utilities.

Handles loading Phase 1 PCA components (.pth files) as combined reward
head tensors, and the utility for reloading a saved debiased adapter.
"""
import glob
import json
import logging
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

logger = logging.getLogger(__name__)


def load_pca_components(
    heads_dir: str,
    num_heads: int,
    device: torch.device,
    kept_heads_filter: str = None,
) -> torch.Tensor:
    """
    Load Phase 1 PCA / SVM score head .pth files and stack into a single weight matrix.

    Args:
        heads_dir:          Directory containing component*.pth files.
        num_heads:          Maximum number of components to consider (taken from
                            sorted-by-index file list before any filtering).
        device:             Target torch device.
        kept_heads_filter:  Optional path to kept_heads.json produced by
                            evaluate_drm_heads.py (Phase 0.6 D2). When given,
                            only heads whose original index is in
                            kept_indices are returned. The order is preserved
                            from kept_indices.

    Returns:
        Tensor of shape (kept_count, hidden_dim) in bfloat16.

    Raises:
        FileNotFoundError: If no .pth files are found in heads_dir.
        ValueError:        If kept_heads_filter is given but kept_indices is empty.
    """
    pth_files = glob.glob(os.path.join(heads_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {heads_dir}.")

    def _extract_index(f: str) -> int:
        m = re.search(r"component(\d+)\.pth$", f)
        return int(m.group(1)) if m else 0

    pth_files = sorted(pth_files, key=_extract_index)[:num_heads]

    # Apply kept_heads filter if provided.
    if kept_heads_filter:
        with open(kept_heads_filter, "r") as f:
            payload = json.load(f)
        kept_indices = payload.get("kept_indices", [])
        if not kept_indices:
            raise ValueError(
                f"kept_heads_filter {kept_heads_filter} has empty kept_indices; "
                f"refusing to load zero reward heads."
            )
        index_to_path = {_extract_index(p): p for p in pth_files}
        missing = [i for i in kept_indices if i not in index_to_path]
        if missing:
            raise ValueError(
                f"kept_heads_filter references indices not present in {heads_dir}: {missing}"
            )
        pth_files = [index_to_path[i] for i in kept_indices]
        logger.info(
            f"Applied kept_heads filter ({kept_heads_filter}): "
            f"{len(pth_files)}/{payload.get('num_heads_total', '?')} heads retained"
        )

    weights = []
    for p in pth_files:
        state = torch.load(p, map_location="cpu")
        w = state.get("weight", state)
        if isinstance(w, dict):
            w = w.get("weight")
        weights.append(w)  # each (1, hidden_dim)

    combined = torch.cat(weights, dim=0).to(device, dtype=torch.bfloat16)
    logger.info(f"Loaded {combined.shape[0]} DRM heads of shape {combined.shape}")
    return combined


def load_debiased_model(
    base_model_path: str, peft_adapter_path: str, device: str = "cuda"
) -> PeftModel:
    """
    Utility: reload a frozen base model and apply a saved LoRA adapter.

    Args:
        base_model_path:    HF name or local path to the base model.
        peft_adapter_path:  Path to the saved PEFT adapter directory.
        device:             Device string (e.g. "cuda").

    Returns:
        PeftModel with the adapter merged on top of the frozen base.
    """
    logger.info(f"Loading base model: {base_model_path}")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
        trust_remote_code=True,
    )
    logger.info(f"Applying LoRA adapter: {peft_adapter_path}")
    model = PeftModel.from_pretrained(base_model, peft_adapter_path)
    return model
