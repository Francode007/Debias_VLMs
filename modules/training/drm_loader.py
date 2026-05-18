"""
DRM (Debiased Reward Model) Head Loading Utilities.

Handles loading Phase 1 PCA components (.pth files) as combined reward
head tensors, and the utility for reloading a saved debiased adapter.
"""
import glob
import logging
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

logger = logging.getLogger(__name__)


def load_pca_components(
    heads_dir: str, num_heads: int, device: torch.device
) -> torch.Tensor:
    """
    Load Phase 1 PCA score head .pth files and stack into a single weight matrix.

    Args:
        heads_dir:  Directory containing component*.pth files.
        num_heads:  Maximum number of components to load.
        device:     Target torch device.

    Returns:
        Tensor of shape (num_heads, hidden_dim) in bfloat16.

    Raises:
        FileNotFoundError: If no .pth files are found in heads_dir.
    """
    pth_files = glob.glob(os.path.join(heads_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {heads_dir}.")

    def _extract_index(f: str) -> int:
        m = re.search(r"component(\d+)\.pth$", f)
        return int(m.group(1)) if m else 0

    pth_files = sorted(pth_files, key=_extract_index)[:num_heads]

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
