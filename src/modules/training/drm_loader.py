"""
DRM (Debiased Reward Model) Head Loading Utilities.

Handles loading Phase 1 PCA components (.pth files) as combined reward
head tensors, the Phase 0.9 multi-layer ensemble probe bundle, and the
utility for reloading a saved debiased adapter.
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


# ─── Phase 0.9 R3 — multi-layer ensemble probe bundle ───────────────────────
ENSEMBLE_METADATA_FILENAME = "ensemble_metadata.json"
ENSEMBLE_REQUIRED_METADATA_KEYS = (
    "schema_version", "layers", "pool",
    "per_layer_mu", "per_layer_sigma", "hidden_dim",
)
ENSEMBLE_SUPPORTED_POOLS = ("zmean", "mean", "max")


def load_ensemble_probe_bundle(
    bundle_dir: str,
    device: torch.device,
    overrides: dict = None,
) -> tuple:
    """
    Load a Phase 0.9 multi-layer ensemble probe bundle.

    Bundle layout (built by scripts/phase09_build_ensemble_bundle.py):
        bundle_dir/
            L{N1}.pth, L{N2}.pth, ...   # each = {"weight": Tensor of shape (1, D)}
            ensemble_metadata.json      # {schema_version, layers, pool,
                                        #   per_layer_mu, per_layer_sigma,
                                        #   hidden_dim, ...}

    Args:
        bundle_dir:  Path to the bundle directory.
        device:      Target torch device for the loaded weight tensors.
        overrides:   Optional dict of metadata overrides. Supported keys:
                       - 'layers': list[int] subset of bundle's layers
                       - 'pool':   str ∈ {'zmean', 'mean', 'max'}

    Returns:
        layer_to_weight: dict[int, Tensor]
            Maps layer index → unit-normed weight tensor of shape (1, D)
            in bfloat16 on the target device.
        metadata: dict
            Parsed ensemble_metadata.json with effective `layers`, `pool`,
            and the *aligned* `per_layer_mu` / `per_layer_sigma` (after
            applying any `overrides['layers']` filter).

    Raises:
        FileNotFoundError: bundle_dir or metadata file missing.
        ValueError:        invalid metadata, mismatched layer set,
                           or unsupported pool.
    """
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(f"ensemble bundle dir not found: {bundle_dir}")

    metadata_path = os.path.join(bundle_dir, ENSEMBLE_METADATA_FILENAME)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"missing {ENSEMBLE_METADATA_FILENAME} in {bundle_dir}"
        )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    missing = [k for k in ENSEMBLE_REQUIRED_METADATA_KEYS if k not in metadata]
    if missing:
        raise ValueError(
            f"ensemble metadata in {metadata_path} missing keys: {missing}"
        )
    if metadata["schema_version"] != "1.0":
        raise ValueError(
            f"unsupported ensemble bundle schema_version={metadata['schema_version']}"
        )

    bundle_layers: list = list(metadata["layers"])
    bundle_mu: list = list(metadata["per_layer_mu"])
    bundle_sigma: list = list(metadata["per_layer_sigma"])
    bundle_pool: str = str(metadata["pool"])
    if len(bundle_layers) != len(bundle_mu) or len(bundle_layers) != len(bundle_sigma):
        raise ValueError(
            f"ensemble metadata length mismatch in {metadata_path}: "
            f"layers={len(bundle_layers)} mu={len(bundle_mu)} sigma={len(bundle_sigma)}"
        )

    # Apply optional overrides.
    overrides = overrides or {}
    if "layers" in overrides and overrides["layers"] is not None:
        requested = list(overrides["layers"])
        missing_in_bundle = [L for L in requested if L not in bundle_layers]
        if missing_in_bundle:
            raise ValueError(
                f"requested layers {missing_in_bundle} not present in bundle "
                f"(bundle layers = {bundle_layers})"
            )
        # Re-align mu/sigma to the requested layer order.
        layer_to_mu = dict(zip(bundle_layers, bundle_mu))
        layer_to_sigma = dict(zip(bundle_layers, bundle_sigma))
        eff_layers = requested
        eff_mu = [layer_to_mu[L] for L in eff_layers]
        eff_sigma = [layer_to_sigma[L] for L in eff_layers]
    else:
        eff_layers = bundle_layers
        eff_mu = bundle_mu
        eff_sigma = bundle_sigma

    eff_pool = overrides.get("pool") or bundle_pool
    if eff_pool not in ENSEMBLE_SUPPORTED_POOLS:
        raise ValueError(
            f"unsupported ensemble pool={eff_pool!r}; "
            f"supported = {ENSEMBLE_SUPPORTED_POOLS}"
        )

    hidden_dim = int(metadata["hidden_dim"])
    layer_to_weight: dict = {}
    for L in eff_layers:
        pth = os.path.join(bundle_dir, f"L{L}.pth")
        if not os.path.exists(pth):
            raise FileNotFoundError(f"ensemble bundle missing weight file: {pth}")
        state = torch.load(pth, map_location="cpu")
        w = state.get("weight", state) if isinstance(state, dict) else state
        if w.shape != (1, hidden_dim):
            raise ValueError(
                f"L{L}.pth weight has shape {tuple(w.shape)}, "
                f"expected (1, {hidden_dim})"
            )
        # Re-normalise defensively (bundle stores unit-normed; bf16 round-trip
        # can introduce ~1e-3 error).
        nrm = w.norm()
        if nrm <= 0:
            raise ValueError(f"L{L}.pth weight has zero norm")
        w = (w / nrm).to(device, dtype=torch.bfloat16)
        layer_to_weight[int(L)] = w

    # Bake the effective values back so downstream consumers see one canonical view.
    metadata["effective_layers"] = list(eff_layers)
    metadata["effective_pool"] = eff_pool
    metadata["effective_per_layer_mu"] = list(eff_mu)
    metadata["effective_per_layer_sigma"] = list(eff_sigma)

    logger.info(
        f"Loaded ensemble probe bundle from {bundle_dir} "
        f"layers={eff_layers}  pool={eff_pool}  "
        f"calibration={metadata.get('calibration_dataset', '?')} "
        f"(n={metadata.get('calibration_n', '?')})"
    )
    return layer_to_weight, metadata


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
                            Pass 0 (or any value <= 0) to auto-detect and load
                            ALL component files present in heads_dir.
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

    pth_files = sorted(pth_files, key=_extract_index)
    if num_heads and num_heads > 0:
        pth_files = pth_files[:num_heads]
    # else: auto-detect — load every component file present.

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
