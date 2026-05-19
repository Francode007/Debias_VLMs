"""
Train/Test Split Utility

Generates and persists an 80/20 train/test split on orig_index (the row index
into the SB-Bench parquet before expanding to preference pairs). Both halves of
a preference pair (data_index = orig_index*2 + 0/1) always stay in the same split.

The split is deterministic (seed=42) and saved as JSON so that all downstream
stages (embedding extraction, DRM generation, RL training, evaluation) share
the exact same partition.
"""

import os
import json
import glob
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_SEED = 42
TRAIN_RATIO = 0.8


def get_split_path(data_path: str) -> str:
    """Return the canonical path for the split indices JSON next to the data."""
    return os.path.join(os.path.dirname(data_path.rstrip("/")), "split_indices.json")


def generate_split(data_path: str, output_path: str = None, seed: int = SPLIT_SEED, train_ratio: float = TRAIN_RATIO) -> dict:
    """
    Generate an 80/20 split over original sample indices and persist to disk.

    Args:
        data_path: Directory containing parquet files.
        output_path: Where to save the JSON. Defaults to sibling of data_path.
        seed: Random seed for reproducibility.
        train_ratio: Fraction for training (default 0.8).

    Returns:
        dict with keys 'train_indices', 'test_indices', 'total', 'seed', 'train_ratio'.
    """
    if output_path is None:
        output_path = get_split_path(data_path)

    parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")

    total_rows = 0
    for f in parquet_files:
        df = pd.read_parquet(f, engine="fastparquet")
        total_rows += len(df)

    indices = np.arange(total_rows)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_point = int(len(indices) * train_ratio)
    train_indices = sorted(indices[:split_point].tolist())
    test_indices = sorted(indices[split_point:].tolist())

    split_info = {
        "seed": seed,
        "train_ratio": train_ratio,
        "total": total_rows,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_info, f)

    logger.info(f"Split generated: {len(train_indices)} train / {len(test_indices)} test (total={total_rows}, seed={seed})")
    logger.info(f"Saved to {output_path}")
    return split_info


def load_split(split_path: str) -> dict:
    """Load a previously saved split. Returns dict with 'train_indices' and 'test_indices'."""
    with open(split_path, "r") as f:
        return json.load(f)


def get_or_create_split(data_path: str, split_path: str = None) -> dict:
    """Load existing split or generate a new one."""
    if split_path is None:
        split_path = get_split_path(data_path)

    if os.path.exists(split_path):
        logger.info(f"Loading existing split from {split_path}")
        return load_split(split_path)

    logger.info(f"No split found at {split_path}, generating new split...")
    return generate_split(data_path, split_path)


def orig_indices_to_data_indices(orig_indices: list) -> list:
    """Convert orig_index list to data_index list (each orig_index maps to 2 pairs)."""
    data_indices = []
    for idx in orig_indices:
        data_indices.append(idx * 2)
        data_indices.append(idx * 2 + 1)
    return data_indices
