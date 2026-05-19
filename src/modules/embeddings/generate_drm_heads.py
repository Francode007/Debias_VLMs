"""
Generate orthogonal DRM heads from preference embeddings using PCA.

Loads (chosen, rejected) embeddings from cal_emb output, computes difference
vectors, runs PCA, and saves each component as a PyTorch .pth state dict
(both positive and negated directions) for use with score_head.MultipleHead.

Output (under output_dir/):
  - explained_variance_ratio.npy : (k,) — fraction of total variance per component.
  - explained_variance.npy       : (k,) — eigenvalue (variance) per component.
  - orthogonal_heads.npy         : (k, hidden_dim) — all component vectors.
  - {case_name}-PCA-component/   : directory of .pth files:
      - component0 .. component(k-1) : positive direction (w_i).
      - componentk .. component(2k-1) : negated direction (-w_i).

PCA components (what each one is):
  - Input to PCA is the matrix of difference vectors: d_n = phi(chosen_n) - phi(rejected_n).
  - Component i is the i-th principal direction (eigenvector) of the covariance of these
    differences. So component 0 is the direction of largest variance in (chosen - rejected),
    component 1 is the next orthogonal direction, etc.
  - Each component can be interpreted as an axis of preference: responses with higher
    projection onto w_i are more "in the chosen direction" for that axis. The negated
    head (-w_i) gives the opposite preference.
  - For SB-Bench, early components often capture broad stereotype vs non-stereotype;
    later components can capture finer or category-specific bias dimensions.
  - Reward for a response embedding phi(y) under head i is: r_i(y) = w_i^T phi(y)
    (or using -w_i for the negated head). Used in evaluate_drm_heads and in RL.
"""

import os
import glob
import argparse
import numpy as np
import torch
import time
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from cuml.decomposition import PCA
    # Verify GPU is actually accessible (cuML needs a CUDA driver)
    import cupy
    cupy.cuda.device.get_device_id()
    HAS_CUML = True
except (ImportError, Exception):
    from sklearn.decomposition import PCA
    HAS_CUML = False

def generate_orthogonal_heads(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    n_components = args.n_components
    case_name = getattr(args, "case_name", "sb_bench")
    full_composed = getattr(args, "full_composed", False)

    os.makedirs(output_dir, exist_ok=True)

    emb_files = sorted(glob.glob(os.path.join(input_dir, "emb_*.npy")))
    if not emb_files:
        print(f"No embedding files found in {input_dir}. Run cal_emb_modular.py first.")
        return

    # --- Filter to split indices if requested ---
    split_mode = getattr(args, "split", "all")
    split_indices_path = getattr(args, "split_indices_path", None)
    if split_mode in ("train", "test") and split_indices_path:
        with open(split_indices_path, "r") as f:
            split_info = json.load(f)
        orig_indices = set(split_info["train_indices"] if split_mode == "train" else split_info["test_indices"])
        # data_index = orig_index * 2 + pair_idx, so valid data_indices are {idx*2, idx*2+1}
        valid_data_indices = set()
        for idx in orig_indices:
            valid_data_indices.add(idx * 2)
            valid_data_indices.add(idx * 2 + 1)
        
        import re
        def _get_data_index(path):
            m = re.search(r"emb_(\d+)\.npy$", path)
            return int(m.group(1)) if m else -1
        
        emb_files = [f for f in emb_files if _get_data_index(f) in valid_data_indices]
        print(f"Filtered to {len(emb_files)} embedding files for '{split_mode}' split")

    print(f"Loading {len(emb_files)} embedding files (parallel I/O)...")
    
    def _load_one(path):
        try:
            return np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    arrays = [None] * len(emb_files)
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(_load_one, f): i for i, f in enumerate(emb_files)}
        for future in tqdm.tqdm(as_completed(futures), total=len(emb_files), desc="Loading embeddings"):
            idx = futures[future]
            result = future.result()
            if result is not None:
                arrays[idx] = result
    
    # Filter out failed loads (None entries)
    arrays = [a for a in arrays if a is not None]

    # Merge: each file is (1, 3, hidden_dim) or (3, hidden_dim) -> chosen, rejected, prompt
    merged = np.concatenate(arrays, axis=0)
    if merged.ndim == 3:
        vectors = merged  # (N, 3, hidden_dim)
    else:
        vectors = merged[np.newaxis, ...]  # (1, 3, hidden_dim)

    # Use chosen (0) and rejected (1) only
    diff = vectors[:, 0, :] - vectors[:, 1, :]  # (N, hidden_dim)
    print(f"Difference matrix shape: {diff.shape}")

    # Drop rows with NaN or inf (e.g. from model numerical issues); PCA requires finite input
    valid = np.isfinite(diff).all(axis=1)
    if not valid.all():
        n_bad = int((~valid).sum())
        print(f"Dropping {n_bad} sample(s) with NaN/inf in embeddings (e.g. model numerical issues).")
        diff = diff[valid]
        if diff.shape[0] == 0:
            print("No valid samples left after dropping NaN/inf. Check embedding extraction (e.g. device/dtype).")
            return
        print(f"Difference matrix shape after cleanup: {diff.shape}")

    hidden_dim = diff.shape[1]
    k = min(n_components, diff.shape[0]) if not full_composed else hidden_dim
    if k == 0:
        print("Not enough samples for PCA.")
        return

    print(f"Zero-centering difference matrix... Original mean: {diff.mean():.6f}")
    diff = diff - np.mean(diff, axis=0)

    if HAS_CUML:
        print(f"Running PCA with n_components={k} using RAPIDS cuML (GPU)...")
    else:
        print(f"Running PCA with n_components={k} using sklearn (CPU fallback)...")
        
    pca = PCA(n_components=k)
    
    t_start = time.time()
    pca.fit(diff)
    t_end = time.time()
    pure_pca_time = t_end - t_start

    components = pca.components_  # (k, hidden_dim)
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_
    
    # Save pure pca timing metrics
    try:
        with open("/tmp/pca_metrics.json", "w") as f:
            json.dump({"pure_pca_fit_time_seconds": pure_pca_time}, f)
    except Exception as e:
        print(f"Could not save pca metrics: {e}")

    print(f"Explained variance ratio (first 10): {explained_variance_ratio[:10]}")

    np.save(os.path.join(output_dir, "explained_variance_ratio.npy"), explained_variance_ratio)
    np.save(os.path.join(output_dir, "explained_variance.npy"), explained_variance)
    np.save(os.path.join(output_dir, "orthogonal_heads.npy"), components)

    # Save each component as PyTorch state dict (weight shape [1, hidden_dim]) for nn.Linear
    component_dir = os.path.join(output_dir, f"{case_name}-PCA-component")
    os.makedirs(component_dir, exist_ok=True)

    for i in tqdm.tqdm(range(k), desc="Saving PCA heads"):
        comp = components[i]
        comp_t = torch.tensor(comp, dtype=torch.float32)
        comp_2d = comp_t.unsqueeze(0)  # (1, hidden_dim)
        state = {"weight": comp_2d}
        path_pos = os.path.join(component_dir, f"{case_name}-PCA-component{i}.pth")
        torch.save(state, path_pos)

        state_neg = {"weight": (-comp_2d).contiguous()}
        path_neg = os.path.join(component_dir, f"{case_name}-PCA-component{i + k}.pth")
        torch.save(state_neg, path_neg)

    print(f"Saved {2 * k} head files (positive + negated) to {component_dir}")

    # Orthogonality check
    dot = np.dot(components, components.T)
    off = dot - np.diag(np.diag(dot))
    print(f"Max off-diagonal dot product: {np.max(np.abs(off)):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate orthogonal DRM heads (PCA) from embeddings")
    parser.add_argument("--input_dir", type=str, default="./embeddings_output", help="Directory with emb_*.npy files")
    parser.add_argument("--output_dir", type=str, default="./generated_heads", help="Directory to save heads and metadata")
    parser.add_argument("--n_components", type=int, default=100, help="Number of PCA components")
    parser.add_argument("--case_name", type=str, default="sb_bench", help="Prefix for .pth filenames")
    parser.add_argument("--full_composed", action="store_true", help="Use all dimensions (k=hidden_dim)")
    parser.add_argument("--split_indices_path", type=str, default=None, help="Path to split_indices.json for filtering to train split")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="Which split to use (default: all)")
    args = parser.parse_args()
    generate_orthogonal_heads(args)
