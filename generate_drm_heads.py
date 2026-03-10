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
from sklearn.decomposition import PCA


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

    print(f"Loading {len(emb_files)} embedding files...")
    arrays = []
    for f in emb_files:
        try:
            arr = np.load(f)
            arrays.append(arr)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

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

    print(f"Running PCA with n_components={k}...")
    pca = PCA(n_components=k)
    pca.fit(diff)
    components = pca.components_  # (k, hidden_dim)
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_

    print(f"Explained variance ratio (first 10): {explained_variance_ratio[:10]}")

    np.save(os.path.join(output_dir, "explained_variance_ratio.npy"), explained_variance_ratio)
    np.save(os.path.join(output_dir, "explained_variance.npy"), explained_variance)
    np.save(os.path.join(output_dir, "orthogonal_heads.npy"), components)

    # Save each component as PyTorch state dict (weight shape [1, hidden_dim]) for nn.Linear
    component_dir = os.path.join(output_dir, f"{case_name}-PCA-component")
    os.makedirs(component_dir, exist_ok=True)

    for i in range(k):
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
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--case_name", type=str, default="sb_bench", help="Prefix for .pth filenames")
    parser.add_argument("--full_composed", action="store_true", help="Use all dimensions (k=hidden_dim)")
    args = parser.parse_args()
    generate_orthogonal_heads(args)
