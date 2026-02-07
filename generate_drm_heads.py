
import os
import glob
import numpy as np
import argparse
from sklearn.decomposition import PCA
import torch

def generate_orthogonal_heads(args):
    """
    Generates orthogonal DRM heads using PCA on difference embeddings.
    """
    input_dir = args.input_dir
    output_dir = args.output_dir
    n_components = args.n_components
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Searching for embedding files in {input_dir}...")
    emb_files = glob.glob(os.path.join(input_dir, "emb_*.npy"))
    
    if not emb_files:
        print(f"No embedding files found in {input_dir}. Please run cal_emb_modular.py first.")
        return

    print(f"Found {len(emb_files)} embedding files. Loading...")
    
    diff_vectors = []
    
    for f in emb_files:
        try:
            # Load embedding: shape (3, hidden_dim) -> [chosen, rejected, prompt]
            emb = np.load(f)
            
            # Extract chosen and rejected
            # Note: Verify the shape and order based on reward_trainer.py
            # In reward_trainer.py: cls_emb = emb.float().cpu().numpy() [1, 3, hidden] or [3, hidden]
            # Let's inspect shape dynamically
            if len(emb.shape) == 3:
                emb = emb[0] # Remove batch dim if present [1, 3, H] -> [3, H] or [1, 4, H] -> [4, H]
            
            # We expect at least 2 vectors (chosen, rejected)
            # Extra vectors (prompt) are ignored for head generation
            chosen_emb = emb[0]
            rejected_emb = emb[1]
            
            # Compute difference: chosen - rejected
            diff = chosen_emb - rejected_emb
            diff_vectors.append(diff)
            
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
            
    if not diff_vectors:
        print("No valid difference vectors extracted.")
        return
        
    # Stack vectors: (N, hidden_dim)
    X = np.stack(diff_vectors)
    print(f"Constructed data matrix X with shape {X.shape}")
    
    # Adjust n_components if we have fewer samples than requested
    n_samples = X.shape[0]
    if n_samples < n_components:
        print(f"Warning: n_samples ({n_samples}) < n_components ({n_components}). Adjusting n_components to {n_samples}.")
        n_components = n_samples
        
    if n_components == 0:
        print("Error: Not enough samples to run PCA (need at least 1).")
        return
    
    # Run PCA
    print(f"Running PCA with n_components={n_components}...")
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Extract components (heads)
    # pca.components_ has shape (n_components, hidden_dim)
    heads = pca.components_
    explained_variance = pca.explained_variance_ratio_
    
    print("PCA completed.")
    print(f"Explained variance ratios: {explained_variance}")
    
    # Save heads
    save_path = os.path.join(output_dir, "orthogonal_heads.npy")
    np.save(save_path, heads)
    print(f"Saved orthogonal heads to {save_path}")
    
    # Verify orthogonality
    print("Verifying orthogonality...")
    dot_products = np.dot(heads, heads.T)
    # Off-diagonal elements should be close to 0
    off_diagonal = dot_products - np.diag(np.diag(dot_products))
    max_error = np.max(np.abs(off_diagonal))
    print(f"Max dot product error (should be close to 0): {max_error:.6f}")
    
    # Save explained variance
    np.save(os.path.join(output_dir, "explained_variance.npy"), explained_variance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Orthogonal DRM Heads using PCA")
    parser.add_argument("--input_dir", type=str, default="./embeddings_output", help="Directory containing .npy embedding files")
    parser.add_argument("--output_dir", type=str, default="./generated_heads", help="Directory to save generated heads")
    parser.add_argument("--n_components", type=int, default=5, help="Number of orthogonal heads to generate")
    
    args = parser.parse_args()
    generate_orthogonal_heads(args)
