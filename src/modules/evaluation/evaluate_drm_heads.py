"""
Evaluate DRM heads on SB-Bench (Phase 1 hypothesis check).

This script answers: "Do the PCA-derived reward heads actually separate
chosen (non-stereotypical) from rejected (stereotypical) responses?"

What it does:
  1. Loads all emb_*.npy from the embedding step (each has chosen_emb, rejected_emb).
  2. Loads the DRM heads from generate_drm_heads output (directory of .pth files).
  3. For each head, computes reward_chosen = w^T chosen_emb and reward_rejected = w^T rejected_emb.
  4. Counts "correct" when reward_chosen > reward_rejected (head prefers non-stereotypical).
  5. Reports overall accuracy (fraction of pairs where chosen wins) and per-head accuracy.
  6. Maps each sample back to SB-Bench category (orig_index = data_index // 2) and reports
     per-category accuracy so you can see which bias dimensions each head captures.

Output (--output_json):
  - overall_per_head: [acc_head_0, acc_head_1, ...] — accuracy when using only that head.
  - overall_mean: mean accuracy over all heads/samples.
  - per_category: for each of the 9 categories, accuracy_per_head and accuracy_mean.

How to interpret (Phase 1 hypothesis):
  - If overall_mean >> 0.5, the PCA directions collectively separate chosen vs rejected.
  - If some heads have much higher accuracy than others, those axes are more predictive.
  - Per-category breakdown shows which heads help for Age, Gender, etc.; useful for
    selecting or combining heads in Stage 2 (RL).
"""

import os
import glob
import re
import json
import argparse
import numpy as np
import torch
import pandas as pd
import json
import time

from tqdm import tqdm
from modules.rl_components.score_head import MultipleHead


# SB-Bench category names (9 bias types)
SB_BENCH_CATEGORIES = [
    "Age",
    "Disability",
    "Gender",
    "Nationality",
    "Physical Appearance",
    "Race/Ethnicity",
    "Religion",
    "SES",
    "Sexual Orientation",
]


def _data_index_from_path(path: str) -> int:
    m = re.search(r"emb_(\d+)\.npy$", path)
    return int(m.group(1)) if m else 0


def load_embeddings(emb_dir: str):
    """Load all emb_*.npy from emb_dir, return (embs, data_indices)."""
    paths = sorted(glob.glob(os.path.join(emb_dir, "emb_*.npy")), key=_data_index_from_path)
    data_indices = [_data_index_from_path(p) for p in paths]
    embs = []
    for p in tqdm(paths, desc="Loading embeddings"):
        e = np.load(p)
        if e.ndim == 3:
            e = e[0]
        embs.append(e[:2, :])
    embs = np.stack(embs, axis=0)
    return embs, data_indices


def load_categories(data_path: str, max_data_index: int) -> list:
    """Load parquet(s) from data_path; return category name per data_index (orig_index = data_index // 2)."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(data_path, "*.parquet")))
    if not files:
        return [None] * (max_data_index + 1)
    dfs = [pd.read_parquet(f, engine="fastparquet") for f in files]
    df = pd.concat(dfs, ignore_index=True)
    categories = []
    for i in range(max_data_index + 1):
        orig = i // 2
        if orig < len(df):
            cat = df.iloc[orig].get("category", None)
            if hasattr(cat, "item"):
                cat = cat.item()
            if isinstance(cat, int) and 0 <= cat < len(SB_BENCH_CATEGORIES):
                cat = SB_BENCH_CATEGORIES[cat]
            categories.append(cat)
        else:
            categories.append(None)
    return categories


def main():
    parser = argparse.ArgumentParser(description="Evaluate DRM heads on SB-Bench")
    parser.add_argument("--emb_dir", type=str, default="./embeddings_output", help="Directory with emb_*.npy")
    parser.add_argument("--score_head_weight", type=str, required=True, help="Directory with .pth component files")
    parser.add_argument("--data_path", type=str, default="./sb_bench_data/data", help="SB-Bench parquet dir for categories")
    parser.add_argument("--output_json", type=str, default="./drm_head_results.json", help="Output JSON path")
    parser.add_argument("--num_heads", type=int, default=None, help="Use first N heads (default: all)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for evaluation (default: 1024)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    embs, data_indices = load_embeddings(args.emb_dir)
    n_samples, _, hidden_size = embs.shape
    max_idx = max(data_indices) if data_indices else 0
    categories = load_categories(args.data_path, max_idx)
    category_list = [categories[i] for i in data_indices]

    head = MultipleHead(
        hidden_size=int(hidden_size),
        score_head_weight=args.score_head_weight,
        device=device,
        num_heads=args.num_heads,
    )
    head.eval()

    embs_t = torch.tensor(embs, dtype=torch.float32, device=device)
    
    rewards_chosen_list = []
    rewards_rejected_list = []
    batch_size = args.batch_size
    
    with torch.no_grad():
        t_start = time.time()
        for i in tqdm(range(0, len(embs_t), batch_size), desc="Evaluating heads"):
            batch_embs = embs_t[i:i+batch_size]
            rc, rr = head(batch_embs)
            rewards_chosen_list.append(rc.cpu().numpy())
            rewards_rejected_list.append(rr.cpu().numpy())
        t_end = time.time()
        pure_eval_time = t_end - t_start
        
    try:
        with open("/tmp/eval_metrics.json", "w") as f:
            json.dump({"pure_eval_loop_time_seconds": pure_eval_time}, f)
    except Exception as e:
        print(f"Could not save eval metrics: {e}")
            
    rewards_chosen = np.concatenate(rewards_chosen_list, axis=0)
    rewards_rejected = np.concatenate(rewards_rejected_list, axis=0)
    num_heads = rewards_chosen.shape[1]

    results = {}
    correct = (rewards_chosen > rewards_rejected).astype(np.float64)
    overall_per_head = correct.mean(axis=0)
    results["overall_per_head"] = [float(x) for x in overall_per_head]
    results["overall_mean"] = float(correct.mean())
    results["num_samples"] = n_samples
    results["num_heads"] = num_heads

    per_category = {}
    for cat_name in SB_BENCH_CATEGORIES:
        mask = np.array([c == cat_name for c in category_list])
        if mask.sum() == 0:
            continue
        cat_correct = correct[mask]
        per_head = cat_correct.mean(axis=0)
        per_category[cat_name] = {
            "accuracy_per_head": [float(x) for x in per_head],
            "accuracy_mean": float(cat_correct.mean()),
            "count": int(mask.sum()),
        }
    results["per_category"] = per_category

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_json}")
    print(f"Overall accuracy (mean over heads): {results['overall_mean']:.4f}")
    for cat, v in per_category.items():
        print(f"  {cat}: {v['accuracy_mean']:.4f} (n={v['count']})")


if __name__ == "__main__":
    main()
