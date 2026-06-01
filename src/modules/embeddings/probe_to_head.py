"""
probe_to_head.py
─────────────────────────────────────────────────────────────────────────────
Phase 0.8 A3 — Convert a saved logistic-regression probe (`{variant}_L{N}_probe_weights.npz`
produced by probe_layers.py --save_weights_at_layer) into a reward-head
directory that the existing score_vlbias_offline.py + drm_loader.py
infrastructure can consume unchanged.

Each output dir mimics the layout `generate_drm_heads.py` produces, so the
scorer + the RL trainer treat probe heads exactly like SVM / PCA heads:

  <out_dir>/
    sb_bench-PROBE-component/
      sb_bench-PROBE-component0.pth   # state_dict {"weight": (1, D) float32 tensor}
    kept_heads_probe.json             # {"kept_indices": [0], "threshold": null, ...}
    metadata.json                     # source npz path, normalize flag, residualize_pcs, etc.

Reward semantics: r(phi(y)) = w^T phi(y)  (matches PCA / SVM heads).
The probe's intercept is intentionally dropped — the rank-1 head is used as
a directional reward, exactly like the SVM hyperplane normals.

Optional `--residualize_pcs` projects the probe weight orthogonal to the top-K
PCA component vectors loaded from an `orthogonal_heads.npy` file. This tests
whether removing the nuisance directions (e.g. the L9 84.6%-variance PC0)
sharpens the bias signal carried by the probe.

Usage:
  python -m modules.embeddings.probe_to_head \
      --weights_npz Phase0.8/probe_results/.../base_L9_probe_weights.npz \
      --out_dir    Phase0.8/a3_results/generated_heads_probe_L9_base \
      --normalize

  # With nuisance residualisation:
  python -m modules.embeddings.probe_to_head \
      --weights_npz ... \
      --out_dir    ... \
      --normalize \
      --residualize_pcs Phase0.8/a2_results/generated_heads_letter_post_letter_L9/orthogonal_heads.npy \
      --residualize_topk 2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def _unit_norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Probe weight vector has zero norm; cannot normalise.")
    return v / n


def _residualize(v: np.ndarray, pcs: np.ndarray, topk: int) -> np.ndarray:
    """Project v orthogonal to the first `topk` rows of `pcs` (assumed
    orthonormal as written by sklearn/cuML PCA). Returns residual v."""
    if topk <= 0:
        return v
    k = min(topk, pcs.shape[0])
    P = pcs[:k]                         # (k, D)
    # Re-orthonormalise rows defensively (PCA outputs should already be orthonormal
    # but tiny numerical drift in float32 mixes residual components).
    Q, _ = np.linalg.qr(P.T)            # (D, k) orthonormal columns
    proj = Q @ (Q.T @ v)                # component of v in span(P)
    return v - proj


def _orthogonalize_against(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: subtract from v its projection onto u.
    Returns the component of v orthogonal to u. u need not be unit."""
    nu2 = float(np.dot(u, u))
    if nu2 < 1e-12:
        raise ValueError("orthogonalize_against vector has near-zero norm.")
    return v - (float(np.dot(v, u)) / nu2) * u


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights_npz", required=True,
                   help="Path to <variant>_L{N}_probe_weights.npz from probe_layers.py")
    p.add_argument("--out_dir", required=True,
                   help="Output directory (will be created). Heads land at "
                        "<out_dir>/sb_bench-PROBE-component/ ; the scorer's heads_dir "
                        "should point at that subdir.")
    p.add_argument("--case_name", default="sb_bench",
                   help="Used to mirror generate_drm_heads.py naming convention.")
    p.add_argument("--head_name", default="PROBE",
                   help="Inserted between case_name and 'component'. Default 'PROBE' "
                        "→ sb_bench-PROBE-component/sb_bench-PROBE-component0.pth.")
    p.add_argument("--normalize", action="store_true",
                   help="Unit-normalise the weight vector before saving. "
                        "Recommended: the probe's L2 norm depends on regularisation; "
                        "normalising makes reward magnitudes comparable across "
                        "variants / heads / layers.")
    p.add_argument("--residualize_pcs", default=None,
                   help="Optional path to orthogonal_heads.npy (from generate_drm_heads.py). "
                        "If given, project the probe vector orthogonal to the top-K rows "
                        "before saving. Use to remove dominant nuisance directions "
                        "(e.g. L9 PC0+PC1 carry ~98%% of variance and are orthogonal to bias).")
    p.add_argument("--residualize_topk", type=int, default=2,
                   help="K for --residualize_pcs (default 2: PC0+PC1).")
    p.add_argument("--orthogonalize_against", default=None,
                   help="Optional path to a second probe weights .npz (e.g. is_C). "
                        "If given, Gram-Schmidt project the bias-probe weight orthogonal "
                        "to this vector before saving. Use to isolate the bias-aligned "
                        "residual after removing the C-detection direction (Phase 0.8 A3 Move 1).")
    args = p.parse_args()

    if not os.path.exists(args.weights_npz):
        raise FileNotFoundError(args.weights_npz)

    npz = np.load(args.weights_npz, allow_pickle=True)
    coef = np.asarray(npz["coef"], dtype=np.float32)
    if coef.ndim != 1:
        raise ValueError(f"Expected 1-D coef vector, got shape {coef.shape}")
    intercept = float(npz["intercept"]) if "intercept" in npz.files else 0.0
    layer_idx = int(npz["layer_idx"]) if "layer_idx" in npz.files else -2
    variant = str(npz["variant"]) if "variant" in npz.files else "unknown"
    task = str(npz["task"]) if "task" in npz.files else "is_good_C"
    train_acc = float(npz["train_acc"]) if "train_acc" in npz.files else float("nan")
    holdout_acc = float(npz["holdout_acc"]) if "holdout_acc" in npz.files else float("nan")
    train_n = int(npz["train_n"]) if "train_n" in npz.files else -1
    holdout_n = int(npz["holdout_n"]) if "holdout_n" in npz.files else -1
    token_position = (str(npz["token_position"])
                      if "token_position" in npz.files else "post_letter")
    reward_base = (str(npz["reward_base"])
                   if "reward_base" in npz.files else "unknown")
    D = coef.shape[0]

    print(f"▶ Loaded probe weights")
    print(f"    variant={variant}  layer={layer_idx}  task={task}")
    print(f"    hidden_dim={D}  ||w||={np.linalg.norm(coef):.4f}  intercept={intercept:+.4f}")
    print(f"    train_acc={train_acc:.3f} (n={train_n})  "
          f"holdout_acc={holdout_acc:.3f} (n={holdout_n})")

    w = coef.copy()
    residualize_applied = False
    orthogonalize_meta = None
    if args.orthogonalize_against:
        if not os.path.exists(args.orthogonalize_against):
            raise FileNotFoundError(args.orthogonalize_against)
        npz_u = np.load(args.orthogonalize_against, allow_pickle=True)
        u = np.asarray(npz_u["coef"], dtype=np.float32)
        if u.shape != w.shape:
            raise ValueError(
                f"orthogonalize_against shape {u.shape} != probe shape {w.shape}"
            )
        # Cosine and norm-retention reporting before mutating w.
        cos = float(np.dot(w, u) / (np.linalg.norm(w) * np.linalg.norm(u) + 1e-12))
        w_before_norm = float(np.linalg.norm(w))
        w = _orthogonalize_against(w, u)
        w_after_norm = float(np.linalg.norm(w))
        retained = w_after_norm / max(w_before_norm, 1e-12)
        u_task = str(npz_u["task"]) if "task" in npz_u.files else "unknown"
        print(f"    orthogonalised against {os.path.basename(args.orthogonalize_against)} "
              f"(task={u_task})")
        print(f"    cos(w_bias, w_orth) = {cos:+.4f}  "
              f"||w||_after / ||w||_before = {retained:.4f}  "
              f"(low retention = bias direction was mostly the C direction)")
        orthogonalize_meta = {
            "path": os.path.abspath(args.orthogonalize_against),
            "task": u_task,
            "cos_before": cos,
            "norm_retained": retained,
        }
    if args.residualize_pcs:
        if not os.path.exists(args.residualize_pcs):
            raise FileNotFoundError(args.residualize_pcs)
        pcs = np.load(args.residualize_pcs)
        if pcs.shape[1] != D:
            raise ValueError(
                f"PC dim {pcs.shape[1]} != probe dim {D}; "
                f"check you're using the matching layer's orthogonal_heads.npy"
            )
        w_before_norm = float(np.linalg.norm(w))
        w = _residualize(w, pcs, args.residualize_topk)
        w_after_norm = float(np.linalg.norm(w))
        retained = w_after_norm / max(w_before_norm, 1e-12)
        print(f"    residualised against top-{args.residualize_topk} PCs from "
              f"{os.path.basename(args.residualize_pcs)}")
        print(f"    ||w||_after / ||w||_before = {retained:.4f}  "
              f"(low retention = probe lived in PC nuisance subspace)")
        residualize_applied = True

    if args.normalize:
        w = _unit_norm(w)
        print(f"    unit-normalised → ||w||=1")

    # Build output dir matching generate_drm_heads.py format.
    os.makedirs(args.out_dir, exist_ok=True)
    component_dir = os.path.join(args.out_dir,
                                 f"{args.case_name}-{args.head_name}-component")
    os.makedirs(component_dir, exist_ok=True)

    # Save single head (positive direction only — semantically the probe says
    # "high projection = is_good_C", so the positive direction IS the reward.
    # We do not save a negated head: the scorer treats each .pth as one head,
    # and we want exactly one reward channel for this experiment.
    weight_2d = torch.tensor(w, dtype=torch.float32).unsqueeze(0)  # (1, D)
    state = {"weight": weight_2d.contiguous()}
    head_path = os.path.join(component_dir,
                             f"{args.case_name}-{args.head_name}-component0.pth")
    torch.save(state, head_path)
    print(f"✅ Wrote head: {head_path}  (shape={tuple(weight_2d.shape)})")

    # kept_heads_probe.json — single kept index, no threshold semantics.
    kept_path = os.path.join(args.out_dir, "kept_heads_probe.json")
    with open(kept_path, "w") as f:
        json.dump({
            "kept_indices": [0],
            "num_heads_total": 1,
            "kept_count": 1,
            "threshold": None,
            "source": "probe_layers.py P3 is_good_C logistic regression",
        }, f, indent=2)
    print(f"✅ Wrote {kept_path}")

    # Audit metadata so any downstream analysis can trace exactly how this
    # head was produced.
    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "source_weights_npz": os.path.abspath(args.weights_npz),
            "variant": variant,
            "layer_idx": layer_idx,
            "task": task,
            "token_position": token_position,
            "reward_base": reward_base,
            "hidden_dim": D,
            "normalize": bool(args.normalize),
            "residualize_pcs": (os.path.abspath(args.residualize_pcs)
                                if args.residualize_pcs else None),
            "residualize_topk": (args.residualize_topk
                                 if residualize_applied else 0),
            "orthogonalize_against": orthogonalize_meta,
            "intercept_dropped": True,
            "train_n": train_n,
            "train_acc": train_acc,
            "holdout_n": holdout_n,
            "holdout_acc": holdout_acc,
            "num_heads_written": 1,
        }, f, indent=2)
    print(f"✅ Wrote {meta_path}")
    print()
    print("Next: point score_vlbias_offline.py at this head dir, e.g.")
    print(f"  --heads_dir {component_dir}")
    print(f"  --head_type probe")
    print(f"  --layer_idx {layer_idx}")
    print(f"  --kept_heads {kept_path}")


if __name__ == "__main__":
    main()
