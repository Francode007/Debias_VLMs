#!/usr/bin/env python3
"""Rank-1 bias-geometry experiment.

Question: across the 10 BBQ demographic axes, do the per-axis `bias_aligned`
linear probes (fit on Qwen2.5-VL-3B base hidden states) collapse onto a
single shared direction, and does the degree of collapse change with depth?

Method:
  1. Load cached hidden states (n=900, 37 layers, 2048-d) and per-record metadata.
  2. For each layer L ∈ {1, 5, 9, 13, 17, 21, 25, 29, 33, 35}:
     a. For each axis A ∈ 10 BBQ axes:
        - subset to disambig records of axis A with non-None bias_aligned label
        - fit LogisticRegression(C=1.0, class_weight='balanced', l2)
          using 5-fold StratifiedKFold; average normalised coefficient
          vectors across folds → w_{A,L} ∈ ℝ^{2048}
        - also fit a SHARED probe on all 600 disambig records → w_*_L
     b. Stack 10 per-axis unit vectors into W_L ∈ ℝ^{10×2048}
     c. PCA on W_L: report explained-variance ratio of PC0..PC4,
        mean pairwise |cos(w_i, w_j)|, mean |cos(w_A, w_*)| (axis-shared).
  3. Report cross-layer table and write JSON.

Output: Phase0.8/a3_results/rank1_bias_geometry.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from modules.evaluation.probe_layers import (  # type: ignore
    _eval_parse_choice, _is_bias_aligned,
)

LAYERS = [1, 5, 9, 13, 17, 21, 25, 29, 33, 35]
N_FOLDS = 5
C_REG = 1.0
RNG_SEED = 42

CACHE = ROOT / "Phase0.8/a3_results/local_cache/base_probe_hs.npz"
OUT = ROOT / "Phase0.8/a3_results/rank1_bias_geometry.json"


def fit_axis_probe(X: np.ndarray, y: np.ndarray, seed: int = RNG_SEED) -> tuple[np.ndarray, float]:
    """Return (mean unit-normalised coef across CV folds, mean CV holdout accuracy)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    coefs, accs = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(
            C=C_REG, penalty="l2", class_weight="balanced",
            solver="lbfgs", max_iter=2000, random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
        w = clf.coef_.ravel().astype(np.float64)
        n = np.linalg.norm(w)
        if n > 0:
            w = w / n
        coefs.append(w)
    w_mean = np.mean(coefs, axis=0)
    nrm = np.linalg.norm(w_mean)
    if nrm > 0:
        w_mean = w_mean / nrm
    return w_mean, float(np.mean(accs))


def pca_summary(W: np.ndarray) -> dict:
    """W : (k, D) row-stacked unit vectors. Returns {evr, cum_evr, mean_pairwise_cos}."""
    # center rows
    Wc = W - W.mean(axis=0, keepdims=True)
    # SVD; full_matrices=False → singular values length min(k, D)
    U, s, Vt = np.linalg.svd(Wc, full_matrices=False)
    var = s ** 2
    evr = (var / var.sum()).tolist() if var.sum() > 0 else [0.0] * len(s)
    cum_evr = np.cumsum(evr).tolist()
    # pairwise |cos|
    G = W @ W.T   # already unit-normalised
    iu = np.triu_indices_from(G, k=1)
    pairwise_abs_cos = float(np.abs(G[iu]).mean())
    pairwise_signed_cos = float(G[iu].mean())
    return {
        "explained_variance_ratio": evr[:5],
        "cumulative_evr": cum_evr[:5],
        "mean_pairwise_abs_cos": pairwise_abs_cos,
        "mean_pairwise_signed_cos": pairwise_signed_cos,
    }


def main() -> None:
    print(f"Loading cache: {CACHE}")
    d = np.load(CACHE, allow_pickle=True)
    hs = d["hs_primary"]            # (N, L_all, D)
    meta = d["kept_primary"]        # (N,) object
    N, L_total, D = hs.shape
    print(f"  N={N}  L_total={L_total}  D={D}")

    # Compute per-record bias_aligned label
    labels = np.array([_is_bias_aligned(r) for r in meta], dtype=object)
    axes = np.array([r.get("bbq_axis") for r in meta])
    keep = np.array([l is not None for l in labels])
    y_all = np.array([int(l) if l is not None else -1 for l in labels])

    print(f"  bias_aligned usable: {int(keep.sum())} / {N}")
    print(f"  unique axes: {sorted(set(axes[keep]))}")
    unique_axes = sorted(set(axes[keep]))
    assert len(unique_axes) == 10, f"expected 10 axes, got {len(unique_axes)}"

    out = {"layers": LAYERS, "axes": unique_axes, "n_folds": N_FOLDS, "C_reg": C_REG,
           "n_total_disambig": int(keep.sum()),
           "per_axis_n": {ax: int(((axes == ax) & keep).sum()) for ax in unique_axes},
           "per_axis_pos_rate": {ax: float(y_all[(axes == ax) & keep].mean()) for ax in unique_axes},
           "by_layer": {}}

    print("\n=== Rank-1 bias-geometry sweep ===")
    print(f"{'L':>3}  {'PC0 evr':>8}  {'PC1 evr':>8}  {'PC2 evr':>8}  {'cum PC0-1':>10}  {'|cos| mean':>10}  {'cos signed':>10}  {'avg axis-shared cos':>20}  {'mean per-axis CV acc':>20}")

    for L in LAYERS:
        # Per-axis probes
        W = np.zeros((10, D), dtype=np.float64)
        per_axis_acc = []
        per_axis_n = []
        for i, ax in enumerate(unique_axes):
            mask = (axes == ax) & keep
            X_ax = hs[mask, L, :].astype(np.float64)
            y_ax = y_all[mask].astype(int)
            n_pos, n_neg = int(y_ax.sum()), int((y_ax == 0).sum())
            assert n_pos >= N_FOLDS and n_neg >= N_FOLDS, f"axis {ax} L{L}: pos={n_pos} neg={n_neg}"
            w, acc = fit_axis_probe(X_ax, y_ax)
            W[i] = w
            per_axis_acc.append(acc)
            per_axis_n.append(int(mask.sum()))

        # Shared probe (all 600 disambig)
        Xs = hs[keep, L, :].astype(np.float64)
        ys = y_all[keep].astype(int)
        w_shared, acc_shared = fit_axis_probe(Xs, ys)

        # PCA on per-axis stack
        summ = pca_summary(W)

        # Axis-shared alignment
        cos_shared = np.array([float(np.dot(W[i], w_shared)) for i in range(10)])
        mean_abs_cos_shared = float(np.mean(np.abs(cos_shared)))

        out["by_layer"][L] = {
            "pc_explained_variance_ratio": summ["explained_variance_ratio"],
            "pc_cumulative_evr": summ["cumulative_evr"],
            "mean_pairwise_abs_cos_per_axis": summ["mean_pairwise_abs_cos"],
            "mean_pairwise_signed_cos_per_axis": summ["mean_pairwise_signed_cos"],
            "mean_abs_cos_axis_vs_shared": mean_abs_cos_shared,
            "per_axis_cv_acc_mean": float(np.mean(per_axis_acc)),
            "per_axis_cv_acc_min": float(np.min(per_axis_acc)),
            "shared_probe_cv_acc": acc_shared,
            "per_axis_cv_acc": dict(zip(unique_axes, per_axis_acc)),
            "per_axis_n": dict(zip(unique_axes, per_axis_n)),
            "cos_axis_vs_shared": dict(zip(unique_axes, cos_shared.tolist())),
        }
        evr = summ["explained_variance_ratio"]
        print(f"{L:>3}  {evr[0]:>8.3f}  {evr[1]:>8.3f}  "
              f"{evr[2]:>8.3f}  {summ['cumulative_evr'][1]:>10.3f}  "
              f"{summ['mean_pairwise_abs_cos']:>10.3f}  "
              f"{summ['mean_pairwise_signed_cos']:>10.3f}  "
              f"{mean_abs_cos_shared:>20.3f}  "
              f"{np.mean(per_axis_acc):>20.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ wrote {OUT}")


if __name__ == "__main__":
    main()
