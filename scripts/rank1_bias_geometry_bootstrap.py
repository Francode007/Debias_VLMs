#!/usr/bin/env python3
"""R1 — Bootstrap CIs on per-layer PC0 evr and mean pairwise |cos|.

Hardens §9.4 of Workshop_Submission/REPORT_EXTENSIVE.md (two-regime
geometric structure). Original `rank1_bias_geometry.py` reports point
estimates from the single 5-fold CV fit per axis. This script
resamples records *within each axis with replacement* B times, refits
the per-axis probe each resample, and reports 95% CIs on:

    * PC0 explained-variance ratio
    * mean pairwise |cos(w_i, w_j)|

Decision rule (per the roadmap):
    Two-regime claim is publication-grade iff the L13 PC0 95% CI does
    not overlap the L25 PC0 95% CI, AND the L13 mean |cos| 95% CI does
    not overlap the L25 mean |cos| 95% CI.

Method:
  for b in 1..B:
      for each axis A:
          resample records in axis A with replacement (n=n_A)
          fit LogisticRegression on the resampled records (5-fold avg,
          same hyperparameters as rank1_bias_geometry.py)
          → w^(b)_A
      stack 10 unit vectors → W^(b)
      PCA: store PC0 evr^(b) and mean pairwise |cos|^(b)
  per layer: report mean, 2.5%, 97.5% percentiles across the B draws.

Output: Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from modules.evaluation.probe_layers import _is_bias_aligned  # type: ignore

# Same defaults as rank1_bias_geometry.py for direct comparability
LAYERS = [1, 5, 9, 13, 17, 21, 25, 29, 33, 35]
N_FOLDS = 5
C_REG = 1.0
RNG_SEED = 42
DEFAULT_B = 200

CACHE = ROOT / "Phase0.8/a3_results/local_cache/base_probe_hs.npz"
OUT = ROOT / "Phase0.8/a3_results/rank1_bias_geometry_bootstrap.json"


def fit_axis_probe_unit(
    X: np.ndarray, y: np.ndarray, seed: int = RNG_SEED, n_folds: int = N_FOLDS
) -> np.ndarray:
    """Mean unit-normalised coef across stratified-CV folds. Matches
    rank1_bias_geometry.py.fit_axis_probe but drops the held-out accuracy
    return (not needed for bootstrap).

    Falls back gracefully when class-stratification breaks under resampling
    (some bootstrap draws can have <n_folds positives or negatives): in
    that case fit a single LogReg on the full bootstrap sample.
    """
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos < n_folds or n_neg < n_folds:
        clf = LogisticRegression(
            C=C_REG, penalty="l2", class_weight="balanced",
            solver="lbfgs", max_iter=2000, random_state=seed,
        )
        clf.fit(X, y)
        w = clf.coef_.ravel().astype(np.float64)
        nrm = np.linalg.norm(w)
        return w / nrm if nrm > 0 else w

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    coefs = []
    for tr, _ in skf.split(X, y):
        clf = LogisticRegression(
            C=C_REG, penalty="l2", class_weight="balanced",
            solver="lbfgs", max_iter=2000, random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        w = clf.coef_.ravel().astype(np.float64)
        nrm = np.linalg.norm(w)
        if nrm > 0:
            w = w / nrm
        coefs.append(w)
    w_mean = np.mean(coefs, axis=0)
    nrm = np.linalg.norm(w_mean)
    return w_mean / nrm if nrm > 0 else w_mean


def pca_stats(W: np.ndarray) -> tuple[float, float]:
    """Return (PC0 evr, mean pairwise |cos|) for row-stacked unit vectors."""
    Wc = W - W.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Wc, full_matrices=False)
    var = s ** 2
    evr0 = float(var[0] / var.sum()) if var.sum() > 0 else 0.0
    G = W @ W.T  # already unit-normalised
    iu = np.triu_indices_from(G, k=1)
    mean_abs = float(np.abs(G[iu]).mean())
    return evr0, mean_abs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-bootstrap", "-B", type=int, default=DEFAULT_B,
                   help="number of bootstrap resamples (default 200)")
    p.add_argument("--seed", type=int, default=RNG_SEED)
    p.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    print(f"Loading cache: {args.cache}")
    d = np.load(args.cache, allow_pickle=True)
    hs = d["hs_primary"]            # (N, L_all, D)
    meta = d["kept_primary"]        # (N,) object
    N, L_total, D = hs.shape
    print(f"  N={N}  L_total={L_total}  D={D}")

    labels = np.array([_is_bias_aligned(r) for r in meta], dtype=object)
    axes = np.array([r.get("bbq_axis") for r in meta])
    keep = np.array([l is not None for l in labels])
    y_all = np.array([int(l) if l is not None else -1 for l in labels])
    unique_axes = sorted(set(axes[keep]))
    assert len(unique_axes) >= 2, f"expected >=2 axes, got {len(unique_axes)}"
    n_axes = len(unique_axes)
    print(f"  n_axes = {n_axes}")

    # Pre-slice the data per axis to avoid repeated boolean indexing
    per_axis: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ax in unique_axes:
        mask = (axes == ax) & keep
        idx = np.where(mask)[0]
        per_axis[ax] = (
            idx,
            hs[mask].astype(np.float32),  # (n_ax, L_all, D)
            y_all[mask].astype(int),
        )
        print(f"  axis {ax:24s} n={len(idx)}  pos_rate={float(y_all[mask].mean()):.3f}")

    rng = np.random.default_rng(args.seed)

    out: dict = {
        "layers": list(args.layers),
        "axes": unique_axes,
        "n_folds": N_FOLDS,
        "C_reg": C_REG,
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "per_axis_n": {ax: int(len(per_axis[ax][0])) for ax in unique_axes},
        "per_axis_pos_rate": {ax: float(per_axis[ax][2].mean()) for ax in unique_axes},
        "by_layer": {},
    }

    print(
        f"\n=== Bootstrap (B={args.n_bootstrap}) — 95% CIs per layer "
        f"(percentile method) ==="
    )
    print(
        f"{'L':>3}  "
        f"{'PC0 mean':>9}  {'PC0  2.5%':>10}  {'PC0 97.5%':>10}  "
        f"{'|cos| mean':>11}  {'|cos|  2.5%':>11}  {'|cos| 97.5%':>11}"
    )

    for L in args.layers:
        t0 = time.time()
        pc0_draws: list[float] = []
        cos_draws: list[float] = []
        # Pre-compute per-axis HS at this layer (saves an .astype(float64) per draw)
        per_axis_XL = {
            ax: per_axis[ax][1][:, L, :].astype(np.float64) for ax in unique_axes
        }

        for b in range(args.n_bootstrap):
            W = np.zeros((n_axes, D), dtype=np.float64)
            degenerate_b = False
            for i, ax in enumerate(unique_axes):
                _, _, y_ax = per_axis[ax]
                X_L = per_axis_XL[ax]
                n_ax = len(y_ax)
                # Bootstrap-resample records of this axis with replacement
                resample_idx = rng.integers(0, n_ax, size=n_ax)
                Xb = X_L[resample_idx]
                yb = y_ax[resample_idx]
                # Edge case: resampled draw has only one class → degenerate
                if len(np.unique(yb)) < 2:
                    degenerate_b = True
                    break
                W[i] = fit_axis_probe_unit(Xb, yb, seed=args.seed + b)
            if degenerate_b:
                continue
            evr0, mean_abs = pca_stats(W)
            pc0_draws.append(evr0)
            cos_draws.append(mean_abs)

        pc0_arr = np.array(pc0_draws)
        cos_arr = np.array(cos_draws)
        pc0_mean = float(pc0_arr.mean())
        pc0_lo, pc0_hi = (float(x) for x in np.percentile(pc0_arr, [2.5, 97.5]))
        cos_mean = float(cos_arr.mean())
        cos_lo, cos_hi = (float(x) for x in np.percentile(cos_arr, [2.5, 97.5]))

        out["by_layer"][str(L)] = {
            "n_draws_used": int(len(pc0_arr)),
            "n_draws_degenerate": int(args.n_bootstrap - len(pc0_arr)),
            "pc0_evr_mean": pc0_mean,
            "pc0_evr_ci_low": pc0_lo,
            "pc0_evr_ci_high": pc0_hi,
            "mean_pairwise_abs_cos_mean": cos_mean,
            "mean_pairwise_abs_cos_ci_low": cos_lo,
            "mean_pairwise_abs_cos_ci_high": cos_hi,
        }
        elapsed = time.time() - t0
        print(
            f"{L:>3}  "
            f"{pc0_mean:>9.3f}  {pc0_lo:>10.3f}  {pc0_hi:>10.3f}  "
            f"{cos_mean:>11.3f}  {cos_lo:>11.3f}  {cos_hi:>11.3f}  "
            f"[{elapsed:.0f}s, {len(pc0_arr)}/{args.n_bootstrap} usable]"
        )

    # Decision check: do L13 and L25 95% CIs overlap?
    by = out["by_layer"]
    if "13" in by and "25" in by:
        pc0_separable = by["13"]["pc0_evr_ci_high"] < by["25"]["pc0_evr_ci_low"]
        cos_separable = (
            by["13"]["mean_pairwise_abs_cos_ci_high"]
            < by["25"]["mean_pairwise_abs_cos_ci_low"]
        )
        out["verdict"] = {
            "L13_vs_L25_pc0_separable": bool(pc0_separable),
            "L13_vs_L25_cos_separable": bool(cos_separable),
            "two_regime_publication_grade": bool(pc0_separable and cos_separable),
        }
        print(
            f"\nL13 vs L25 PC0 95% CIs: "
            f"L13=[{by['13']['pc0_evr_ci_low']:.3f}, {by['13']['pc0_evr_ci_high']:.3f}] "
            f"L25=[{by['25']['pc0_evr_ci_low']:.3f}, {by['25']['pc0_evr_ci_high']:.3f}]"
            f"  →  separable: {pc0_separable}"
        )
        print(
            f"L13 vs L25 |cos| 95% CIs: "
            f"L13=[{by['13']['mean_pairwise_abs_cos_ci_low']:.3f}, {by['13']['mean_pairwise_abs_cos_ci_high']:.3f}] "
            f"L25=[{by['25']['mean_pairwise_abs_cos_ci_low']:.3f}, {by['25']['mean_pairwise_abs_cos_ci_high']:.3f}]"
            f"  →  separable: {cos_separable}"
        )
        print(
            f"Two-regime publication-grade: "
            f"{out['verdict']['two_regime_publication_grade']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ wrote {args.out}")


if __name__ == "__main__":
    main()
