#!/usr/bin/env python3
"""R1 extra: compute (a) per-draw separability proportions p(L25 > L_i) and
(b) regime-vs-regime separability across all layer pairs.

Reuses the bootstrap loop from rank1_bias_geometry_bootstrap.py but stores
the full B-vector of per-draw PC0 evr and mean |cos| per layer, so we can
compute joint statistics.

Output: Phase0.8/a3_results/rank1_bias_geometry_bootstrap_full.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Reuse helpers from the sibling bootstrap script
sys.path.insert(0, str(ROOT / "scripts"))
from rank1_bias_geometry_bootstrap import (  # type: ignore
    fit_axis_probe_unit, pca_stats, LAYERS, RNG_SEED, CACHE, N_FOLDS, C_REG,
)
from modules.evaluation.probe_layers import _is_bias_aligned  # type: ignore

DEFAULT_B = 200
OUT = ROOT / "Phase0.8/a3_results/rank1_bias_geometry_bootstrap_full.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-B", "--n-bootstrap", type=int, default=DEFAULT_B)
    p.add_argument("--seed", type=int, default=RNG_SEED)
    p.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    print(f"Loading cache: {args.cache}")
    d = np.load(args.cache, allow_pickle=True)
    hs = d["hs_primary"]
    meta = d["kept_primary"]
    N, L_total, D = hs.shape
    print(f"  N={N}  L_total={L_total}  D={D}")

    labels = np.array([_is_bias_aligned(r) for r in meta], dtype=object)
    axes = np.array([r.get("bbq_axis") for r in meta])
    keep = np.array([l is not None for l in labels])
    y_all = np.array([int(l) if l is not None else -1 for l in labels])
    unique_axes = sorted(set(axes[keep]))

    per_axis: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ax in unique_axes:
        mask = (axes == ax) & keep
        per_axis[ax] = (hs[mask].astype(np.float32), y_all[mask].astype(int))

    rng = np.random.default_rng(args.seed)
    B = args.n_bootstrap

    out: dict = {
        "layers": list(args.layers),
        "axes": unique_axes,
        "n_folds": N_FOLDS,
        "C_reg": C_REG,
        "n_bootstrap": B,
        "seed": int(args.seed),
        "per_axis_n": {ax: int(per_axis[ax][1].shape[0]) for ax in unique_axes},
        "draws": {},   # layer → {pc0: list, abs_cos: list}
    }

    # Generate one shared resample index per draw per axis so the draws are
    # comparable across layers (same records resampled across layers).
    print(f"\nRunning B={B} bootstrap draws, all layers each draw, "
          f"shared resample index across layers ...")
    layer_pc0: dict[int, list[float]] = {L: [] for L in args.layers}
    layer_cos: dict[int, list[float]] = {L: [] for L in args.layers}

    # Pre-compute per-axis per-layer X arrays to save the float64 cast per draw
    per_axis_per_layer = {
        ax: {L: per_axis[ax][0][:, L, :].astype(np.float64) for L in args.layers}
        for ax in unique_axes
    }

    t_start = time.time()
    n_axes = len(unique_axes)
    print(f"  n_axes = {n_axes}")
    for b in range(B):
        # Per-axis resample indices (shared across layers in this draw)
        resample_idxs: dict[str, np.ndarray] = {}
        for ax in unique_axes:
            n_ax = per_axis[ax][1].shape[0]
            resample_idxs[ax] = rng.integers(0, n_ax, size=n_ax)

        for L in args.layers:
            W = np.zeros((n_axes, D), dtype=np.float64)
            degenerate = False
            for i, ax in enumerate(unique_axes):
                idx = resample_idxs[ax]
                X_L = per_axis_per_layer[ax][L]
                _, y_ax = per_axis[ax]
                Xb = X_L[idx]
                yb = y_ax[idx]
                if len(np.unique(yb)) < 2:
                    degenerate = True
                    break
                W[i] = fit_axis_probe_unit(Xb, yb, seed=args.seed + b)
            if degenerate:
                continue
            evr0, mean_abs = pca_stats(W)
            layer_pc0[L].append(evr0)
            layer_cos[L].append(mean_abs)
        if (b + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (b + 1) / elapsed
            eta = (B - b - 1) / rate
            print(f"  draw {b+1}/{B}  [{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    # Save full per-draw vectors
    for L in args.layers:
        out["draws"][str(L)] = {
            "pc0_evr": layer_pc0[L],
            "mean_pairwise_abs_cos": layer_cos[L],
            "n_draws": int(len(layer_pc0[L])),
        }

    # Compute pair-wise separability proportions for both metrics
    print("\n=== Pair-wise per-draw separability: P(L_hi metric > L_lo metric) ===")
    print(f"  (B = {B}, N comparable draws = min across pair)")

    def pair_proportion(lo: int, hi: int, metric: dict[int, list[float]]) -> float:
        # Use the per-draw paired vectors (we shared the resample index across layers
        # so the draws are paired). Take the min length to handle degenerate skips.
        v_lo = np.array(metric[lo])
        v_hi = np.array(metric[hi])
        n = min(len(v_lo), len(v_hi))
        return float(np.mean(v_hi[:n] > v_lo[:n]))

    regime1 = [L for L in args.layers if L in (5, 9, 13, 17, 21)]
    regime2 = [L for L in args.layers if L in (25, 29, 33, 35)]
    print(f"  Regime 1 (mid)  = {regime1}")
    print(f"  Regime 2 (deep) = {regime2}")

    pairwise: dict = {"pc0_evr": {}, "mean_pairwise_abs_cos": {}}
    for lo in regime1:
        for hi in regime2:
            p_pc0 = pair_proportion(lo, hi, layer_pc0)
            p_cos = pair_proportion(lo, hi, layer_cos)
            pairwise["pc0_evr"][f"L{lo}_vs_L{hi}"] = p_pc0
            pairwise["mean_pairwise_abs_cos"][f"L{lo}_vs_L{hi}"] = p_cos
            print(f"  L{lo} < L{hi}:  P(PC0 hi > lo) = {p_pc0:.3f}   "
                  f"P(|cos| hi > lo) = {p_cos:.3f}")

    out["pairwise_proportions"] = pairwise

    # Critical pair (L13 vs L25, headline)
    head = {
        "L13_vs_L25_p_pc0": pair_proportion(13, 25, layer_pc0),
        "L13_vs_L25_p_cos": pair_proportion(13, 25, layer_cos),
    }
    print(f"\nHeadline (paired-draw): "
          f"P(L25 PC0 > L13 PC0) = {head['L13_vs_L25_p_pc0']:.3f}")
    print(f"Headline (paired-draw): "
          f"P(L25 |cos| > L13 |cos|) = {head['L13_vs_L25_p_cos']:.3f}")
    out["headline"] = head

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ wrote {args.out}")


if __name__ == "__main__":
    main()
