#!/usr/bin/env python3
"""Phase 0.9 R3 step 1 — Build the multi-layer ensemble probe bundle.

Loads the 5 per-layer `base_L{17,21,25,29,33}_probe_weights_biasA.npz`
files (each = single `bias_aligned` LogReg coef vector trained on n=600
disambig VLBias records at that layer), unit-normalises them, and
computes per-layer offline μ/σ on the SB-Bench-9axis HS cache so the
PPO trainer can z-score normalise on the fly during reward
computation.

Output bundle:
    Phase0.9/ensemble_bundle/
        L17.pth, L21.pth, L25.pth, L29.pth, L33.pth
        ensemble_metadata.json

Each .pth file contains:
    {"weight": torch.Tensor of shape (1, 2048)}  (unit-normed)

The bundle is then pushed to the Modal volume at
    /generated_heads_probe_L17-33_zmean_base_biasA/
where the new ensemble-aware drm_loader (R3 step 2) will consume it.

Pre-committed window per Phase0.8_Strategic_Plan.md §3bis + R3 user choice:
    {L17, L21, L25, L29, L33}  with pool='zmean'

Usage:
    python scripts/phase09_build_ensemble_bundle.py \
        --raw-dir Phase0.9/ensemble_bundle_raw \
        --hs-cache Phase0.8/a3_results/local_cache/sbbench9_probe_hs.npz \
        --out-dir Phase0.9/ensemble_bundle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
WINDOW = [17, 21, 25, 29, 33]
POOL = "zmean"  # per-layer z-score then mean


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "Phase0.9/ensemble_bundle_raw",
                   help="dir holding base_L{N}_probe_weights_biasA.npz files")
    p.add_argument("--hs-cache", type=Path,
                   default=ROOT / "Phase0.8/a3_results/local_cache/sbbench9_probe_hs.npz",
                   help="HS cache for offline μ/σ calibration "
                        "(use SB-Bench to match the PPO training distribution)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "Phase0.9/ensemble_bundle")
    p.add_argument("--variant", default="base")
    p.add_argument("--task", default="bias_aligned")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Load per-layer probe coef vectors ────────────────────────────
    probes: dict[int, np.ndarray] = {}
    src_paths: dict[int, str] = {}
    for L in WINDOW:
        path = args.raw_dir / f"{args.variant}_L{L}_probe_weights_biasA.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing probe weights: {path}")
        d = np.load(path, allow_pickle=True)
        coef = d["coef"].astype(np.float64)            # (D,)
        assert coef.shape == (2048,), f"L{L}: unexpected coef shape {coef.shape}"
        nrm = np.linalg.norm(coef)
        if nrm <= 0:
            raise ValueError(f"L{L}: probe coef has zero norm")
        probes[L] = coef / nrm                          # unit-normed (D,)
        src_paths[L] = str(path)
        layer_idx = int(d["layer_idx"]) if "layer_idx" in d.files else L
        assert layer_idx == L, f"L{L} probe has layer_idx={layer_idx} (mismatch)"
        print(f"  L{L}: loaded {path.name}  ||coef||={nrm:.4f}  "
              f"normed_norm={np.linalg.norm(probes[L]):.6f}")

    # ─── 2. Compute per-layer offline μ/σ on the calibration HS cache ───
    print(f"\nLoading calibration HS cache: {args.hs_cache}")
    cache = np.load(args.hs_cache, allow_pickle=True)
    hs = cache["hs_primary"]          # (N, L_all, D) float32
    N, L_all, D = hs.shape
    print(f"  N={N}  L_all={L_all}  D={D}")
    assert D == 2048, f"hidden dim mismatch: {D} != 2048"
    assert L_all >= max(WINDOW) + 1, f"cache only has {L_all} layers, need >= {max(WINDOW)+1}"

    per_layer_mu: list[float] = []
    per_layer_sigma: list[float] = []
    print(f"\n=== Per-layer reward scale (over all {N} cached records) ===")
    print(f"  {'L':>3}  {'mean(r)':>10}  {'std(r)':>10}  {'min':>10}  {'max':>10}  {'p1':>8}  {'p99':>8}")
    for L in WINDOW:
        w = probes[L]                                   # unit-normed (D,)
        # r[i] = dot(w, hs[i, L, :])  — matches the PPO reward formula
        # exactly (in training, h is at post-letter token; cache stores post-letter HS).
        r = hs[:, L, :].astype(np.float64) @ w           # (N,)
        mu = float(r.mean())
        sd = float(r.std(ddof=1))                       # sample sd for downstream z-score
        per_layer_mu.append(mu)
        per_layer_sigma.append(sd)
        print(f"  {L:>3}  {mu:>10.4f}  {sd:>10.4f}  "
              f"{r.min():>10.4f}  {r.max():>10.4f}  "
              f"{np.percentile(r, 1):>8.3f}  {np.percentile(r, 99):>8.3f}")

    # ─── 3. Write per-layer .pth files (1, D) bf16 tensors ──────────────
    print(f"\nWriting bundle to: {args.out_dir}")
    for L in WINDOW:
        w = torch.from_numpy(probes[L]).unsqueeze(0).to(torch.float32)  # (1, D) float32 for archival
        out_path = args.out_dir / f"L{L}.pth"
        torch.save({"weight": w}, out_path)
        print(f"  wrote {out_path.name}  shape={tuple(w.shape)}  dtype={w.dtype}")

    # ─── 4. Write ensemble_metadata.json ────────────────────────────────
    metadata = {
        "schema_version": "1.0",
        "layers": WINDOW,
        "pool": POOL,
        "per_layer_mu": per_layer_mu,
        "per_layer_sigma": per_layer_sigma,
        "hidden_dim": int(D),
        "variant": args.variant,
        "task": args.task,
        "token_position": "post_letter",
        "reward_base": "Qwen/Qwen2.5-VL-3B-Instruct",
        "normalise": True,                  # probes are unit-normed
        "intercept_dropped": True,          # no intercept term in reward
        "calibration_cache": str(args.hs_cache.name),
        "calibration_n": int(N),
        "calibration_dataset": "sb_bench_9axis"
        if "sbbench" in args.hs_cache.name else "vlbias",
        "probe_sources": {f"L{L}": Path(src_paths[L]).name for L in WINDOW},
        "provenance_notes": (
            "5-layer ensemble probe bundle for Phase 0.9 R3 (Action 1 multi-layer "
            "ensemble reward). Probes were originally fit per-layer on n=600 VLBias "
            "disambig records by phase08_layer_probe (probe_layers.py P3 logistic "
            "regression). Per-layer μ/σ computed offline on the SB-Bench-9axis HS "
            "cache to match the PPO training distribution; the trainer's reward "
            "computation will apply z-normalisation as "
            "z_L = (proj_L - μ_L) / σ_L, then pool by mean over the 5 layers, then "
            "negate (consistent with bias_aligned reward sign convention)."
        ),
    }
    meta_path = args.out_dir / "ensemble_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  wrote {meta_path.name}")

    print(f"\n✅ Ensemble bundle complete at {args.out_dir}")
    print(f"   Layers: {WINDOW}")
    print(f"   Pool:   {POOL}")
    print(f"   Calibration: {metadata['calibration_dataset']} (n={N})")


if __name__ == "__main__":
    main()
