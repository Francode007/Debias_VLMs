#!/usr/bin/env python3
"""Phase 0.8 Strategic Plan §3 — Wash-out diagnostic CLASSIFIER.

Reads the per-layer offline-reward JSONs produced by
`scripts/phase08_washout_score.sh` and emits:

  - per-(variant, layer) Δμ_corr = mean_reward[ans_correct=1] − mean_reward[ans_correct=0]
  - per-variant pattern classification {1,2,3} per the strategic plan §3:

    Pattern 1 (clean):     Δμ_full ≤ −0.30 σ_van at L13 AND ≤ −0.15 σ_van at every L≥17
    Pattern 2 (wash-out):  Δμ_full ≤ −0.30 σ_van at L13 AND |Δμ_full| < 0.10 σ_van at every L≥21
    Pattern 3 (proxy-hack):Δμ_full ≤ −0.30 σ_van at L13 AND Δμ_full > 0 at any L≥25

  σ_van is the per-layer std of mean_reward across cells in the vanilla run.

Output: Phase0.8/washout_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SWEEP_LAYERS = [1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35]
DEFAULT_VARIANTS = ["vanilla", "phase08_full", "phase08_corrOnly", "phase08_biasOnly"]


def load_summary(score_root: str, variant: str, layer: int) -> dict | None:
    """Locate the offline-reward JSON for (variant, layer) under either:
        <root>/L{layer}/<variant>__probe_L{layer}__offline_reward.json
    or  <root>/<variant>__probe_L{layer}__offline_reward.json
    """
    candidates = [
        os.path.join(score_root, f"L{layer}", f"{variant}__probe_L{layer}__offline_reward.json"),
        os.path.join(score_root,             f"{variant}__probe_L{layer}__offline_reward.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c) as f:
                return json.load(f)
    return None


def delta_mu_correct(summary: dict) -> Tuple[float, float]:
    """Return (Δμ_corr, σ_pooled) where:
       Δμ_corr = mean over cells with ans_correct=1 minus mean over cells with ans_correct=0
       σ_pooled = std across all cells (used for normalisation)
    Aggregates over all `cond` values in by_cell.
    """
    by_cell = summary.get("by_cell", {})
    pos, neg, allv = [], [], []
    for k, v in by_cell.items():
        if v.get("mean_reward") is None:
            continue
        m = float(v["mean_reward"])
        allv.append(m)
        # cell key format: "<cond>::<correctness>"; correctness in {"correct","incorrect"}
        if k.endswith("::correct"):
            pos.append(m)
        elif k.endswith("::incorrect"):
            neg.append(m)
    if not pos or not neg:
        return float("nan"), float("nan")
    return float(np.mean(pos) - np.mean(neg)), float(np.std(allv, ddof=0)) or 1e-9


def classify(deltas: Dict[int, float], sigma_van: Dict[int, float]) -> str:
    """Apply §3 decision rule on the *full* variant's per-layer Δμ.

    deltas    : layer → Δμ_corr(full)
    sigma_van : layer → σ_pooled(vanilla)
    """
    def _z(L):  # signed z-score in vanilla σ units
        d, s = deltas.get(L), sigma_van.get(L)
        if d is None or s is None or not np.isfinite(d) or not np.isfinite(s) or s == 0:
            return float("nan")
        return d / s

    z13 = _z(13)
    if not np.isfinite(z13) or z13 > -0.30:
        return "inconclusive (no L13 drop)"

    deep_layers = [L for L in (17, 21, 25, 29, 33, 35) if L in deltas]
    z_deep = {L: _z(L) for L in deep_layers}

    if any(np.isfinite(z_deep[L]) and z_deep[L] > 0 for L in deep_layers if L >= 25):
        return "Pattern 3 (proxy-hack)"

    if all(np.isfinite(z_deep[L]) and z_deep[L] <= -0.15 for L in deep_layers):
        return "Pattern 1 (clean)"

    deeper = [L for L in deep_layers if L >= 21]
    if deeper and all(np.isfinite(z_deep[L]) and abs(z_deep[L]) < 0.10 for L in deeper):
        return "Pattern 2 (wash-out)"

    return "intermediate (mixed signals)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--score-root", default="/mnt/data/phase08_washout",
                   help="Root dir containing the offline-reward JSONs (or its L<layer>/ subdirs).")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    p.add_argument("--layers", type=int, nargs="+", default=SWEEP_LAYERS)
    p.add_argument("--output-json", default="Phase0.8/washout_diagnostic.json")
    args = p.parse_args()

    table: Dict[str, Dict[int, dict]] = {}
    for var in args.variants:
        table[var] = {}
        for L in args.layers:
            summ = load_summary(args.score_root, var, L)
            if summ is None:
                table[var][L] = {"missing": True}
                continue
            d, s = delta_mu_correct(summ)
            table[var][L] = {
                "delta_mu_corr": d,
                "sigma_pooled": s,
                "n_samples": summ.get("num_samples_scored"),
            }

    sigma_van = {L: table.get("vanilla", {}).get(L, {}).get("sigma_pooled")
                 for L in args.layers}
    deltas_full = {L: table.get("phase08_full", {}).get(L, {}).get("delta_mu_corr")
                   for L in args.layers}
    pattern = classify(
        {L: deltas_full[L] for L in args.layers if isinstance(deltas_full.get(L), (int, float))},
        {L: sigma_van[L] for L in args.layers if isinstance(sigma_van.get(L), (int, float))},
    )

    out = {
        "variants": args.variants,
        "layers": args.layers,
        "by_variant": table,
        "verdict": {
            "phase08_full_pattern": pattern,
        },
    }
    Path(os.path.dirname(args.output_json) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["verdict"], indent=2))
    print(f"✅ Wrote {args.output_json}")

    # Pretty per-layer table
    print("\nLayer  " + "  ".join(f"{v:>16}" for v in args.variants))
    for L in args.layers:
        row = [f"L{L:>3}  "]
        for v in args.variants:
            cell = table[v].get(L, {})
            if cell.get("missing"):
                row.append(f"{'<missing>':>16}")
            else:
                d = cell.get("delta_mu_corr", float("nan"))
                row.append(f"{d:>16.4f}")
        print("".join(row))


if __name__ == "__main__":
    main()
