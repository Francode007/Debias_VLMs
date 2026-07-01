#!/usr/bin/env python3
"""Phase 0.9.5 E2 — Probe-direction convergence diagnostic.

Strategic-plan spec: Phase0.9_Strategic_Plan.md §3.2
Critical-review motivation: Phase0.9_Critical_Review.md §7

Question
────────
With n records and D=2048 features, sklearn's L2-regularised
LogisticRegression (C=1.0) finds the minimum-‖w‖² solution from a
1148-dimensional null-space. Does the recovered direction `w_n`
converge to a stable target as n grows? Specifically, is the shipped
`bias_aligned` probe fit on n=600 disambig records close to the
direction we would have recovered at n=3000?

Method (CPU-only Phase A — this script)
───────────────────────────────────────
Uses the existing local HS cache:
    Phase0.8/a3_results/local_cache/base_probe_hs.npz
which holds (900, 37, 2048) HS at `post_letter`. After filtering to
disambig records (bias_aligned label defined), n_max in Phase A is
~600 (300 ambig records have None label by construction).

For each layer L ∈ {13, 17, 21, 25, 29, 33}:
  For each n ∈ {150, 300, 450, 600}:
    For each seed s ∈ {0, 1, 2, 3, 4}:
      • Random subsample n of the ~600 usable records (without replacement)
      • Fit sklearn LogisticRegression(C=1.0, class_weight='balanced')
      • Unit-normalise coef → w(L, n, s)

Outputs three diagnostic numbers per layer:
  • seed_cos(L, n)   = mean cos between (s_i, s_j) at fixed n
                      (does the n records always give the same direction?)
  • size_cos(L, n)   = cos(mean_seed w(L, n, ·), mean_seed w(L, 600, ·))
                      (does the n-sample direction match the 600-sample direction?)
  • shipped_cos(L)   = cos(mean_seed w(L, 600, ·), w_shipped(L))
                      sanity that the bundled probe is reproducible from cache.

Decision rule (Phase0.9_Strategic_Plan.md §3.2):
  • size_cos(L13, n=300) ≥ 0.95 AND shipped_cos(L13) ≥ 0.99 →
        existing n=600 probe is converged on cache; n=900 (claim in spec)
        likely also converged. Close E2, do NOT trigger E7.
  • size_cos(L13, n=300) < 0.90 →
        probe is under-determined; trigger E7 (Phase B Modal extract,
        rebuild bundle at n>=2000).
  • 0.90 ≤ size_cos(L13, n=300) < 0.95 →
        ambiguous; recommend Phase B Modal extract (~$1) to settle
        with a real n_max ≈ 3000.

Output: Phase0.9/probe_convergence/diagnostic.json

Phase B (separate, Modal): re-extract HS at `max_per_cell=100` for a
n_max ≈ 3000 ground truth, only if Phase A is in the 0.90-0.95 band.

Usage
─────
    python scripts/phase09_probe_convergence.py

CPU-only. No Modal dependency. Reads cached HS + shipped probes from
the local repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Use the canonical label fn from probe_layers so the labels are byte-identical
# to the ones the shipped probe was trained on.
from modules.evaluation.probe_layers import _is_bias_aligned  # type: ignore

LAYERS = [13, 17, 21, 25, 29, 33]
SIZES = [150, 300, 450, 600]
SEEDS = [0, 1, 2, 3, 4]
LOGREG_C = 1.0
MAX_ITER = 1000

# Decision thresholds (mirror strategic plan §3.2).
THRESHOLD_PASS = 0.95
THRESHOLD_TRIGGER_E7 = 0.90


def fit_unit_w(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a single `bias_aligned` LR on (X, y) and return unit-normed coef."""
    clf = LogisticRegression(
        C=LOGREG_C, max_iter=MAX_ITER, class_weight="balanced", n_jobs=1,
    )
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float64)
    nrm = float(np.linalg.norm(w))
    if nrm <= 0:
        raise ValueError("zero-norm coef")
    return w / nrm


def mean_pairwise_cos(W: np.ndarray) -> float:
    """Mean pairwise cosine over rows of W (rows are unit-normed vectors)."""
    K = W.shape[0]
    if K < 2:
        return float("nan")
    cs = []
    for i in range(K):
        for j in range(i + 1, K):
            cs.append(float(np.dot(W[i], W[j])))
    return float(np.mean(cs))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "Phase0.8/a3_results/local_cache/base_probe_hs.npz",
        help="HS cache from probe_layers.py (VLBias subsample at post_letter).",
    )
    p.add_argument(
        "--shipped-l13",
        type=Path,
        default=ROOT / "Phase0.8/a3_results/base_L13_probe_weights_biasA.npz",
        help="Shipped L13 bias_aligned probe weights (sanity reference).",
    )
    p.add_argument(
        "--shipped-bundle-raw",
        type=Path,
        default=ROOT / "Phase0.9/ensemble_bundle_raw",
        help="Dir with base_L{17,21,25,29,33}_probe_weights_biasA.npz.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "Phase0.9/probe_convergence/diagnostic.json",
    )
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ─── 1. Load cache + labels ──────────────────────────────────────
    print(f"▶ Loading HS cache: {args.cache}")
    d = np.load(args.cache, allow_pickle=True)
    hs = d["hs_primary"]              # (N, L_all, D)
    kept = [r.item() if hasattr(r, "item") else r for r in d["kept_primary"]]
    N_total, L_all, D = hs.shape
    print(f"  shape: N={N_total}  L_all={L_all}  D={D}")
    assert max(LAYERS) < L_all, f"layer {max(LAYERS)} out of cache range {L_all}"

    labels = np.array(
        [_is_bias_aligned(r) for r in kept], dtype=object
    )
    mask = np.array([y is not None for y in labels])
    n_usable = int(mask.sum())
    print(f"  usable bias_aligned labels: {n_usable} / {N_total}")
    y_all = np.array([y for y in labels if y is not None], dtype=np.int64)
    print(f"  class balance: 1={int((y_all == 1).sum())}  0={int((y_all == 0).sum())}")

    # Cap SIZES at the available n_usable.
    sizes = [n for n in SIZES if n <= n_usable]
    if not sizes or sizes[-1] < n_usable:
        sizes.append(n_usable)
    print(f"  subsample sizes (capped to usable): {sizes}")

    hs_usable_all = hs[mask]          # (n_usable, L_all, D)

    # ─── 2. Load shipped probes for sanity comparison ─────────────────
    shipped: dict[int, np.ndarray] = {}
    if args.shipped_l13.exists():
        d13 = np.load(args.shipped_l13, allow_pickle=True)
        w13 = d13["coef"].astype(np.float64)
        shipped[13] = w13 / np.linalg.norm(w13)
        print(f"  shipped L13: {args.shipped_l13.name}  ||coef||={np.linalg.norm(w13):.3f}")
    for L in (17, 21, 25, 29, 33):
        path = args.shipped_bundle_raw / f"base_L{L}_probe_weights_biasA.npz"
        if path.exists():
            di = np.load(path, allow_pickle=True)
            wi = di["coef"].astype(np.float64)
            shipped[L] = wi / np.linalg.norm(wi)

    # ─── 3. Fit + record per-(L, n, seed) ─────────────────────────────
    rng_master = np.random.default_rng(20260624)
    seed_perms: dict[int, np.ndarray] = {
        s: rng_master.permutation(n_usable) for s in SEEDS
    }

    per_layer: dict[int, dict] = {}
    for L in LAYERS:
        Xfull = hs_usable_all[:, L, :].astype(np.float64)  # (n_usable, D)
        print(f"\n── Layer L={L} ──")
        # weights[n][seed] = unit-normed coef
        weights: dict[int, list[np.ndarray]] = {n: [] for n in sizes}
        train_accs: dict[int, list[float]] = {n: [] for n in sizes}
        for n in sizes:
            for s in SEEDS:
                idx = seed_perms[s][:n]
                X = Xfull[idx]
                y = y_all[idx]
                if np.unique(y).size < 2:
                    # Edge case: subsample has only one class — skip
                    continue
                w = fit_unit_w(X, y)
                weights[n].append(w)
                # quick train acc as a stability signal
                clf = LogisticRegression(
                    C=LOGREG_C, max_iter=MAX_ITER, class_weight="balanced", n_jobs=1,
                )
                clf.fit(X, y)
                train_accs[n].append(float(clf.score(X, y)))
            print(
                f"  n={n:>4d}  fitted seeds={len(weights[n])}/{len(SEEDS)}  "
                f"mean_train_acc={np.mean(train_accs[n]) if train_accs[n] else float('nan'):.3f}"
            )

        # seed_cos(L, n) and size_cos(L, n) ----------------------------
        seed_cos = {}
        size_cos = {}
        mean_w_per_n: dict[int, np.ndarray] = {}
        for n in sizes:
            W = np.array(weights[n])              # (k_seeds, D)
            if W.size == 0:
                seed_cos[n] = float("nan")
                continue
            seed_cos[n] = mean_pairwise_cos(W)
            mw = W.mean(axis=0)
            mean_w_per_n[n] = mw / (np.linalg.norm(mw) + 1e-12)

        ref_n = sizes[-1]
        for n in sizes:
            if n in mean_w_per_n and ref_n in mean_w_per_n:
                size_cos[n] = float(
                    np.dot(mean_w_per_n[n], mean_w_per_n[ref_n])
                )
            else:
                size_cos[n] = float("nan")

        shipped_cos = float("nan")
        if L in shipped and ref_n in mean_w_per_n:
            shipped_cos = float(np.dot(mean_w_per_n[ref_n], shipped[L]))

        for n in sizes:
            print(
                f"    n={n:>4d}  seed_cos={seed_cos[n]:>6.3f}  "
                f"size_cos(vs n={ref_n})={size_cos[n]:>6.3f}"
            )
        if not np.isnan(shipped_cos):
            print(f"    shipped_cos(L{L}, n_max=600): {shipped_cos:.6f}")

        per_layer[L] = {
            "sizes": sizes,
            "seed_cos": {str(n): seed_cos[n] for n in sizes},
            "size_cos_vs_n_max": {str(n): size_cos[n] for n in sizes},
            "train_acc_per_n": {
                str(n): {
                    "mean": float(np.mean(train_accs[n])) if train_accs[n] else float("nan"),
                    "std": float(np.std(train_accs[n])) if len(train_accs[n]) > 1 else 0.0,
                }
                for n in sizes
            },
            "shipped_cos_vs_n_max": shipped_cos,
        }

    # ─── 4. Verdict (anchored on L13 per strategic plan §3.2) ─────────
    l13 = per_layer.get(13, {})
    l13_size_cos = l13.get("size_cos_vs_n_max", {})
    # The strategic-plan spec asks about cos(w_900, w_3000); the closest
    # proxy we have on the cache is cos(w_300, w_600) for L13.
    n_proxy = 300
    proxy_cos = l13_size_cos.get(str(n_proxy), float("nan"))
    if proxy_cos >= THRESHOLD_PASS:
        verdict = "phase_a_pass"
        decision = (
            f"size_cos(L13, n=300 vs n=600) = {proxy_cos:.3f} ≥ {THRESHOLD_PASS}. "
            "Probe direction is stable on the cache; doubling n from 300 to 600 "
            "did not materially rotate the L13 direction. Phase B (Modal extract) "
            "is NOT required. Phase 0.9.5 may proceed with the existing bundle."
        )
    elif proxy_cos < THRESHOLD_TRIGGER_E7:
        verdict = "phase_a_trigger_e7"
        decision = (
            f"size_cos(L13, n=300 vs n=600) = {proxy_cos:.3f} < {THRESHOLD_TRIGGER_E7}. "
            "Probe direction is still moving substantially with n on the cache; "
            "n=600 is likely under-determined. STRONGLY recommend triggering E7 "
            "(Phase B Modal extract at max_per_cell=100 → n≈3000, re-fit probes, "
            "rebuild ensemble bundle) before any downstream PPO ablations."
        )
    else:
        verdict = "phase_a_ambiguous"
        decision = (
            f"size_cos(L13, n=300 vs n=600) = {proxy_cos:.3f} in [0.90, 0.95). "
            "Inconclusive on the cache. Recommend Phase B Modal extract "
            "(max_per_cell=100, ~$1) to compute cos(w_600, w_3000) and settle "
            "whether the bundle should be rebuilt."
        )
    print("\n── Verdict ─────────────────────────────────────────────")
    print(f"  {verdict}")
    print(f"  {decision}")

    summary = {
        "spec": "Phase0.9_Strategic_Plan.md §3.2 (E2)",
        "method": "CPU-only Phase A on existing 600-record bias_aligned cache",
        "cache_path": str(args.cache),
        "cache_n_total": int(N_total),
        "n_usable_bias_aligned": n_usable,
        "label_class_balance": {
            "1": int((y_all == 1).sum()),
            "0": int((y_all == 0).sum()),
        },
        "logreg_C": LOGREG_C,
        "max_iter": MAX_ITER,
        "subsample_sizes": sizes,
        "seeds": SEEDS,
        "layers": LAYERS,
        "thresholds": {
            "pass": THRESHOLD_PASS,
            "trigger_e7": THRESHOLD_TRIGGER_E7,
        },
        "per_layer": {str(L): per_layer[L] for L in LAYERS},
        "primary_anchor_layer": 13,
        "primary_anchor_n_proxy": n_proxy,
        "primary_anchor_cos": proxy_cos,
        "verdict": verdict,
        "decision": decision,
        "phase_b_required": verdict != "phase_a_pass",
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Wrote {args.out}")


if __name__ == "__main__":
    main()
