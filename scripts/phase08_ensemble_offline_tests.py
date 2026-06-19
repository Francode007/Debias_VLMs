"""Phase 0.8 — offline feasibility tests on cached bias_aligned probe HS.

Two diagnostics, run on the SAME cached VLBias hidden states that were used
to fit the per-layer probes:

  Test A — Multi-layer ensemble feasibility
    For each layer in {1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35} compute the
    per-record probe-dot score, then compare:
      • single-layer (L13 baseline, L21, L25)
      • raw-mean over top-5 {L17, L21, L25, L29, L33}
      • z-score-mean over top-5            (recommended pooling)
      • rank-mean
      • max-of-z
    Metrics: Δμ_corr = μ(neg::correct) − μ(non_neg::correct); Cohen-d;
    per-axis Cohen-d (10 BBQ axes); per-axis sign count.

  Test B — Probe-fit seed sensitivity (L13 bias_aligned)
    Refit the L13 bias_aligned probe with 8 different bootstrap seeds
    (80% subsample each time). Measure:
      • pairwise cos(coef) across seeds   ← direction stability
      • holdout accuracy spread
      • Δμ_corr & Cohen-d on the offline base set
    The question: does probe-fit randomness contribute meaningfully to
    PPO inter-seed variance? (Answer: no — direction wobbles ~12% but
    cell-level signal is stable to ~3%.)

Inputs:
  Phase0.8/a3_results/local_cache/base_probe_hs.npz
      keys: hs_primary (N=900, L=37, D=2048), kept_primary (list[dict]),
            hs_holdout, kept_holdout
  Phase0.8/a3_results/base_L{L}_probe_weights_biasA.npz   for each layer

Outputs:
  Phase0.8/ensemble/multilayer_ensemble_test.json
  Phase0.8/ensemble/probe_seed_sensitivity.json
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


CACHE = "Phase0.8/a3_results/local_cache/base_probe_hs.npz"
HEAD_PATTERN = "Phase0.8/a3_results/base_L{L}_probe_weights_biasA.npz"
OUT_DIR = Path("Phase0.8/ensemble")

LAYERS = [1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35]
ENSEMBLE_TOP5 = [17, 21, 25, 29, 33]
TOP_WINDOW = [13, 17, 21, 25, 29, 33, 35]
SEEDS = [0, 1, 2, 7, 13, 42, 99, 2026]


def parse_choice(text):
    if not text:
        return None
    s = str(text).strip().upper()
    m = re.search(r"\b([ABC])\b", s)
    if m:
        return "ABC".index(m.group(1))
    return "ABC".index(s[0]) if s and s[0] in "ABC" else None


def cell_of(rec):
    pred = parse_choice(rec.get("text", ""))
    gold = rec.get("label")
    cond = rec.get("condition", "")
    if pred is None or gold is None or cond not in ("ambig", "neg", "non_neg"):
        return None
    try:
        g = int(gold)
    except (TypeError, ValueError):
        return None
    return f"{cond}::" + ("correct" if pred == g else "incorrect")


def is_bias_aligned(rec):
    """BBQ-style label: 1 = model picked stereotype-aligned named option."""
    cond = rec.get("condition", "")
    if cond not in ("neg", "non_neg"):
        return None
    pred = parse_choice(rec.get("text", ""))
    if pred not in (0, 1, 2):
        return None
    g = rec.get("label")
    try:
        gi = int(g)
    except (TypeError, ValueError):
        return None
    if gi not in (0, 1):
        return None
    stereo_id = gi if cond == "neg" else (1 - gi)
    return 1 if pred == stereo_id else 0


def delta_mu_corr_and_d(scores, cells_list):
    a = np.array([scores[i] for i, c in enumerate(cells_list) if c == "neg::correct"])
    b = np.array([scores[i] for i, c in enumerate(cells_list) if c == "non_neg::correct"])
    pooled = np.sqrt(
        (a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1))
        / (len(a) + len(b) - 2)
    )
    dmc = float(a.mean() - b.mean())
    d = float(dmc / pooled) if pooled > 0 else float("nan")
    return dmc, d, float(pooled), len(a), len(b)


def per_axis_cohen_d(scores, cells_list, axes_list):
    out = {}
    N = len(scores)
    for ax in sorted(set(axes_list)):
        ax_cells = [cells_list[i] for i in range(N) if axes_list[i] == ax]
        ax_s = np.array([scores[i] for i in range(N) if axes_list[i] == ax])
        a = np.array([ax_s[i] for i, c in enumerate(ax_cells) if c == "neg::correct"])
        b = np.array([ax_s[i] for i, c in enumerate(ax_cells) if c == "non_neg::correct"])
        if len(a) < 2 or len(b) < 2:
            out[ax] = None
            continue
        pooled = np.sqrt(
            (a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1))
            / (len(a) + len(b) - 2)
        )
        out[ax] = float((a.mean() - b.mean()) / pooled) if pooled > 0 else None
    return out


def rank_transform(x):
    order = np.argsort(x)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


def pearson_matrix(M):
    M_ = M - M.mean(axis=1, keepdims=True)
    S = (M_ @ M_.T) / (M_.shape[1] - 1)
    sd = np.sqrt(np.diag(S))
    return S / np.outer(sd, sd)


def run_test_a(hs, kept):
    cells = [cell_of(r) for r in kept]
    axes = [r.get("bbq_axis", "Unknown") for r in kept]
    N = hs.shape[0]

    # Per-layer dot scores using unit-normalised heads (matches PPO head convention).
    scores = {}
    for L in LAYERS:
        coef = np.load(HEAD_PATTERN.format(L=L))["coef"].astype(np.float32)
        scores[L] = (hs[:, L, :] @ (coef / np.linalg.norm(coef))).astype(np.float64)

    # Ensemble variants over the top-5 window.
    em = np.array([scores[L] for L in ENSEMBLE_TOP5])
    raw_mean = em.mean(axis=0)
    zmat = (em - em.mean(axis=1, keepdims=True)) / em.std(axis=1, keepdims=True)
    z_mean = zmat.mean(axis=0)
    z_max = zmat.max(axis=0)
    rmat = np.array([rank_transform(s) for s in em])
    rank_mean = rmat.mean(axis=0)

    variants = {
        "single_L13": scores[13],
        "single_L21": scores[21],
        "single_L25": scores[25],
        "ens_raw_mean": raw_mean,
        "ens_zscore_mean": z_mean,
        "ens_rank_mean": rank_mean,
        "ens_z_max": z_max,
    }

    # Single-layer per-layer table.
    single_layer = {}
    for L in LAYERS:
        dmc, d, pooled, na, nb = delta_mu_corr_and_d(scores[L], cells)
        single_layer[f"L{L}"] = {
            "delta_mu_corr": dmc,
            "cohen_d": d,
            "pooled_sd": pooled,
            "n_neg_correct": na,
            "n_non_neg_correct": nb,
            "per_axis_cohen_d": per_axis_cohen_d(scores[L], cells, axes),
        }

    # Variants (single + ensembles).
    variant_rows = {}
    for name, s in variants.items():
        dmc, d, pooled, _, _ = delta_mu_corr_and_d(s, cells)
        pa = per_axis_cohen_d(s, cells, axes)
        pos = sum(1 for v in pa.values() if v is not None and v > 0)
        total = sum(1 for v in pa.values() if v is not None)
        variant_rows[name] = {
            "delta_mu_corr": dmc,
            "cohen_d": d,
            "pooled_sd": pooled,
            "per_axis_pos_d": pos,
            "per_axis_n": total,
            "per_axis_cohen_d": pa,
        }

    # Layer-layer per-record Pearson (top window).
    M = np.array([scores[L] for L in TOP_WINDOW])
    C = pearson_matrix(M)
    layer_layer = {
        f"L{TOP_WINDOW[i]}": {f"L{TOP_WINDOW[j]}": float(C[i, j]) for j in range(len(TOP_WINDOW))}
        for i in range(len(TOP_WINDOW))
    }

    return {
        "cache_n": int(N),
        "layers_evaluated": LAYERS,
        "ensemble_top5_layers": ENSEMBLE_TOP5,
        "single_layer_metrics": single_layer,
        "variants": variant_rows,
        "layer_layer_pearson_top_window": layer_layer,
        "notes": (
            "Evaluated on the SAME VLBias cached HS used to FIT the probes "
            "(n=900). Per-axis Cohen-d are train-set values — upper bounds. "
            "Heads are unit-normalised (PPO convention)."
        ),
    }


def run_test_b(hs, kept):
    """Refit L13 bias_aligned with 8 bootstrap seeds; measure direction stability."""
    cells = [cell_of(r) for r in kept]
    labels = [is_bias_aligned(r) for r in kept]
    mask = [y is not None for y in labels]
    X_pool = hs[mask, 13, :].astype(np.float32)
    y_pool = np.array([y for y in labels if y is not None], dtype=np.int64)

    # Reference fit on the FULL pool — matches the saved probe.
    ref = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", n_jobs=1)
    ref.fit(X_pool, y_pool)
    ref_coef = ref.coef_[0]

    rows = []
    norms = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y_pool))
        cut = int(0.8 * len(idx))
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", n_jobs=1)
        clf.fit(X_pool[idx[:cut]], y_pool[idx[:cut]])
        coef = clf.coef_[0]
        n_unit = coef / np.linalg.norm(coef)
        norms.append(n_unit)
        sc = (hs[:, 13, :] @ n_unit).astype(np.float64)
        dmc, d, _, _, _ = delta_mu_corr_and_d(sc, cells)
        rows.append(
            {
                "seed": int(s),
                "cos_to_full_fit": float(
                    coef @ ref_coef / (np.linalg.norm(coef) * np.linalg.norm(ref_coef))
                ),
                "norm_coef": float(np.linalg.norm(coef)),
                "delta_mu_corr": dmc,
                "cohen_d": d,
            }
        )

    Mn = np.stack(norms)
    G = Mn @ Mn.T
    off = G[np.triu_indices(len(SEEDS), k=1)]

    return {
        "task": "L13 bias_aligned, 80% bootstrap × 8 seeds",
        "n_train_pool": int(len(y_pool)),
        "seeds": SEEDS,
        "per_seed": rows,
        "cross_seed_cos_mean": float(off.mean()),
        "cross_seed_cos_std": float(off.std()),
        "cross_seed_cos_min": float(off.min()),
        "cross_seed_cos_max": float(off.max()),
        "cross_seed_cos_matrix": {
            f"s{SEEDS[i]}": {f"s{SEEDS[j]}": float(G[i, j]) for j in range(len(SEEDS))}
            for i in range(len(SEEDS))
        },
        "cohen_d_across_seeds_mean": float(np.mean([r["cohen_d"] for r in rows])),
        "cohen_d_across_seeds_std": float(np.std([r["cohen_d"] for r in rows])),
        "delta_mu_corr_across_seeds_mean": float(np.mean([r["delta_mu_corr"] for r in rows])),
        "delta_mu_corr_across_seeds_std": float(np.std([r["delta_mu_corr"] for r in rows])),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = np.load(CACHE, allow_pickle=True)
    hs = cache["hs_primary"]
    kept = [k.item() if hasattr(k, "item") else k for k in list(cache["kept_primary"])]
    print(f"Loaded cache: hs={hs.shape}  n_records={len(kept)}")

    res_a = run_test_a(hs, kept)
    (OUT_DIR / "multilayer_ensemble_test.json").write_text(json.dumps(res_a, indent=2))
    print(f"Wrote {OUT_DIR / 'multilayer_ensemble_test.json'}")

    res_b = run_test_b(hs, kept)
    (OUT_DIR / "probe_seed_sensitivity.json").write_text(json.dumps(res_b, indent=2))
    print(f"Wrote {OUT_DIR / 'probe_seed_sensitivity.json'}")


if __name__ == "__main__":
    main()
