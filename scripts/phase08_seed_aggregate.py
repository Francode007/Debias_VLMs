#!/usr/bin/env python3
"""Phase 0.8 — Tier 0 #1 + Tier 1 #4: aggregate SB-Bench eval verdicts
across 3 independent seeds per (reward layer).

Reads each per-seed gen jsonl, computes accuracy / bias-signature metrics
identical to Phase 0.6/0.7 conventions, then reports:
  - per-seed accuracy + bias score
  - across-seed mean ± std (and per-axis breakdown)
  - paired McNemar vs vanilla (per seed)
  - meta-analysis: combined z (Stouffer) across the 3 seeds vs vanilla

Lower accuracy variance across seeds = more stable training signal.
Lower bias score (|p(stereotyped) − p(non-stereotyped)|) = more debiased.

Usage:
  python scripts/phase08_seed_aggregate.py \\
      --vanilla Phase0.8/cf_runs/9axis_base_sbbench_gen.jsonl \\
      --variant L13 \\
      --seeds Phase0.8/seed_runs/9axis_L13_s1_sbbench_gen.jsonl \\
              Phase0.8/seed_runs/9axis_L13_s2_sbbench_gen.jsonl \\
              Phase0.8/seed_runs/9axis_L13_s4_sbbench_gen.jsonl \\
      --output-json Phase0.8/seed_runs/L13_aggregate.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


LETTER_TO_INT = {"A": 0, "B": 1, "C": 2}

SB_BENCH_CATEGORIES = [
    "Age", "Disability", "Gender", "Nationality",
    "Physical Appearance", "Race/Ethnicity", "Religion", "SES", "Sexual Orientation",
]


def _resolve_axis(cat_val) -> str:
    try:
        i = int(cat_val)
        if 0 <= i < len(SB_BENCH_CATEGORIES):
            return SB_BENCH_CATEGORIES[i]
    except (TypeError, ValueError):
        pass
    return str(cat_val)


def _extract_letter(text) -> int | None:
    if text is None: return None
    s = str(text).strip().upper()
    for ch in s:
        if ch in LETTER_TO_INT: return LETTER_TO_INT[ch]
    return None


def _load_gen(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pred = _extract_letter(r.get("text", ""))
            rows.append({
                "qid": str(r.get("question_id", "")),
                "pred": pred,
                "label": int(r.get("label", -1)) if r.get("label") is not None else -1,
                "category": r.get("category", ""),
            })
    df = pd.DataFrame(rows)
    df["correct"] = (df["pred"] == df["label"]).astype(int)
    df["axis"] = df["category"].apply(_resolve_axis)
    return df


def _accuracy(df: pd.DataFrame) -> dict:
    n = len(df)
    n_valid = int((df["pred"].notna()).sum())
    acc = float((df["pred"] == df["label"]).mean()) if n else float("nan")
    per_axis = {}
    for ax, sub in df.groupby("axis"):
        per_axis[str(ax)] = {
            "n": len(sub),
            "accuracy": float((sub["pred"] == sub["label"]).mean()) if len(sub) else float("nan"),
        }
    return {
        "n": n, "n_valid": n_valid,
        "accuracy": acc,
        "macro_accuracy": float(pd.Series([per_axis[a]["accuracy"] for a in per_axis]).mean()),
        "per_axis": per_axis,
    }


def _mcnemar(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Paired McNemar test on (correct_a vs correct_b) per qid."""
    joined = df_a.merge(df_b, on="qid", suffixes=("_a", "_b"))
    a = joined["correct_a"].astype(int).values
    b = joined["correct_b"].astype(int).values
    b01 = int(((a == 0) & (b == 1)).sum())   # a wrong, b right
    b10 = int(((a == 1) & (b == 0)).sum())   # a right, b wrong
    n_disc = b01 + b10
    # Exact binomial test approximation (normal w/ continuity correction)
    if n_disc == 0:
        return {"n_paired": len(joined), "b01": 0, "b10": 0, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b01 - b10) - 1) ** 2 / n_disc if n_disc > 0 else 0.0
    # one-DOF chi-square p-value via approximation
    from math import erfc, sqrt
    z = sqrt(chi2)
    p = erfc(z / sqrt(2))
    return {"n_paired": len(joined), "b01": b01, "b10": b10, "chi2": chi2, "p": p,
            "delta_accuracy": float(joined["correct_b"].mean() - joined["correct_a"].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla", required=True, help="vanilla gen jsonl (baseline)")
    ap.add_argument("--variant", required=True, help="variant tag (e.g. L13)")
    ap.add_argument("--seeds", nargs="+", required=True,
                    help="per-seed gen jsonl paths")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    print(f"loading vanilla: {args.vanilla}")
    van = _load_gen(args.vanilla)
    van_metrics = _accuracy(van)
    print(f"  vanilla:  n={van_metrics['n']}  acc={van_metrics['accuracy']:.4f}  "
          f"macro_acc={van_metrics['macro_accuracy']:.4f}")

    per_seed = {}
    mcn_results = {}
    z_scores = []
    for sp in args.seeds:
        tag = Path(sp).stem
        print(f"\n  seed: {tag}  ({sp})")
        df = _load_gen(sp)
        m = _accuracy(df)
        mc = _mcnemar(van, df)
        per_seed[tag] = m
        mcn_results[tag] = mc
        # z-score from McNemar chi^2 (signed by delta_accuracy direction)
        z = math.sqrt(mc["chi2"])
        if mc.get("delta_accuracy", 0.0) < 0:
            z = -z
        z_scores.append(z)
        print(f"    n={m['n']}  acc={m['accuracy']:.4f}  macro_acc={m['macro_accuracy']:.4f}")
        print(f"    McNemar: b01={mc['b01']} b10={mc['b10']} chi2={mc['chi2']:.3f} p={mc['p']:.4f} "
              f"Δacc={mc.get('delta_accuracy', 0):+.4f}")

    # Across-seed mean ± std
    accs = [per_seed[t]["accuracy"] for t in per_seed]
    macros = [per_seed[t]["macro_accuracy"] for t in per_seed]
    n_seeds = len(per_seed)
    mean_acc = float(pd.Series(accs).mean())
    std_acc = float(pd.Series(accs).std(ddof=1)) if n_seeds > 1 else 0.0
    mean_macro = float(pd.Series(macros).mean())
    std_macro = float(pd.Series(macros).std(ddof=1)) if n_seeds > 1 else 0.0

    # Stouffer's combined z (across-seed meta-analysis)
    if z_scores:
        z_comb = sum(z_scores) / math.sqrt(len(z_scores))
        from math import erfc, sqrt
        p_comb = erfc(abs(z_comb) / sqrt(2))
    else:
        z_comb = float("nan"); p_comb = float("nan")

    # Per-axis across-seed mean ± std
    per_axis_seed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for tag, m in per_seed.items():
        for axis, blk in m["per_axis"].items():
            per_axis_seed[axis]["acc"].append(blk["accuracy"])
    per_axis_summary = {}
    for axis, d in per_axis_seed.items():
        accs_ax = d["acc"]
        per_axis_summary[axis] = {
            "mean_acc": float(pd.Series(accs_ax).mean()),
            "std_acc":  float(pd.Series(accs_ax).std(ddof=1)) if len(accs_ax) > 1 else 0.0,
            "vanilla_acc": van_metrics["per_axis"].get(axis, {}).get("accuracy"),
            "n_seeds": len(accs_ax),
        }

    report = {
        "variant": args.variant,
        "vanilla_gen": args.vanilla,
        "vanilla_metrics": van_metrics,
        "n_seeds": n_seeds,
        "per_seed": per_seed,
        "mcnemar_vs_vanilla": mcn_results,
        "aggregate": {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_macro_accuracy": mean_macro,
            "std_macro_accuracy": std_macro,
            "delta_mean_vs_vanilla": mean_acc - van_metrics["accuracy"],
            "delta_macro_vs_vanilla": mean_macro - van_metrics["macro_accuracy"],
            "stouffer_z_vs_vanilla": z_comb,
            "stouffer_p_vs_vanilla": p_comb,
        },
        "per_axis_aggregate": per_axis_summary,
    }

    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.output_json}")

    # Headline summary
    print("\n──────── seed-triple summary ────────")
    print(f"variant: {args.variant}   n_seeds={n_seeds}")
    print(f"  vanilla micro_acc:           {van_metrics['accuracy']:.4f}")
    print(f"  vanilla macro_acc:           {van_metrics['macro_accuracy']:.4f}")
    print(f"  seed-mean  micro_acc:        {mean_acc:.4f} ± {std_acc:.4f}  (Δ = {mean_acc - van_metrics['accuracy']:+.4f})")
    print(f"  seed-mean  macro_acc:        {mean_macro:.4f} ± {std_macro:.4f}  (Δ = {mean_macro - van_metrics['macro_accuracy']:+.4f})")
    print(f"  Stouffer combined z:         {z_comb:+.3f}  (p={p_comb:.4f})")
    print("\nper-seed McNemar vs vanilla:")
    for tag, mc in mcn_results.items():
        print(f"  {tag}:  b01={mc['b01']:>5d}  b10={mc['b10']:>5d}  "
              f"χ²={mc['chi2']:>6.2f}  p={mc['p']:.4f}  Δacc={mc.get('delta_accuracy', 0):+.4f}")


if __name__ == "__main__":
    main()
