"""Compare canonical (real/n=2916) vs alternative (test2/n=3747) SB-Bench test splits.

Statistical comparison of split quality:
  1. Per-category balance (entropy, KL to uniform, Gini).
  2. Polarity (negative-question / question_polarity) balance.
  3. Image / context characteristics.
  4. CF-pair coverage: how many (file_name, question_polarity) twins land in each split.

Both splits use seed=42, train_ratio=0.8 stratified by something (or random). We
need to inspect the indices vs the parent parquet.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path('/Users/f0s03xp/Debias_VLMs')
SHARDS_DIR = ROOT / 'sb_bench_data' / 'data'

# Split-indices JSONs we have on the volume — pull them locally if not cached.
import modal

VOLUME = modal.Volume.from_name('debias-vlm-persistent-storage')


def load_split(name: str) -> dict:
    cache = ROOT / 'scratch' / f'_{name}'
    if cache.exists():
        return json.loads(cache.read_text())
    buf = b''
    for c in VOLUME.read_file(name):
        buf += c
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(buf)
    return json.loads(buf.decode())


def load_parent(prefix: str) -> pd.DataFrame:
    shards = sorted(SHARDS_DIR.glob(f'{prefix}-*.parquet'))
    if not shards:
        raise FileNotFoundError(f'no shards for {prefix}')
    dfs = [pd.read_parquet(s) for s in shards]
    return pd.concat(dfs, ignore_index=True)


def entropy(counts) -> float:
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def kl_to_uniform(counts) -> float:
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    k = len(p)
    q = 1.0 / k
    p_safe = np.where(p > 0, p, 1e-12)
    return float((p * np.log(p_safe / q)).sum())


def gini(counts) -> float:
    x = np.sort(np.asarray(counts, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def describe_split(label: str, df: pd.DataFrame, indices: list[int]) -> dict:
    sub = df.iloc[indices]
    n = len(sub)

    cat_counts = sub['category'].value_counts().sort_index()
    cat_props = (cat_counts / cat_counts.sum() * 100).round(2)
    pol_counts = sub['question_polarity'].value_counts().to_dict() if 'question_polarity' in sub.columns else {}

    H_cat = entropy(cat_counts.values)
    H_max = np.log(len(cat_counts))
    H_norm = H_cat / H_max if H_max > 0 else 0.0

    KL_uniform = kl_to_uniform(cat_counts.values)
    G = gini(cat_counts.values)
    chi2_obs = ((cat_counts.values - cat_counts.sum() / len(cat_counts)) ** 2 / (cat_counts.sum() / len(cat_counts))).sum()

    # CF pair coverage: pair = (file_name, context); each pair has both polarities.
    if {'file_name', 'context', 'question_polarity'}.issubset(sub.columns):
        pair_key = list(zip(sub['file_name'].astype(str), sub['context'].astype(str)))
        pair_counter = Counter(pair_key)
        n_paired = sum(1 for _, c in pair_counter.items() if c == 2)
        n_singletons = sum(1 for _, c in pair_counter.items() if c == 1)
        pair_coverage = (2 * n_paired) / n if n > 0 else 0.0
    else:
        n_paired = n_singletons = 0
        pair_coverage = 0.0

    print(f"\n=== {label} (n={n}) ===")
    print(f"  Per-category counts:")
    for c, cnt in cat_counts.items():
        print(f"    cat {c:>2}: {cnt:>5d}  ({cat_props[c]:>5.2f}%)")
    print(f"  Polarity balance: {pol_counts}")
    print(f"  Entropy:        {H_cat:.4f} (uniform: {H_max:.4f}, normalized: {H_norm:.4f})")
    print(f"  KL→uniform:     {KL_uniform:.4f}  (lower = more balanced)")
    print(f"  Gini:           {G:.4f}  (lower = more balanced)")
    print(f"  Chi² vs uniform:{chi2_obs:.2f}  (df={len(cat_counts)-1})")
    print(f"  CF pairs:       {n_paired} pairs ({2*n_paired} items, {pair_coverage:.1%} of split)")
    print(f"  Singletons:     {n_singletons}")

    return {
        'n': n,
        'cat_counts': cat_counts.to_dict(),
        'cat_props': cat_props.to_dict(),
        'pol_counts': pol_counts,
        'entropy': H_cat,
        'entropy_normalized': H_norm,
        'kl_uniform': KL_uniform,
        'gini': G,
        'chi2_uniform': chi2_obs,
        'n_paired': n_paired,
        'n_singletons': n_singletons,
        'pair_coverage': pair_coverage,
    }


def main():
    print("Loading split indices and parent parquets...")
    canonical = load_split('split_indices.json')
    test2 = load_split('split_indices_9axis.json')

    print(f"  canonical: total={canonical['total']}, train={len(canonical['train_indices'])}, test={len(canonical['test_indices'])}")
    print(f"  test2:     total={test2['total']}, train={len(test2['train_indices'])}, test={len(test2['test_indices'])}")

    df_real = load_parent('real')      # 14578 rows
    df_test2 = load_parent('test2')    # 18732 rows
    print(f"  parent real:  {len(df_real)} rows")
    print(f"  parent test2: {len(df_test2)} rows")

    assert canonical['total'] == len(df_real), f"canonical/{len(df_real)}"
    assert test2['total'] == len(df_test2), f"test2/{len(df_test2)}"

    res_canon = describe_split('CANONICAL test (split_indices.json over `real`)',
                               df_real, canonical['test_indices'])
    res_t2 = describe_split('TEST2 test (split_indices_9axis.json over `test2`)',
                            df_test2, test2['test_indices'])

    # Cross-comparison summary
    print("\n══ HEAD-TO-HEAD ══")
    rows = [
        ("split_size",      res_canon['n'],                    res_t2['n']),
        ("entropy_norm",    res_canon['entropy_normalized'],   res_t2['entropy_normalized']),
        ("kl_uniform↓",     res_canon['kl_uniform'],           res_t2['kl_uniform']),
        ("gini↓",           res_canon['gini'],                 res_t2['gini']),
        ("chi2↓",           res_canon['chi2_uniform'],         res_t2['chi2_uniform']),
        ("CF_pairs",        res_canon['n_paired'],             res_t2['n_paired']),
        ("CF_coverage",     res_canon['pair_coverage'],        res_t2['pair_coverage']),
    ]
    print(f"  {'metric':<18} {'canonical':>14}    {'test2':>14}")
    for name, a, b in rows:
        if isinstance(a, float):
            print(f"  {name:<18} {a:>14.4f}    {b:>14.4f}")
        else:
            print(f"  {name:<18} {a:>14d}    {b:>14d}")

    # ── Verdict ──
    print("\n══ VERDICT ══")
    cat_a = np.array(list(sorted(res_canon['cat_counts'].items())))
    cat_b = np.array(list(sorted(res_t2['cat_counts'].items())))

    if res_canon['entropy_normalized'] > res_t2['entropy_normalized']:
        print(f"  • canonical is MORE BALANCED across 9 axes "
              f"(H_norm {res_canon['entropy_normalized']:.4f} vs {res_t2['entropy_normalized']:.4f}).")
    else:
        print(f"  • test2 is more balanced (H_norm {res_t2['entropy_normalized']:.4f} vs {res_canon['entropy_normalized']:.4f}).")

    if res_canon['kl_uniform'] < res_t2['kl_uniform']:
        print(f"  • canonical KL→uniform {res_canon['kl_uniform']:.3f} < test2 {res_t2['kl_uniform']:.3f}: canonical closer to uniform.")
    else:
        print(f"  • test2 KL→uniform {res_t2['kl_uniform']:.3f} < canonical {res_canon['kl_uniform']:.3f}: test2 closer to uniform.")

    if res_canon['n_paired'] > res_t2['n_paired']:
        print(f"  • canonical has MORE CF pairs ({res_canon['n_paired']} vs {res_t2['n_paired']}).")
    elif res_canon['n_paired'] < res_t2['n_paired']:
        print(f"  • test2 has more CF pairs ({res_t2['n_paired']} vs {res_canon['n_paired']}).")
    else:
        print(f"  • CF pair counts equal.")


if __name__ == '__main__':
    main()
