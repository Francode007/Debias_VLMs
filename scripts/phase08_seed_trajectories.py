#!/usr/bin/env python3
"""Compare training trajectories across phase08_2k seeds (production + 1/2/3/4).

Reads Phase0.8/seed_diag/{tag}_metrics.jsonl, emits a side-by-side trajectory
table at fixed step bins, plus aggregate summaries highlighting where seed 2
diverges from the others.
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

DIAG_DIR = Path('Phase0.8/seed_diag')
RUNS = ['prod_s42', 's1', 's2', 's3', 's4']

# Series we care about (aliased -> printed name). Pulled from prod_s42 row 0.
SERIES = [
    'binary_accuracy',
    'reward_dense_mean',
    'reward_task',
    'reward_correctness_mean',
    'reward_bias_aligned_mean',
    'kl',
    'pg_loss',
    'v_loss',
    'ratio_mean',
    'mean_abs_logprob_diff',
    'policy_grad_norm',
    'parse_success_rate',
    'ambig_preserve_rate',
    'reward_task_std',
    'adv_abs_mean',
]


def load(tag: str) -> list[dict]:
    fp = DIAG_DIR / f'{tag}_metrics.jsonl'
    rows = []
    with fp.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def step_bins(n_total: int, n_bins: int = 10) -> list[tuple[int, int]]:
    """Return list of (lo, hi) inclusive step ranges covering [0, n_total)."""
    if n_total <= 0:
        return []
    edges = [int(round(i * n_total / n_bins)) for i in range(n_bins + 1)]
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1] - 1
        if hi < lo:
            hi = lo
        bins.append((lo, hi))
    return bins


def bin_mean(rows: list[dict], lo: int, hi: int, key: str) -> float | None:
    vals = [r[key] for r in rows if lo <= r['global_step'] <= hi and key in r]
    if not vals:
        return None
    return float(st.mean(vals))


def main() -> None:
    runs = {tag: load(tag) for tag in RUNS}
    n_max = max(rs[-1]['global_step'] + 1 for rs in runs.values())
    print(f'Max global_step+1 across runs: {n_max}')
    for tag, rs in runs.items():
        last = rs[-1]
        print(f'  {tag:>9s}: rows={len(rs):>3d}  last_step={last["global_step"]:>3d}  epoch={last["epoch"]}')
    print()

    bins = step_bins(n_max, n_bins=10)

    for key in SERIES:
        print(f'=== {key} ===')
        # Header
        cols = '  '.join(f'{tag:>9s}' for tag in RUNS)
        print(f'  bin steps         {cols}')
        for (lo, hi) in bins:
            row = []
            for tag in RUNS:
                v = bin_mean(runs[tag], lo, hi, key)
                row.append(f'{v:>+9.4f}' if v is not None else f'{"-":>9s}')
            print(f'  [{lo:>3d}-{hi:>3d}]      ' + '  '.join(row))
        print()

    # End-of-run comparison
    print('=== Final-step values (last logged row of each run) ===')
    cols = '  '.join(f'{tag:>9s}' for tag in RUNS)
    print(f'  metric              {cols}')
    for key in SERIES:
        row = []
        for tag in RUNS:
            v = runs[tag][-1].get(key)
            row.append(f'{v:>+9.4f}' if v is not None else f'{"-":>9s}')
        print(f'  {key:<20s}' + '  '.join(row))
    print()

    # Reward-task collapse signal: last 20% mean
    print('=== Last-20% means (training tail, robust to single-step spikes) ===')
    cols = '  '.join(f'{tag:>9s}' for tag in RUNS)
    print(f'  metric              {cols}')
    for key in SERIES:
        row = []
        for tag in RUNS:
            rs = runs[tag]
            last_step = rs[-1]['global_step']
            tail_lo = int(last_step * 0.8)
            v = bin_mean(rs, tail_lo, last_step, key)
            row.append(f'{v:>+9.4f}' if v is not None else f'{"-":>9s}')
        print(f'  {key:<20s}' + '  '.join(row))


if __name__ == '__main__':
    main()
