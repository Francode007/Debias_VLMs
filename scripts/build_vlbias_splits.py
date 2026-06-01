#!/usr/bin/env python3
"""
Partition VLBias gen JSONL into train / holdout splits by *template*.

A "template" is the tuple (bbq_axis, subgroup, qformat). A fixed fraction of
templates is assigned to the holdout split; every row with that template-tuple
goes to holdout, every other row goes to train. This guarantees the holdout
contains *unseen* templates, not merely unseen instances.

Usage:
    python scripts/build_vlbias_splits.py \
        --input  /path/to/base_vlbias_gen.jsonl \
        --out-train  /path/to/vlbias_train.jsonl \
        --out-holdout /path/to/vlbias_holdout.jsonl \
        --holdout-frac 0.20 \
        --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def template_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("bbq_axis", "")),
        str(row.get("subgroup", "")),
        str(row.get("qformat", "")),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-train", required=True, type=Path)
    ap.add_argument("--out-holdout", required=True, type=Path)
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stratify-by",
        default="bbq_axis",
        help="Stratify holdout-template sampling by this field so every axis "
        "contributes proportionally. Use empty string to disable.",
    )
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.open()]
    print(f"loaded {len(rows)} rows from {args.input}")

    # Group templates by stratification bucket.
    buckets: dict[str, list[tuple]] = defaultdict(list)
    seen_templates: set[tuple] = set()
    for r in rows:
        t = template_key(r)
        if t in seen_templates:
            continue
        seen_templates.add(t)
        bucket = str(r.get(args.stratify_by, "")) if args.stratify_by else "_all"
        buckets[bucket].append(t)
    print(f"unique templates: {len(seen_templates)} across {len(buckets)} bucket(s)")

    rng = random.Random(args.seed)
    holdout_templates: set[tuple] = set()
    for bucket, tmpls in buckets.items():
        tmpls_sorted = sorted(tmpls)
        rng.shuffle(tmpls_sorted)
        n_hold = max(1, round(len(tmpls_sorted) * args.holdout_frac)) if tmpls_sorted else 0
        n_hold = min(n_hold, max(0, len(tmpls_sorted) - 1))  # keep ≥1 in train
        for t in tmpls_sorted[:n_hold]:
            holdout_templates.add(t)
        print(
            f"  bucket={bucket!r:30s}  templates={len(tmpls_sorted):4d}  "
            f"→ holdout={n_hold}  train={len(tmpls_sorted) - n_hold}"
        )

    train_rows, holdout_rows = [], []
    for r in rows:
        (holdout_rows if template_key(r) in holdout_templates else train_rows).append(r)

    def dump(path: Path, items: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")

    dump(args.out_train, train_rows)
    dump(args.out_holdout, holdout_rows)

    def summarise(name: str, items: list[dict]) -> None:
        ax = Counter(r.get("bbq_axis", "") for r in items)
        cond = Counter(r.get("condition", "") for r in items)
        qf = Counter(r.get("qformat", "") for r in items)
        tmpls = len({template_key(r) for r in items})
        print(f"\n{name}: n={len(items)}  templates={tmpls}")
        print(f"  axis     : {dict(ax)}")
        print(f"  condition: {dict(cond)}")
        print(f"  qformat  : {dict(qf)}")

    summarise("TRAIN  ", train_rows)
    summarise("HOLDOUT", holdout_rows)

    overlap = {template_key(r) for r in train_rows} & {
        template_key(r) for r in holdout_rows
    }
    assert not overlap, f"template overlap detected: {overlap}"
    print(f"\n✅ wrote {args.out_train}  ({len(train_rows)} rows)")
    print(f"✅ wrote {args.out_holdout}  ({len(holdout_rows)} rows)")


if __name__ == "__main__":
    main()
