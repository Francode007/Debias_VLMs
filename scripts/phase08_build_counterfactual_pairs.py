#!/usr/bin/env python3
"""Phase 0.8 Strategic Plan §4 — Counterfactual flip-rate eval set builder.

Constructs paired SB-Bench rows differing ONLY in `question_polarity`
(neg ↔ non-neg) but sharing the same image and `context` (BBQ scene
description). A bias-free model should give the SAME letter answer to both
members of a pair (the gold `label` is identical within a pair — same
ans-slot — only the question wording flips). A biased model swaps its
answer toward whichever demographic the new polarity flatters / blames.

Flip-rate = P(pred_orig ≠ pred_swap | same image+context).
Lower flip-rate = less polarity-sensitive = less biased.

This script is read-only. It does NOT call the model. It produces:

  Phase0.8/counterfactual_eval/cf_pairs.parquet
     columns:
       pair_id          : str — stable identifier for the (orig, swap) pair
       qid_orig         : str — SB-Bench id with question_polarity == 0
       qid_swap         : str — SB-Bench id with question_polarity == 1
       image_path       : str — same for both members of pair
       bbq_axis         : str — Age / Gender / Disability / … (resolved name)
       label_orig       : int — gold answer index for `qid_orig` (0/1/2)
       label_swap       : int — gold answer index for `qid_swap` (== orig)
       context          : str — BBQ scene description (shared)
       question_orig    : str — neg-polarity question wording
       question_swap    : str — non_neg-polarity question wording
       ans0, ans1, ans2 : str — answer slate (identical across pair)

Pairing key: (image_path, context).
On the SB-Bench 9-axis test split this yields ~6166 pairs.

Run:
  python scripts/phase08_build_counterfactual_pairs.py \\
      --parquet-glob 'sb_bench_data/data/test-*.parquet' \\
      --out Phase0.8/counterfactual_eval/cf_pairs.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SB_BENCH_CATEGORIES = [
    "Age", "Disability", "Gender", "Nationality",
    "Physical Appearance", "Race/Ethnicity", "Religion", "SES", "Sexual Orientation",
]


def _read_parquet_safe(path: str) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    parts = [pf.read_row_group(i).combine_chunks() for i in range(pf.num_row_groups)]
    return pa.concat_tables(parts).to_pandas()


def _img_path(fn) -> str:
    if isinstance(fn, dict):
        return str(fn.get("path", ""))
    return str(fn) if fn is not None else ""


def _resolve_axis(cat_val) -> str:
    try:
        i = int(cat_val)
        if 0 <= i < len(SB_BENCH_CATEGORIES):
            return SB_BENCH_CATEGORIES[i]
    except (TypeError, ValueError):
        pass
    return str(cat_val)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default=None,
                   help="Path to a single consolidated SB-Bench parquet.")
    p.add_argument("--parquet-glob", default=None,
                   help="Glob over multiple SB-Bench parquet shards "
                        "(e.g. 'sb_bench_data/data/test-*.parquet').")
    p.add_argument("--out", required=True,
                   help="Output parquet of (qid_orig, qid_swap, …) pair rows.")
    p.add_argument("--report-json", default=None,
                   help="Optional summary stats JSON.")
    args = p.parse_args()

    if not args.parquet and not args.parquet_glob:
        raise SystemExit("Must provide either --parquet or --parquet-glob")

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    if args.parquet_glob:
        files = sorted(glob.glob(args.parquet_glob))
        if not files:
            raise SystemExit(f"No files matched --parquet-glob {args.parquet_glob!r}")
        dfs = []
        for f in files:
            dfs.append(_read_parquet_safe(f))
        df = pd.concat(dfs, ignore_index=True)
        print(f"loaded {len(df)} rows from {len(files)} parquet shards")
    else:
        df = _read_parquet_safe(args.parquet)
        print(f"loaded {len(df)} rows from {args.parquet}")

    df["image_path"] = df["file_name"].apply(_img_path)

    rows_by_key: Dict[tuple, Dict[int, dict]] = defaultdict(dict)
    skipped = 0
    for _, row in df.iterrows():
        try:
            pol = int(row["question_polarity"])
        except Exception:
            skipped += 1
            continue
        if pol not in (0, 1):
            skipped += 1
            continue
        key = (str(row["image_path"]), str(row["context"]))
        rows_by_key[key][pol] = {
            "qid": str(row["id"]),
            "label": int(row["label"]),
            "image_path": str(row["image_path"]),
            "axis": _resolve_axis(row.get("category")),
            "context": str(row.get("context", "")),
            "question": str(row.get("question", "")),
            "ans0": str(row.get("ans0", "")),
            "ans1": str(row.get("ans1", "")),
            "ans2": str(row.get("ans2", "")),
        }

    print(f"  unique (image,context) keys: {len(rows_by_key)}  (skipped {skipped} invalid rows)")

    pairs: List[dict] = []
    incomplete = 0
    label_mismatch = 0
    answer_mismatch = 0
    for k, members in rows_by_key.items():
        if 0 not in members or 1 not in members:
            incomplete += 1
            continue
        a, b = members[0], members[1]
        if a["label"] != b["label"]:
            label_mismatch += 1
            # Keep these — still valid CF pairs even if gold label differs,
            # but flag for downstream filtering.
        if (a["ans0"], a["ans1"], a["ans2"]) != (b["ans0"], b["ans1"], b["ans2"]):
            answer_mismatch += 1
            continue  # different answer slates would invalidate flip-rate
        pairs.append({
            "pair_id": f"{a['image_path']}::{hash(a['context']) & 0xFFFFFFFF:08x}",
            "qid_orig": a["qid"],
            "qid_swap": b["qid"],
            "image_path": a["image_path"],
            "bbq_axis": a["axis"],
            "label_orig": a["label"],
            "label_swap": b["label"],
            "context": a["context"],
            "question_orig": a["question"],
            "question_swap": b["question"],
            "ans0": a["ans0"],
            "ans1": a["ans1"],
            "ans2": a["ans2"],
        })

    print(f"  complete pairs: {len(pairs)}")
    print(f"  incomplete (missing one polarity): {incomplete}")
    print(f"  label_mismatch within pair (kept): {label_mismatch}")
    print(f"  answer-slate mismatch (dropped): {answer_mismatch}")

    out_df = pd.DataFrame(pairs)
    out_df.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(out_df)} pairs)")

    if not out_df.empty:
        print("\nPair counts by axis:")
        print(out_df.groupby("bbq_axis").size().to_string())

    if args.report_json:
        report = {
            "total_pairs": len(pairs),
            "incomplete": incomplete,
            "label_mismatch": label_mismatch,
            "answer_mismatch": answer_mismatch,
            "skipped_rows": skipped,
            "by_axis": (out_df.groupby("bbq_axis").size().to_dict()
                        if not out_df.empty else {}),
        }
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.report_json}")


if __name__ == "__main__":
    main()
