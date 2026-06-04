#!/usr/bin/env python3
"""Phase 0.8 Strategic Plan §4 — Counterfactual flip-rate eval set builder.

Constructs paired SB-Bench rows differing ONLY in `question_polarity`
(neg ↔ non-neg) but sharing the same image, demographic context, and
candidate referents. A bias-free model should flip its predicted referent
between the two members of each pair (because the gold answer flips, by
construction in BBQ); a biased model collapses toward one demographic
regardless of polarity.

This script is read-only. It does NOT call the model. It produces:

  Phase0.8/counterfactual_eval/cf_pairs.parquet
     columns:
       pair_id          : str — stable identifier for the (orig, swap) pair
       qid_orig         : str — SB-Bench id with question_polarity == 0 (neg)
       qid_swap         : str — SB-Bench id with question_polarity == 1 (non-neg)
       image_path       : str — same for both members of pair
       bbq_axis         : str — Age / Gender / Disability / …
       label_orig       : int — gold answer index for `qid_orig` (0/1/2)
       label_swap       : int — gold answer index for `qid_swap`
       condition_orig   : str — neg / non_neg (always neg)
       condition_swap   : str — neg / non_neg (always non_neg)
       context_condition: str — ambig / disambig (must match across pair)
       referents        : list[str] — the two named entities A,B

The pairing key is everything in the SB-Bench BBQ id EXCEPT the polarity
suffix. SB-Bench ids look like "01_03_0042_0_01" where the second-to-last
field is `question_polarity` (0 or 1) and the last is `context_condition`
(00=ambig, 01=disambig — verify against your local parquet).

Validate by hand on 50 random pairs before trusting the output.

Usage:
  python scripts/phase08_build_counterfactual_pairs.py \\
      --parquet /mnt/data/sb_bench_data/sb_bench_data.parquet \\
      --out Phase0.8/counterfactual_eval/cf_pairs.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _read_parquet_safe(path: str) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    parts = [pf.read_row_group(i).combine_chunks() for i in range(pf.num_row_groups)]
    return pa.concat_tables(parts).to_pandas()


def _parse_id(qid: str) -> Tuple[str, int, str] | None:
    """Return (pair_key, polarity, condition_suffix) or None.

    SB-Bench id grammar (BBQ-derived):
        <axis>_<template>_<instance>_<polarity>_<condition>
    where <polarity> ∈ {0,1}. We strip the polarity dimension to form
    `pair_key = axis_template_instance__<condition>` so that exactly two
    rows (polarity=0 and polarity=1) share a pair_key.
    """
    parts = qid.split("_")
    if len(parts) < 5:
        return None
    pol_str = parts[-2]
    cond = parts[-1]
    if pol_str not in {"0", "1"}:
        return None
    head = "_".join(parts[:-2])
    return f"{head}__cond{cond}", int(pol_str), cond


def _img_path(row: pd.Series) -> str:
    fn = row.get("file_name")
    if isinstance(fn, dict):
        return str(fn.get("path", ""))
    return str(fn) if fn is not None else ""


def _referents(row: pd.Series) -> List[str]:
    """Best-effort parse of additional_metadata for the two answer-named entities."""
    md = row.get("additional_metadata")
    if md is None:
        return []
    try:
        if isinstance(md, str):
            md_obj = json.loads(md)
        elif isinstance(md, dict):
            md_obj = md
        else:
            return []
    except Exception:
        return []
    # BBQ schemas typically encode answer_info: {ans0: [text, group], ans1: [...], ans2: [...]}
    refs = []
    for k in ("ans0", "ans1"):
        v = md_obj.get("answer_info", {}).get(k) or md_obj.get(k)
        if isinstance(v, (list, tuple)) and v:
            refs.append(str(v[0]))
    return refs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True,
                   help="Path to SB-Bench parquet (id, file_name, category, "
                        "question_polarity, label, additional_metadata, …)")
    p.add_argument("--out", required=True,
                   help="Output parquet of (qid_orig, qid_swap, …) pair rows.")
    p.add_argument("--report-json", default=None,
                   help="Optional summary stats JSON.")
    args = p.parse_args()

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    df = _read_parquet_safe(args.parquet)
    print(f"loaded {len(df)} rows from {args.parquet}")

    rows_by_key: Dict[str, Dict[int, dict]] = defaultdict(dict)
    skipped = 0
    for _, row in df.iterrows():
        qid = str(row["id"])
        parsed = _parse_id(qid)
        if parsed is None:
            skipped += 1
            continue
        pair_key, polarity, cond_suf = parsed
        rows_by_key[pair_key][polarity] = {
            "qid": qid,
            "polarity": polarity,
            "label": int(row["label"]),
            "image_path": _img_path(row),
            "axis": str(row.get("category", "Unknown")),
            "context_condition": "ambig" if cond_suf in {"00", "0"} else "disambig",
            "referents": _referents(row),
        }
    print(f"  pair_keys: {len(rows_by_key)}  (skipped {skipped} unparseable ids)")

    pairs: List[dict] = []
    incomplete = 0
    image_mismatch = 0
    for pk, members in rows_by_key.items():
        if 0 not in members or 1 not in members:
            incomplete += 1
            continue
        a, b = members[0], members[1]
        if a["image_path"] != b["image_path"]:
            image_mismatch += 1
            continue
        pairs.append({
            "pair_id": pk,
            "qid_orig": a["qid"],
            "qid_swap": b["qid"],
            "image_path": a["image_path"],
            "bbq_axis": a["axis"],
            "label_orig": a["label"],
            "label_swap": b["label"],
            "condition_orig": "neg",
            "condition_swap": "non_neg",
            "context_condition": a["context_condition"],
            "referents": a["referents"] or b["referents"],
        })
    print(f"  complete pairs: {len(pairs)}")
    print(f"  incomplete (missing one polarity): {incomplete}")
    print(f"  image mismatch (likely id-grammar wrong): {image_mismatch}")

    out_df = pd.DataFrame(pairs)
    out_df.to_parquet(args.out, index=False)
    print(f"✅ wrote {args.out} ({len(out_df)} pairs)")

    # Stats per axis × context_condition
    if not out_df.empty:
        cell = out_df.groupby(["bbq_axis", "context_condition"]).size().reset_index(name="n")
        print("\nPair counts by axis × context_condition:")
        print(cell.to_string(index=False))

    if args.report_json:
        report = {
            "total_pairs": len(pairs),
            "incomplete": incomplete,
            "image_mismatch": image_mismatch,
            "skipped_ids": skipped,
            "by_axis": (out_df.groupby("bbq_axis").size().to_dict()
                        if not out_df.empty else {}),
        }
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ wrote {args.report_json}")


if __name__ == "__main__":
    main()
