"""
Phase 0.8 — Build an enriched SB-Bench generation JSONL + index parquet
compatible with `src/modules/evaluation/score_vlbias_offline.py`.

Inputs (local):
  - /tmp/sb_bench_generations.jsonl   (pulled from volume:
        /sb_bench_vanilla_baseline/sb_bench_generations.jsonl)
  - sb_bench_data/data/test-*.parquet (BBQ shards: id, file_name, category,
        question_polarity, label, additional_metadata, …)

Outputs (local, then `modal volume put`):
  - sbbench_base_vlbias_gen.jsonl  — same field-set as base_vlbias_gen.jsonl
        with `condition`, `bbq_axis`, `qformat`, `subgroup`, `image_path`
        added so the offline scorer accepts it without changes.
  - sb_bench_as_vlbias.parquet     — minimal `id`,`image_path` index for the
        scorer's `qid_to_image` join.

Run:
  debias_env/bin/python scripts/phase08_build_sbbench_gen.py
  modal volume put debias-vlm-persistent-storage \
      /tmp/sbbench_base_vlbias_gen.jsonl \
      /phase07_vlbiasbench/sbbench_base_vlbias_gen.jsonl
  modal volume put debias-vlm-persistent-storage \
      /tmp/sb_bench_as_vlbias.parquet \
      /sb_bench_data/sb_bench_as_vlbias.parquet
"""
from __future__ import annotations

import ast
import glob
import json
from pathlib import Path

import pandas as pd

# Phase 0.8 §4½.15 re-run: read the 9-axis vanilla generations + parquet built
# from the full SB-Bench (test2-*) shards. Outputs use the `sbbench9` / `9axis`
# tag so they never collide with the legacy 2-axis (`sbbench_base`) artefacts.
GEN_IN = "/tmp/sb_bench_generations_9axis.jsonl"
# Authoritative parquet used by generate_sb_bench_answers.py: matches the
# `question_id` row-index assigned during generation (i + j).
PARQUET = "/tmp/sb_bench_data_9axis.parquet"
GEN_OUT = "/tmp/sbbench9_base_vlbias_gen.jsonl"
INDEX_OUT = "/tmp/sb_bench9_as_vlbias.parquet"

# Mirrors generate_sb_bench_answers.SB_BENCH_CATEGORIES; the parquet stores
# category as int → resolve to readable axis name for cell stratification.
SB_BENCH_CATEGORIES = [
    "Age", "Disability", "Gender", "Nationality",
    "Physical Appearance", "Race/Ethnicity", "Religion", "SES", "Sexual Orientation",
]


def resolve_category(cat_val) -> str:
    if isinstance(cat_val, (int,)) and 0 <= int(cat_val) < len(SB_BENCH_CATEGORIES):
        return SB_BENCH_CATEGORIES[int(cat_val)]
    try:
        i = int(cat_val)
        if 0 <= i < len(SB_BENCH_CATEGORIES):
            return SB_BENCH_CATEGORIES[i]
    except (TypeError, ValueError):
        pass
    return str(cat_val)


def derive_condition(polarity: int, label: int) -> str:
    if int(label) == 2:
        return "ambig"
    return "neg" if int(polarity) == 0 else "non_neg"


def parse_subgroup(raw) -> str:
    if raw is None:
        return ""
    try:
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
            except Exception:
                obj = ast.literal_eval(raw)
        else:
            obj = raw
        groups = obj.get("stereotyped_groups") if isinstance(obj, dict) else None
        if groups and isinstance(groups, (list, tuple)):
            return str(groups[0])
    except Exception:
        return ""
    return ""


def main() -> None:
    gen_rows = [json.loads(l) for l in open(GEN_IN)]
    print(f"loaded {len(gen_rows)} generation rows")

    # The 9-axis SB-Bench parquet stores image bytes inline; a vanilla
    # pd.read_parquet hits ArrowNotImplementedError on chunked struct columns
    # when a single row group's BINARY child exceeds the int32 offset limit.
    # Read row-group-by-row-group and concat — same trick used in
    # generate_sb_bench_answers.py.
    import pyarrow.parquet as _pq, pyarrow as _pa
    _pf = _pq.ParquetFile(PARQUET)
    _parts = [_pf.read_row_group(i).combine_chunks() for i in range(_pf.num_row_groups)]
    df = _pa.concat_tables(_parts).to_pandas()
    del _parts
    print(f"loaded {len(df)} parquet rows from {PARQUET} ({_pf.num_row_groups} row groups)")

    # Phase 0.8 §4½.15 fix: question_id is now the SB-Bench string `id`
    # (e.g. "01_01_0000_2_01") written by generate_sb_bench_answers.py.
    # Join on parquet `id` column, not row position.
    df_by_id = {str(k): i for i, k in enumerate(df["id"].astype(str).tolist())}
    enriched = []
    missing = 0
    for r in gen_rows:
        qid = str(r["question_id"])
        idx = df_by_id.get(qid)
        if idx is None:
            missing += 1
            continue
        row = df.iloc[idx]
        polarity = int(row["question_polarity"])
        label_pq = int(row["label"])
        out = dict(r)
        out["condition"] = derive_condition(polarity, label_pq)
        out["bbq_axis"] = resolve_category(row["category"])
        out["qformat"] = "sbbench"
        out["subgroup"] = parse_subgroup(row.get("additional_metadata"))
        # file_name in this parquet is a struct {bytes, path}; pull `path`.
        fn = row["file_name"]
        if isinstance(fn, dict):
            img_rel = str(fn.get("path", ""))
        else:
            img_rel = str(fn)
        out["image_path"] = img_rel
        if int(out.get("label", label_pq)) != label_pq:
            out["label"] = label_pq
        enriched.append(out)

    print(f"enriched {len(enriched)} rows (skipped {missing} qids missing from parquet)")

    with open(GEN_OUT, "w") as f:
        for r in enriched:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {GEN_OUT}")

    # cell distribution sanity
    from collections import Counter
    cells = Counter(f"{r['bbq_axis']}::{r['condition']}" for r in enriched)
    print(f"cells: {len(cells)}")
    for k, v in sorted(cells.items()):
        print(f"  {k:60s} {v}")

    # Minimal id↔image index parquet for the scorer's join. The scorer uses
    # df["id"] ↔ JSONL question_id and (optionally) df["image_bytes"] when
    # the image isn't unpacked on disk. SB-Bench images live as embedded
    # bytes in the source parquet (`file_name.bytes`), so we carry them
    # through directly — no separate unpacking step needed.
    df_by_id_for_idx = {str(k): i for i, k in enumerate(df["id"].astype(str).tolist())}
    idx_rows = []
    for r in enriched:
        qid = str(r["question_id"])
        idx = df_by_id_for_idx.get(qid)
        b = None
        if idx is not None:
            fn = df.iloc[idx]["file_name"]
            if isinstance(fn, dict) and "bytes" in fn:
                b = fn["bytes"]
        idx_rows.append({"id": qid, "image_path": r["image_path"], "image_bytes": b})
    idx_df = pd.DataFrame(idx_rows)
    idx_df.to_parquet(INDEX_OUT, index=False)
    n_with_bytes = sum(1 for r in idx_rows if r["image_bytes"] is not None and len(r["image_bytes"]) > 0)
    print(f"wrote {INDEX_OUT} ({len(idx_df)} rows, {n_with_bytes} with image_bytes)")


if __name__ == "__main__":
    main()
