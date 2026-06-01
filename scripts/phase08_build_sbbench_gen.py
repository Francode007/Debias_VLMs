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

GEN_IN = "/tmp/sb_bench_generations.jsonl"
# Authoritative parquet used by generate_sb_bench_answers.py: matches the
# `question_id` row-index assigned during generation (i + j).
PARQUET = "/tmp/sb_bench_data.parquet"
GEN_OUT = "/tmp/sbbench_base_vlbias_gen.jsonl"
INDEX_OUT = "/tmp/sb_bench_as_vlbias.parquet"

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

    df = pd.read_parquet(PARQUET)
    print(f"loaded {len(df)} parquet rows from {PARQUET}")

    # question_id is the row-index assigned during generation (i + j).
    enriched = []
    out_of_range = 0
    for r in gen_rows:
        qid = int(r["question_id"])
        if qid < 0 or qid >= len(df):
            out_of_range += 1
            continue
        row = df.iloc[qid]
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

    print(f"enriched {len(enriched)} rows (skipped {out_of_range} out-of-range qids)")

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

    # Minimal id↔image_path index parquet for the scorer's join. The scorer
    # uses df["id"] ↔ JSONL question_id, so we expose the row-index as `id`
    # (string), and copy the unpacked image relative path.
    idx_rows = [{"id": str(r["question_id"]), "image_path": r["image_path"]} for r in enriched]
    idx_df = pd.DataFrame(idx_rows)
    idx_df.to_parquet(INDEX_OUT, index=False)
    print(f"wrote {INDEX_OUT} ({len(idx_df)} rows)")


if __name__ == "__main__":
    main()
