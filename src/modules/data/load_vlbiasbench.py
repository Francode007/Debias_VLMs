"""
load_vlbiasbench.py
────────────────────────────────────────────────────────────────────────────
Download VLBiasBench close-ended split and pack it into a single parquet file
whose schema mirrors `sb_bench_data.parquet`, so the downstream inference and
evaluation code (`generate_vlbiasbench_answers.py`, `eval_vlbiasbench.py`) can
re-use the same constrained-decode / per-category-accuracy pipeline used for
SB-Bench in Phase 0.6 / 0.7.

Source mirror:  https://huggingface.co/datasets/Aaron080108/VLBiasBench_Close_ended
File:           close_ended.zip   (~1.71 GB)
Unpacks to:     close-ended/json/<category>/*.json
                close-ended/images/<image_path>

Per-record schema produced (one row per close-ended sample):
    id              : int       — running 0..N-1
    qformat         : str       — "base" | "scene" | "scene_text" | "text"
                                  (the question-format dir under json/)
    bbq_axis        : str       — "Age" | "SES" | "Race_ethnicity" | ...
                                  (the 10 BBQ axes, from the JSON filename stem)
    subgroup        : str       — inner `category` field: fine-grained label
                                  like "old", "highSES", "F-Black" (122+ values)
    condition       : str       — "ambig" | "neg" | "non_neg"
                                  (VLBiasBench convention; `disambig` = neg ∪ non_neg)
    context         : str
    question        : str
    question_neg    : str       — paired negated question (consistency probe)
    ans0,ans1,ans2  : str
    label           : int       — 0/1/2 index of gold answer
    label_neg       : int       — gold answer for the negated question
    image_path      : str       — relative path under images/ (read at inference)
    additional_metadata : str   — JSON-dumped extras from the source record

Env vars (with defaults):
    VLBIAS_DATA_PATH   = /mnt/data/vlbiasbench_data
    VLBIAS_HF_REPO     = Aaron080108/VLBiasBench_Close_ended
    VLBIAS_HF_FILENAME = close_ended.zip
"""

import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


HF_REPO = os.environ.get("VLBIAS_HF_REPO", "Aaron080108/VLBiasBench_Close_ended")
HF_FILENAME = os.environ.get("VLBIAS_HF_FILENAME", "close_ended.zip")
DATA_ROOT = os.environ.get("VLBIAS_DATA_PATH", "/mnt/data/vlbiasbench_data")
OUTPUT_PARQUET = os.path.join(DATA_ROOT, "vlbiasbench_close_ended.parquet")


def _download_zip() -> str:
    """Download the close-ended zip from HF Hub. Returns the local cached path."""
    print(f"📥 Downloading {HF_FILENAME} from {HF_REPO} (HuggingFace Hub)...", flush=True)
    local_zip = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        repo_type="dataset",
        cache_dir=os.path.join(DATA_ROOT, "hf_cache"),
    )
    print(f"✅ Cached at {local_zip}", flush=True)
    return local_zip


def _find_close_ended_root(base: str) -> Path:
    """Locate the close-ended subtree under `base`, tolerating either the
    hyphenated ('close-ended') or underscored ('close_ended') spelling, and
    accepting it either as a direct child or as the current dir itself."""
    candidates = {"close-ended", "close_ended"}
    for root, dirs, _ in os.walk(base):
        if os.path.basename(root) in candidates:
            return Path(root)
        for d in dirs:
            if d in candidates:
                return Path(root) / d
    # Diagnostic listing to help debug zips with unexpected layouts.
    print(f"❌ Could not locate close-ended/close_ended under {base}. Tree:")
    for root, dirs, files in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth > 3:
            continue
        print(f"  {'  ' * depth}{os.path.basename(root)}/")
        for fn in files[:3]:
            print(f"  {'  ' * (depth + 1)}{fn}")
    raise RuntimeError(f"Could not find close-ended/close_ended under {base}")


def _unzip(zip_path: str, dest: str) -> Path:
    """Unzip `zip_path` under `dest`. Returns the close-ended root dir."""
    print(f"📦 Unzipping → {dest}", flush=True)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        print(f"   {total} members to extract...", flush=True)
        for i, m in enumerate(members):
            zf.extract(m, dest)
            if (i + 1) % 5000 == 0 or (i + 1) == total:
                print(f"   Extracted {i + 1}/{total} ({100*(i+1)//total}%)", flush=True)
    return _find_close_ended_root(dest)


def _collect_records(close_ended_dir: Path):
    """Walk close-ended/json/<qformat>/<BBQAxis>.json and yield (qformat, bbq_axis, record).

    qformat  ∈ {base, scene, scene_text, text}  (question-format dir)
    bbq_axis ∈ {Age, Disability_status, Gender_identity, Nationality,
                Physical_appearance, Race_ethnicity, Race_x_SES, Race_x_gender,
                Religion, SES}  (the BBQ axis, taken from the JSON filename stem)
    """
    json_root = close_ended_dir / "json"
    if not json_root.exists():
        raise RuntimeError(f"Missing {json_root}")
    qformats = sorted([d.name for d in json_root.iterdir() if d.is_dir()])
    print(f"📂 Discovered {len(qformats)} qformats: {qformats}", flush=True)
    # Count total JSON files for the progress bar.
    json_files = []
    for qf in qformats:
        for jf in sorted((json_root / qf).glob("*.json")):
            json_files.append((qf, jf.stem, jf))  # (qformat, bbq_axis, path)
    print(f"   Total JSON files to process: {len(json_files)}", flush=True)
    for i, (qf, axis, jf) in enumerate(json_files):
        if (i + 1) % 10 == 0 or (i + 1) == len(json_files):
            print(f"   Loading JSONs: {i + 1}/{len(json_files)}", flush=True)
        with open(jf, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for rec in data:
            yield qf, axis, rec


def build_parquet():
    os.makedirs(DATA_ROOT, exist_ok=True)
    unpack_dir = os.path.join(DATA_ROOT, "unpacked")

    # Skip download/unzip if already unpacked
    candidate = None
    if os.path.isdir(unpack_dir):
        try:
            candidate = _find_close_ended_root(unpack_dir)
        except RuntimeError:
            candidate = None
    if candidate is None or not candidate.exists():
        zip_path = _download_zip()
        candidate = _unzip(zip_path, unpack_dir)
    else:
        print(f"✅ Found existing unpacked tree at {candidate}", flush=True)

    image_root = candidate / "images"
    if not image_root.exists():
        raise RuntimeError(f"Missing image root: {image_root}")

    records = list(_collect_records(candidate))
    print(f"📋 Building metadata parquet for {len(records)} records (images stay on disk)...", flush=True)

    # Verify a sample of images exist on disk (don't embed bytes in parquet).
    missing_images = 0
    rows = []
    for idx, (qformat, bbq_axis, rec) in enumerate(records):
        img_path = rec.get("image_path", "")
        # Quick existence check (no byte read — just stat).
        full = image_root / img_path
        if not full.exists():
            alt = image_root / os.path.basename(img_path)
            if not alt.exists():
                missing_images += 1
                if missing_images <= 5:
                    print(f"⚠️  image not found: {full}", flush=True)
                continue

        if "label" in rec and rec["label"] is not None:
            label = int(rec["label"])
        elif rec.get("condition") == "ambig":
            label = 2
        else:
            label = -1

        label_neg_raw = rec.get("label_neg")
        label_neg = int(label_neg_raw) if label_neg_raw is not None else -1

        extras = {k: v for k, v in rec.items()
                  if k not in {"context", "question", "question_neg",
                               "ans0", "ans1", "ans2",
                               "label", "label_neg", "image_path",
                               "category", "condition", "idx"}}

        rows.append({
            "id": idx,
            "qformat": qformat,
            "bbq_axis": bbq_axis,
            "subgroup": rec.get("category", ""),
            "condition": rec.get("condition", "unknown"),
            "context": rec.get("context", ""),
            "question": rec.get("question", ""),
            "question_neg": rec.get("question_neg", ""),
            "ans0": rec.get("ans0", ""),
            "ans1": rec.get("ans1", ""),
            "ans2": rec.get("ans2", ""),
            "label": label,
            "label_neg": label_neg,
            "image_path": img_path,
            "additional_metadata": json.dumps(extras, ensure_ascii=False),
        })

        if (idx + 1) % 10000 == 0:
            print(f"   Processed {idx + 1}/{len(records)}...", flush=True)

    if missing_images > 5:
        print(f"⚠️  Skipped {missing_images} records with missing images.")

    if not rows:
        raise RuntimeError("No records collected — check the unpacked layout.")

    print(f"📊 Total records: {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    print(f"💾 Writing parquet → {OUTPUT_PARQUET}", flush=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✅ Done. ({os.path.getsize(OUTPUT_PARQUET) / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    try:
        build_parquet()
    except Exception as e:
        print(f"❌ Error building VLBiasBench parquet: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
