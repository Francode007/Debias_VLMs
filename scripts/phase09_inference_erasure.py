"""
phase09_inference_erasure.py
─────────────────────────────────────────────────────────────────────────────
Phase 0.9.5 E1 — Inference-time bias-subspace erasure baseline.

Strategic-plan spec: Phase0.9_Strategic_Plan.md §3.1
Critical-review motivation: Phase0.9_Critical_Review.md §6 (the missing baseline)

What it does
────────────
1. Loads vanilla Qwen2.5-VL-3B-Instruct (no LoRA, no PPO).
2. Loads one or more unit-normalised `bias_aligned` probe directions
   `v_bias_L ∈ R^D` from `base_L{N}_probe_weights_biasA.npz` files.
3. Registers a forward-hook at each `model.layers[L]` that projects
   `v_bias_L` out of the layer's output at every position:

       h_post = h_pre - (h_pre @ v) * v               # v unit-normed (D,)

   (matches the spec formula verbatim; standard INLP / linear concept
   erasure.)
4. Runs constrained-decoding generation over SB-Bench or VLBiasBench
   using the EXACT same prompt format and letter-token argmax used by
   `generate_sb_bench_answers.py` / `generate_vlbiasbench_answers.py`.
5. Dumps a JSONL with the same per-record schema as the canonical gen
   files so downstream evaluators (`eval_sb_bench.py`,
   `eval_vlbiasbench.py`) work unchanged.

E1a (single-layer): `--layer-indices 13 --probe-paths .../base_L13_probe_weights_biasA.npz`
E1b (multi-layer): `--layer-indices 13 17 21 25 \
                    --probe-paths .../L13.npz .../L17.npz .../L21.npz .../L25.npz`

Usage examples (local; on Modal use `run_inference_erasure` in run_modal.py)
────────────────────────────────────────────────────────────────────────────
SB-Bench, L13-only erasure:

    python -m scripts.phase09_inference_erasure \
        --dataset sb_bench \
        --data_path /mnt/data/sb_bench_data/sb_bench_data.parquet \
        --split test \
        --split_indices_path /mnt/data/split_indices.json \
        --layer-indices 13 \
        --probe-paths Phase0.8/a3_results/base_L13_probe_weights_biasA.npz \
        --output_jsonl /mnt/data/phase09_erasure/E1a_L13_sbbench_gen.jsonl

VLBiasBench, multi-layer {L13,L17,L21,L25}:

    python -m scripts.phase09_inference_erasure \
        --dataset vlbias \
        --data_path /mnt/data/vlbiasbench_data/vlbiasbench_close_ended.parquet \
        --image_root /mnt/data/vlbiasbench_data/unpacked/close_ended/images \
        --num_samples 2000 \
        --layer-indices 13 17 21 25 \
        --probe-paths Phase0.8/a3_results/base_L13_probe_weights_biasA.npz \
                      Phase0.9/ensemble_bundle_raw/base_L17_probe_weights_biasA.npz \
                      Phase0.9/ensemble_bundle_raw/base_L21_probe_weights_biasA.npz \
                      Phase0.9/ensemble_bundle_raw/base_L25_probe_weights_biasA.npz \
        --output_jsonl /mnt/data/phase09_erasure/E1b_L13-25_vlbias_gen.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Re-use the canonical helpers from the inference scripts so the prompt
# format + letter token resolution stay byte-identical to the baselines.
from modules.inference.generate_sb_bench_answers import (  # type: ignore
    SB_BENCH_CATEGORIES,
    build_prompt as sb_build_prompt,
    resolve_category,
    resolve_letter_ids,
)
from modules.inference.generate_vlbiasbench_answers import (  # type: ignore
    build_prompt as vlbias_build_prompt,
    _stratified_subsample,
)

CHOICE_MAP = {0: "A", 1: "B", 2: "C"}


# ─── Args ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset", required=True, choices=["sb_bench", "vlbias"])
    p.add_argument("--data_path", required=True)
    p.add_argument(
        "--image_root",
        default="/mnt/data/vlbiasbench_data/unpacked/close_ended/images",
        help="Only used for --dataset vlbias.",
    )
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--batch_size", type=int, default=4)

    # ── SB-Bench-only flags (ignored for vlbias) ────────────────────────
    p.add_argument(
        "--split", default="test",
        choices=["train", "test", "all"],
        help="SB-Bench split to evaluate on.",
    )
    p.add_argument("--split_indices_path", default=None)
    p.add_argument("--max_pixels", type=int, default=512 * 512)
    p.add_argument("--min_pixels", type=int, default=0)

    # ── VLBias-only flags ──────────────────────────────────────────────
    p.add_argument(
        "--num_samples", type=int, default=2000,
        help="Stratified subsample for VLBias; 0 = full set.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--condition", default="all",
        choices=["all", "ambig", "disambig"],
    )
    p.add_argument("--qformat", default="base,scene,scene_text")

    # ── Erasure hook config ───────────────────────────────────────────
    p.add_argument(
        "--layer-indices", type=int, nargs="+", required=True,
        help="LM-decoder layer indices to register erasure hooks at "
             "(e.g. 13 for E1a; 13 17 21 25 for E1b).",
    )
    p.add_argument(
        "--probe-paths", type=Path, nargs="+", required=True,
        help="Per-layer probe npz files (one per --layer-indices entry, in "
             "the same order). Each file must contain a `coef` array of "
             "shape (D,) which will be unit-normalised before use.",
    )
    p.add_argument(
        "--tag", default="",
        help="Optional tag for the JSONL records (carried in the per-record "
             "metadata as `erasure_tag`).",
    )
    return p.parse_args()


# ─── Probe loading ───────────────────────────────────────────────────────────
def load_probe_directions(
    probe_paths: List[Path], layer_indices: List[int],
) -> List[Tuple[int, np.ndarray]]:
    """Load each probe .npz, unit-normalise `coef`, return list of (L, w_unit)."""
    if len(probe_paths) != len(layer_indices):
        raise ValueError(
            f"--layer-indices ({len(layer_indices)}) and --probe-paths "
            f"({len(probe_paths)}) must have the same length."
        )
    out: List[Tuple[int, np.ndarray]] = []
    for L, path in zip(layer_indices, probe_paths):
        if not path.exists():
            raise FileNotFoundError(f"probe missing: {path}")
        d = np.load(path, allow_pickle=True)
        if "coef" not in d.files:
            raise KeyError(
                f"{path} has no `coef` key (got {list(d.files)}). "
                f"Expected probe_layers.py-style npz."
            )
        coef = d["coef"].astype(np.float32)
        if coef.ndim != 1:
            raise ValueError(f"{path}: coef must be 1-D, got shape {coef.shape}")
        nrm = float(np.linalg.norm(coef))
        if nrm <= 0:
            raise ValueError(f"{path}: coef has zero norm")
        # Cross-check: file should match the layer index claimed in the args.
        layer_idx_in_file = int(d["layer_idx"]) if "layer_idx" in d.files else L
        if layer_idx_in_file != L:
            print(
                f"⚠  {path.name}: layer_idx={layer_idx_in_file} but --layer-indices "
                f"gives {L}; using --layer-indices value (hook will fire at L={L})."
            )
        out.append((L, coef / nrm))
        print(
            f"  loaded probe L{L:>2d}  shape={coef.shape}  ||coef||={nrm:.3f}  "
            f"task={d.get('task', 'unknown')!s}  src={path.name}"
        )
    return out


# ─── Hook factory + layer-path resolution ────────────────────────────────────
def make_erasure_hook(v_unit: torch.Tensor) -> Callable:
    """Return a forward hook that projects v_unit out of every position of
    the layer's output residual stream.

    Spec (Phase0.9_Strategic_Plan.md §3.1):
        h_post = h_pre - (h_pre @ v_bias) * v_bias              # v_bias unit-normed
    """
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            v = v_unit.to(h.dtype).to(h.device)                # (D,)
            proj = (h @ v).unsqueeze(-1) * v                   # (B, T, D)
            return (h - proj,) + output[1:]
        h = output
        v = v_unit.to(h.dtype).to(h.device)
        proj = (h @ v).unsqueeze(-1) * v
        return h - proj
    return hook


def resolve_layers(model) -> Tuple[str, List[torch.nn.Module]]:
    """Find the LM-decoder layer list on the Qwen2.5-VL model.

    Mirrors the resolver used in scripts/phase0_activation_steering.py.
    Returns (path_string, layer_list).
    """
    candidates = [
        "model.language_model.layers",
        "language_model.model.layers",
        "model.layers",
        "language_model.layers",
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "__getitem__"):
            return path, obj
    raise RuntimeError(
        "Could not locate LM-decoder layer list on the model. Tried: "
        + ", ".join(candidates)
    )


# ─── Dataset loaders (mirror canonical gen scripts) ─────────────────────────
def load_sb_bench(args: argparse.Namespace):
    """Load SB-Bench parquet → datasets.Dataset, applying the split filter.

    Mirrors generate_sb_bench_answers.py exactly (including the multi-row-group
    workaround so the chunked struct file_name column doesn't trip pyarrow).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset

    pf = pq.ParquetFile(args.data_path)
    print(f"  parquet: rows={pf.metadata.num_rows} row_groups={pf.num_row_groups}")
    parts = [pf.read_row_group(i).combine_chunks() for i in range(pf.num_row_groups)]
    table = pa.concat_tables(parts)
    ds = Dataset(arrow_table=table)
    del parts, table

    if args.split in ("train", "test") and args.split_indices_path:
        with open(args.split_indices_path, "r") as f:
            split_info = json.load(f)
        key = "train_indices" if args.split == "train" else "test_indices"
        ds = ds.select(split_info[key])
    print(f"  → {len(ds)} SB-Bench rows after split filter")
    return ds


def load_vlbias(args: argparse.Namespace):
    """Load VLBiasBench parquet → datasets.Dataset, applying qformat + condition
    + label + stratified-subsample filters. Mirrors
    generate_vlbiasbench_answers.py."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files={"test": args.data_path})["test"]
    print(f"  Total examples: {len(ds)}")

    keep_qf = {x.strip() for x in args.qformat.split(",") if x.strip()}
    if "qformat" in ds.column_names and keep_qf:
        keep = [i for i, qf in enumerate(ds["qformat"]) if qf in keep_qf]
        ds = ds.select(keep)
        print(f"  qformat ∈ {sorted(keep_qf)} → {len(ds)} rows")

    if args.condition == "ambig":
        keep = [i for i, c in enumerate(ds["condition"]) if c == "ambig"]
        ds = ds.select(keep)
        print(f"  condition='ambig' → {len(ds)} rows")
    elif args.condition == "disambig":
        keep = [i for i, c in enumerate(ds["condition"]) if c in ("neg", "non_neg")]
        ds = ds.select(keep)
        print(f"  condition∈{{neg,non_neg}} → {len(ds)} rows")

    label_keep = [i for i, l in enumerate(ds["label"]) if int(l) >= 0]
    if len(label_keep) != len(ds):
        ds = ds.select(label_keep)
        print(f"  Dropped rows with label=-1; remaining: {len(ds)}")

    if args.num_samples and args.num_samples > 0:
        ds = _stratified_subsample(ds, args.num_samples, args.seed)
        print(f"  Stratified subsample → {len(ds)} rows")
    return ds


# ─── Generation loops ───────────────────────────────────────────────────────
def generate_sb_bench(model, processor, ds, args, abc_ids, device, erasure_tag):
    """Mirror generate_sb_bench_answers.py generation loop (single forward
    pass + argmax over A/B/C). The model already has erasure hooks attached."""
    results: List[dict] = []
    for i in tqdm(range(0, len(ds), args.batch_size), desc="SB-Bench (erasure)"):
        batch = ds[i: i + args.batch_size]
        bsz = len(batch["question"])

        batch_messages, batch_images, meta = [], [], []
        for j in range(bsz):
            context = batch["context"][j]
            question = batch["question"][j]
            ans0, ans1, ans2 = batch["ans0"][j], batch["ans1"][j], batch["ans2"][j]
            label = int(batch["label"][j])
            category = resolve_category(batch["category"][j])
            sb_id_raw = batch.get("id", [None] * bsz)[j]
            sb_qid = str(sb_id_raw) if sb_id_raw is not None else f"row{i + j}"
            prompt_text = sb_build_prompt(context, question, ans0, ans1, ans2)

            img_field = batch.get("file_name", [None] * bsz)[j]
            if img_field is None:
                img_field = batch.get("file_name.bytes", [None] * bsz)[j]
            if isinstance(img_field, dict) and "bytes" in img_field:
                img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
            elif isinstance(img_field, bytes):
                img = Image.open(io.BytesIO(img_field)).convert("RGB")
            else:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))

            batch_images.append(img)
            batch_messages.append([
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ]}
            ])
            meta.append({
                "question_id": sb_qid,
                "category": category,
                "label": label,
                "ans0": ans0, "ans1": ans1, "ans2": ans2,
                "context": context, "question": question,
                "erasure_tag": erasure_tag,
            })

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        inputs = processor(
            text=texts, images=batch_images, padding=True, return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            letter_logits = next_token_logits[:, list(abc_ids)]
            picks = letter_logits.argmax(dim=-1).tolist()

        output_texts = [CHOICE_MAP[p] for p in picks]
        for m, text in zip(meta, output_texts):
            results.append({**m, "text": text.strip()})
    return results


def generate_vlbias(model, processor, ds, args, abc_ids, device, erasure_tag):
    """Mirror generate_vlbiasbench_answers.py generation loop."""
    results: List[dict] = []
    for i in tqdm(range(0, len(ds), args.batch_size), desc="VLBias (erasure)"):
        batch = ds[i: i + args.batch_size]
        bsz = len(batch["question"])

        batch_messages, batch_images, meta = [], [], []
        for j in range(bsz):
            context = batch["context"][j]
            question = batch["question"][j]
            ans0, ans1, ans2 = batch["ans0"][j], batch["ans1"][j], batch["ans2"][j]
            label = int(batch["label"][j])
            bbq_axis = str(batch["bbq_axis"][j]) if "bbq_axis" in batch else ""
            qformat = str(batch["qformat"][j]) if "qformat" in batch else ""
            subgroup = str(batch["subgroup"][j]) if "subgroup" in batch else ""
            condition = str(batch["condition"][j])

            prompt_text = vlbias_build_prompt(context, question, ans0, ans1, ans2)

            img_rel = batch["image_path"][j]
            img_full = os.path.join(args.image_root, img_rel)
            if not os.path.exists(img_full):
                img_full = os.path.join(args.image_root, os.path.basename(img_rel))
            if os.path.exists(img_full):
                img = Image.open(img_full).convert("RGB")
            else:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))

            batch_images.append(img)
            batch_messages.append([
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ]}
            ])
            meta.append({
                "question_id": int(batch["id"][j]) if "id" in batch else i + j,
                "bbq_axis": bbq_axis,
                "qformat": qformat,
                "subgroup": subgroup,
                "condition": condition,
                "label": label,
                "ans0": ans0, "ans1": ans1, "ans2": ans2,
                "context": context, "question": question,
                "erasure_tag": erasure_tag,
            })

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        inputs = processor(
            text=texts, images=batch_images, padding=True, return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            letter_logits = next_token_logits[:, list(abc_ids)]
            picks = letter_logits.argmax(dim=-1).tolist()

        output_texts = [CHOICE_MAP[p] for p in picks]
        for m, text in zip(meta, output_texts):
            results.append({**m, "text": text.strip()})
    return results


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load probe directions ────────────────────────────────────────
    print("▶ Loading probe directions")
    pairs = load_probe_directions(args.probe_paths, args.layer_indices)

    erasure_tag = (
        args.tag
        or "L" + "-".join(str(L) for L, _ in pairs) + "_erasure"
    )
    print(f"  erasure_tag = {erasure_tag}")

    # ── 2. Load model + processor ───────────────────────────────────────
    print(f"▶ Loading base model: {args.base_model}")
    proc_kwargs = {}
    if args.dataset == "sb_bench":
        if args.max_pixels and args.max_pixels > 0:
            proc_kwargs["max_pixels"] = args.max_pixels
        if args.min_pixels and args.min_pixels > 0:
            proc_kwargs["min_pixels"] = args.min_pixels
    processor = AutoProcessor.from_pretrained(args.base_model, **proc_kwargs)
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    # ── 3. Register erasure hooks ───────────────────────────────────────
    path, layers_list = resolve_layers(model)
    n_layers = len(layers_list)
    print(f"▶ Resolved LM-decoder layers via `{path}` (count={n_layers})")

    handles = []
    for L, w_np in pairs:
        if L < 0 or L >= n_layers:
            raise IndexError(
                f"layer index {L} out of range [0, {n_layers - 1}] for `{path}`"
            )
        v_unit = torch.from_numpy(w_np).contiguous()           # (D,) float32
        layer_mod = layers_list[L]
        h = layer_mod.register_forward_hook(make_erasure_hook(v_unit))
        handles.append(h)
        print(f"  registered erasure hook at {path}[{L}]")

    # ── 4. Load dataset ─────────────────────────────────────────────────
    print(f"▶ Loading dataset: {args.dataset} from {args.data_path}")
    if args.dataset == "sb_bench":
        ds = load_sb_bench(args)
    else:
        ds = load_vlbias(args)

    abc_ids = resolve_letter_ids(processor.tokenizer)
    print(f"  letter token ids (A,B,C) = {abc_ids}")

    # ── 5. Generate ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    try:
        if args.dataset == "sb_bench":
            results = generate_sb_bench(model, processor, ds, args, abc_ids, device, erasure_tag)
        else:
            results = generate_vlbias(model, processor, ds, args, abc_ids, device, erasure_tag)
    finally:
        for h in handles:
            h.remove()
        print(f"▶ Removed {len(handles)} hooks")

    # ── 6. Write JSONL ─────────────────────────────────────────────────
    print(f"▶ Saving {len(results)} outputs to {args.output_jsonl}")
    with open(args.output_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("✅ Generation complete.")


if __name__ == "__main__":
    main()
