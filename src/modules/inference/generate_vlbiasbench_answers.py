"""
generate_vlbiasbench_answers.py
────────────────────────────────────────────────────────────────────────────
Generate model answers for the VLBiasBench close-ended split (G4a transfer
evaluation, Phase 0.7).

Mirrors `generate_sb_bench_answers.py` 1:1 — same constrained-decode pipeline,
same prompt format, same JSONL schema — with VLBiasBench-specific extras:

  * `condition` (ambig | neg | non_neg) is carried through. The CLI flag
    `--condition` accepts {all, ambig, disambig}; "disambig" maps internally
    to `condition ∈ {neg, non_neg}` (BBQ convention).
  * `--qformat` filters question-format dirs (default: base,scene,scene_text —
    excludes the text-only `text` split which has no visual signal).
  * `--num_samples N` does stratified sampling across (bbq_axis × condition)
    so all 10 BBQ axes and both condition groups stay represented.

Output JSONL per record:
    { "question_id", "text", "bbq_axis", "qformat", "subgroup",
      "condition", "label", "ans0", "ans1", "ans2", "context", "question" }
"""

import argparse
import json
import os
import random

import torch
from PIL import Image
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


CHOICE_MAP = {0: "A", 1: "B", 2: "C"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Path to LoRA checkpoint (omit for vanilla baseline)")
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to vlbiasbench_close_ended.parquet")
    p.add_argument("--image_root", type=str,
                   default="/mnt/data/vlbiasbench_data/unpacked/close_ended/images",
                   help="Root directory for VLBiasBench images (image_path column is relative to this)")
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=0,
                   help="If >0, randomly subsample N rows stratified by category. "
                        "0 means use the full dataset.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for stratified subsampling.")
    p.add_argument("--condition", type=str, default="all",
                   choices=["all", "ambig", "disambig"],
                   help="Restrict by condition. 'disambig' = condition ∈ {neg, non_neg}.")
    p.add_argument("--qformat", type=str, default="base,scene,scene_text",
                   help="Comma-separated question-format dirs to keep "
                        "(subset of {base,scene,scene_text,text}). "
                        "Default excludes 'text' (no visual signal).")
    return p.parse_args()


def build_prompt(context: str, question: str, ans0: str, ans1: str, ans2: str) -> str:
    return (
        f"{context} {question}\n"
        f"A) {ans0}\n"
        f"B) {ans1}\n"
        f"C) {ans2}\n"
        "Answer with only the letter (A, B, or C):"
    )


def resolve_letter_ids(tokenizer):
    """Single-token ids for ' A' / ' B' / ' C' (fallback no-space). Same logic
    as generate_sb_bench_answers.py to keep constrained decoding consistent."""
    ids = []
    for letter in ("A", "B", "C"):
        cand = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(cand) != 1:
            cand = tokenizer.encode(letter, add_special_tokens=False)
        if len(cand) != 1:
            raise ValueError(
                f"Tokenizer does not encode '{letter}' as a single token; "
                f"constrained decoding requires it."
            )
        ids.append(cand[0])
    return tuple(ids)


def _stratified_subsample(ds, n: int, seed: int):
    """Pick `n` indices stratified by (bbq_axis, condition) to keep every
    (axis, condition) cell represented. Falls back to flat-random if those
    fields are missing."""
    if n <= 0 or n >= len(ds):
        return ds
    cols = ds.column_names
    if "bbq_axis" in cols and "condition" in cols:
        axes = ds["bbq_axis"]
        conds = ds["condition"]
        key_iter = zip(axes, conds)
    else:
        key_iter = ((c,) for c in ds["category"])
    buckets: dict = {}
    for i, k in enumerate(key_iter):
        buckets.setdefault(k, []).append(i)
    rng = random.Random(seed)
    chosen = []
    keys = list(buckets.keys())
    for k in keys:
        rng.shuffle(buckets[k])
    pointers = {k: 0 for k in keys}
    while len(chosen) < n:
        progressed = False
        for k in keys:
            if pointers[k] < len(buckets[k]):
                chosen.append(buckets[k][pointers[k]])
                pointers[k] += 1
                progressed = True
                if len(chosen) >= n:
                    break
        if not progressed:
            break
    chosen.sort()
    return ds.select(chosen)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Base Model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.tokenizer.padding_side = "left"

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if args.checkpoint_dir:
        print(f"Loading LoRA Checkpoint: {args.checkpoint_dir}")
        adapter_cfg = os.path.join(args.checkpoint_dir, "adapter_config.json")
        if not os.path.isdir(args.checkpoint_dir) or not os.path.isfile(adapter_cfg):
            raise FileNotFoundError(
                f"Checkpoint dir invalid (no adapter_config.json): {args.checkpoint_dir}\n"
                f"  Hint: list available checkpoints with `modal volume ls "
                f"debias-vlm-persistent-storage /<path>`"
            )
        model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    else:
        print("ℹ️  No --checkpoint_dir → running vanilla Qwen2.5-VL-3B-Instruct.")
        model = base_model
    model.eval()

    print(f"Loading Dataset: {args.data_path}")
    ds = load_dataset("parquet", data_files={"test": args.data_path})["test"]
    print(f"  Total examples: {len(ds)}")

    # qformat filter (e.g. drop the text-only split).
    keep_qf = {x.strip() for x in args.qformat.split(",") if x.strip()}
    if "qformat" in ds.column_names and keep_qf:
        keep = [i for i, qf in enumerate(ds["qformat"]) if qf in keep_qf]
        ds = ds.select(keep)
        print(f"  qformat ∈ {sorted(keep_qf)} → {len(ds)} rows")

    # condition filter (with disambig = {neg, non_neg}).
    if args.condition == "ambig":
        keep = [i for i, c in enumerate(ds["condition"]) if c == "ambig"]
        ds = ds.select(keep)
        print(f"  condition='ambig' → {len(ds)} rows")
    elif args.condition == "disambig":
        keep = [i for i, c in enumerate(ds["condition"]) if c in ("neg", "non_neg")]
        ds = ds.select(keep)
        print(f"  condition∈{{neg,non_neg}} (disambig) → {len(ds)} rows")

    # Drop rows with no resolvable gold label.
    label_keep = [i for i, l in enumerate(ds["label"]) if int(l) >= 0]
    if len(label_keep) != len(ds):
        ds = ds.select(label_keep)
        print(f"  Dropped rows with label=-1; remaining: {len(ds)}")

    if args.num_samples > 0:
        ds = _stratified_subsample(ds, args.num_samples, args.seed)
        print(f"  Stratified subsample → {len(ds)} rows")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    results = []
    abc_ids = None

    for i in tqdm(range(0, len(ds), args.batch_size), desc="Generating VLBiasBench Answers"):
        batch = ds[i: i + args.batch_size]
        bsz = len(batch["question"])

        batch_messages = []
        batch_images = []
        meta = []

        for j in range(bsz):
            context = batch["context"][j]
            question = batch["question"][j]
            ans0 = batch["ans0"][j]
            ans1 = batch["ans1"][j]
            ans2 = batch["ans2"][j]
            label = int(batch["label"][j])
            bbq_axis = str(batch["bbq_axis"][j]) if "bbq_axis" in batch else ""
            qformat = str(batch["qformat"][j]) if "qformat" in batch else ""
            subgroup = str(batch["subgroup"][j]) if "subgroup" in batch else ""
            condition = str(batch["condition"][j])

            prompt_text = build_prompt(context, question, ans0, ans1, ans2)

            # Load image from disk (image_path column is relative to --image_root).
            img_rel = batch["image_path"][j]
            img_full = os.path.join(args.image_root, img_rel)
            if not os.path.exists(img_full):
                # Fallback: try basename only.
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
            })

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]

        inputs = processor(
            text=texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            if abc_ids is None:
                abc_ids = resolve_letter_ids(processor.tokenizer)
            letter_logits = next_token_logits[:, list(abc_ids)]
            picks = letter_logits.argmax(dim=-1).tolist()

        output_texts = [CHOICE_MAP[p] for p in picks]
        for m, text in zip(meta, output_texts):
            results.append({**m, "text": text.strip()})

    print(f"Saving {len(results)} outputs to {args.output_jsonl}")
    with open(args.output_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("Generation complete!")


if __name__ == "__main__":
    main()
