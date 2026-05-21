"""
generate_sb_bench_answers.py
────────────────────────────────────────────────────────────────────────────
Generate model answers for the SB-Bench dataset.

SB-Bench is a multiple-choice bias benchmark. Each example has:
  - context   : background scene description
  - question  : the stereotype-probing question
  - ans0/1/2  : three candidate answers
  - label     : index (0/1/2) of the non-stereotypical (correct/fair) answer
  - category  : one of 9 social bias categories

The model receives:  "context question\nA) ans0\nB) ans1\nC) ans2"
and is expected to respond with the letter of the least-biased answer.

The output JSONL has one record per row:
  { "question_id": <int>, "text": <model raw output>, "category": <str>,
    "label": <int>, "ans0": ..., "ans1": ..., "ans2": ... }
"""

import argparse
import os
import json
import io
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image


CHOICE_MAP = {0: "A", 1: "B", 2: "C"}

SB_BENCH_CATEGORIES = [
    "Age", "Disability", "Gender", "Nationality",
    "Physical Appearance", "Race/Ethnicity", "Religion", "SES", "Sexual Orientation",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to LoRA checkpoint (omit for vanilla baseline)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to sb_bench_data.parquet")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Path to save generated outputs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size (lower than POPE — prompts are longer)")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"],
                        help="Which split to evaluate on (default: all)")
    parser.add_argument("--split_indices_path", type=str, default=None,
                        help="Path to split_indices.json for filtering")
    return parser.parse_args()


def build_prompt(context: str, question: str, ans0: str, ans1: str, ans2: str) -> str:
    """Format an SB-Bench example as a multiple-choice prompt."""
    return (
        f"{context} {question}\n"
        f"A) {ans0}\n"
        f"B) {ans1}\n"
        f"C) {ans2}\n"
        "Answer with only the letter (A, B, or C):"
    )


def resolve_category(cat_val) -> str:
    """Convert integer or string category to a readable label."""
    if isinstance(cat_val, int) and 0 <= cat_val < len(SB_BENCH_CATEGORIES):
        return SB_BENCH_CATEGORIES[cat_val]
    return str(cat_val)


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
        trust_remote_code=True
    )

    if args.checkpoint_dir:
        print(f"Loading LoRA Checkpoint: {args.checkpoint_dir}")
        model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    else:
        print("ℹ️  No --checkpoint_dir provided. Running vanilla Qwen2.5-VL-3B-Instruct (no LoRA).")
        model = base_model
    model.eval()

    print(f"Loading Dataset: {args.data_path}")
    ds = load_dataset("parquet", data_files={"test": args.data_path})["test"]
    print(f"  Total examples: {len(ds)}")

    # Filter to train/test split if requested
    if args.split in ("train", "test") and args.split_indices_path:
        with open(args.split_indices_path, "r") as f:
            split_info = json.load(f)
        indices = split_info["train_indices"] if args.split == "train" else split_info["test_indices"]
        ds = ds.select(indices)
        print(f"  Filtered to '{args.split}' split: {len(ds)} examples")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    results = []

    for i in tqdm(range(0, len(ds), args.batch_size), desc="Generating SB-Bench Answers"):
        batch = ds[i: i + args.batch_size]
        batch_size = len(batch["question"])

        batch_messages = []
        batch_images = []
        meta = []

        for j in range(batch_size):
            context = batch["context"][j]
            question = batch["question"][j]
            ans0 = batch["ans0"][j]
            ans1 = batch["ans1"][j]
            ans2 = batch["ans2"][j]
            label = int(batch["label"][j])
            category = resolve_category(batch["category"][j])

            prompt_text = build_prompt(context, question, ans0, ans1, ans2)

            # Load image
            img_field = batch.get("file_name", [None] * batch_size)[j]
            if img_field is None:
                img_field = batch.get("file_name.bytes", [None] * batch_size)[j]
            if isinstance(img_field, dict) and "bytes" in img_field:
                img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
            elif isinstance(img_field, bytes):
                img = Image.open(io.BytesIO(img_field)).convert("RGB")
            else:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))  # grey fallback

            batch_images.append(img)
            batch_messages.append([
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text}
                ]}
            ])
            meta.append({
                "question_id": i + j,
                "category": category,
                "label": label,
                "ans0": ans0, "ans1": ans1, "ans2": ans2,
                "context": context, "question": question,
            })

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        inputs = processor(
            text=texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=10)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        for m, text in zip(meta, output_texts):
            results.append({**m, "text": text.strip()})

    print(f"Saving {len(results)} outputs to {args.output_jsonl}")
    with open(args.output_jsonl, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print("Generation complete!")


if __name__ == "__main__":
    main()
