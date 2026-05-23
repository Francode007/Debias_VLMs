"""
Phase 0b — Reward / Metric Correlation Sanity Gate
─────────────────────────────────────────────────────────────────────────────
Does the SVM reward head agree with what the eval cares about?

For every SB-Bench test example:
  1. Compute h_{-2} at the last non-pad position of  (prompt + " A"), (prompt + " B"),
     (prompt + " C")  — three forward passes per item.
  2. Project each onto the per-category SVM head w_k :   s_letter = h · w_k.
  3. Compare:
        - ranking_acc  : does argmax_letter s_letter == correct letter?
        - margin_corr  : Pearson r between (s_correct − s_wrong)
                          and the model's own preference (logprob of letter).
  4. Distribution check:  how far does h(prompt + full chosen text) sit from
     h(prompt + letter) in the SVM's input space?
     Reports cosine sim & projection ratio.

INTERPRETATION
    - ranking_acc near chance (~33 %) → head provides no usable preference signal.
    - large distribution gap          → train/inference mismatch (Flaw A2).
"""

import argparse
import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

SB_BENCH_CATEGORIES = [
    "Age", "Disability", "Gender", "Nationality",
    "Physical Appearance", "Race/Ethnicity", "Religion", "SES", "Sexual Orientation",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--heads_dir", required=True)
    p.add_argument("--split_indices_path", default=None)
    p.add_argument("--split", default="test", choices=["train", "test", "all"])
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def load_heads(heads_dir: str) -> torch.Tensor:
    paths = sorted(
        Path(heads_dir).glob("*SVM-component*.pth"),
        key=lambda p: int(re.search(r"(\d+)\.pth$", p.name).group(1)),
    )
    weights = []
    for p in paths:
        obj = torch.load(p, map_location="cpu")
        w = obj["weight"] if isinstance(obj, dict) and "weight" in obj else obj
        weights.append(w.reshape(-1).float())
    return torch.stack(weights, dim=0)


def build_prompt(ctx, q, a0, a1, a2):
    return (f"{ctx} {q}\nA) {a0}\nB) {a1}\nC) {a2}\n"
            "Answer with only the letter (A, B, or C):")


def resolve_category_idx(cat_val) -> int:
    if isinstance(cat_val, int) and 0 <= cat_val < len(SB_BENCH_CATEGORIES):
        return cat_val
    s = str(cat_val)
    for i, name in enumerate(SB_BENCH_CATEGORIES):
        if name.lower() == s.lower():
            return i
    return -1


@torch.inference_mode()
def hidden_at_last(model, processor, image: Image.Image, prompt: str,
                   suffix: str, device: str) -> torch.Tensor:
    """Return penultimate-layer hidden state at the last non-pad position
    for input = (prompt + suffix)."""
    msg = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    text = text + suffix
    inputs = processor(text=[text], images=[image], padding=True,
                       return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-2]  # (1, T, D)
    attn = inputs["attention_mask"]
    last_idx = attn.sum(dim=1).long() - 1
    return h[0, last_idx[0]].float().cpu()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        device_map="auto" if device == "cuda" else None, trust_remote_code=True,
    )
    model.eval()

    heads = load_heads(args.heads_dir)  # (K, D), float32

    ds = load_dataset("parquet", data_files={"test": args.data_path})["test"]
    if args.split in ("train", "test") and args.split_indices_path:
        with open(args.split_indices_path, "r") as f:
            split_info = json.load(f)
        key = "train_indices" if args.split == "train" else "test_indices"
        ds = ds.select(split_info[key])
    if args.num_samples and args.num_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(args.num_samples))

    per_category: Dict[int, Dict] = {}
    dist_records: List[Dict] = []
    rank_total = 0
    rank_correct = 0

    for i in tqdm(range(len(ds)), desc="probing"):
        row = ds[i]
        cat_idx = resolve_category_idx(row["category"])
        if cat_idx < 0 or cat_idx >= heads.shape[0]:
            continue
        ans_texts = [row["ans0"], row["ans1"], row["ans2"]]
        label = int(row["label"])
        correct_letter = ["A", "B", "C"][label]

        img_field = row.get("file_name") or row.get("file_name.bytes")
        if isinstance(img_field, dict) and "bytes" in img_field:
            img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        elif isinstance(img_field, bytes):
            img = Image.open(io.BytesIO(img_field)).convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        prompt = build_prompt(row["context"], row["question"], *ans_texts)
        w = heads[cat_idx]

        scores = {}
        for letter in ("A", "B", "C"):
            h = hidden_at_last(model, processor, img, prompt, f" {letter}", device)
            scores[letter] = float((h * w).sum().item())

        # Compare against full chosen text (training distribution)
        chosen_text = ans_texts[label]
        h_full = hidden_at_last(model, processor, img, prompt, f" {chosen_text}", device)
        h_letter = hidden_at_last(model, processor, img, prompt, f" {correct_letter}", device)
        full_score = float((h_full * w).sum().item())
        letter_score = float((h_letter * w).sum().item())
        cos = float(torch.nn.functional.cosine_similarity(
            h_full.unsqueeze(0), h_letter.unsqueeze(0)).item())
        dist_records.append({
            "category": SB_BENCH_CATEGORIES[cat_idx],
            "score_full_chosen": full_score,
            "score_correct_letter": letter_score,
            "cosine_full_vs_letter": cos,
        })

        pred_letter = max(scores, key=scores.get)
        is_correct = (pred_letter == correct_letter)
        rank_total += 1
        rank_correct += int(is_correct)

        c = per_category.setdefault(cat_idx, {"n": 0, "correct": 0,
                                              "margin_sum": 0.0})
        c["n"] += 1
        c["correct"] += int(is_correct)
        wrong_scores = [s for L, s in scores.items() if L != correct_letter]
        c["margin_sum"] += scores[correct_letter] - max(wrong_scores)

    overall = {
        "samples": rank_total,
        "ranking_accuracy_overall": rank_correct / max(rank_total, 1),
        "per_category": {
            SB_BENCH_CATEGORIES[k]: {
                "n": v["n"],
                "ranking_accuracy": v["correct"] / max(v["n"], 1),
                "mean_margin_correct_minus_wrong": v["margin_sum"] / max(v["n"], 1),
            } for k, v in per_category.items()
        },
        "distribution_gap": {
            "n": len(dist_records),
            "mean_score_full_chosen": float(np.mean([r["score_full_chosen"] for r in dist_records])),
            "mean_score_correct_letter": float(np.mean([r["score_correct_letter"] for r in dist_records])),
            "mean_cosine_full_vs_letter": float(np.mean([r["cosine_full_vs_letter"] for r in dist_records])),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(overall, f, indent=2)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
