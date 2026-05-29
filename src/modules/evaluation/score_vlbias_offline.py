"""
score_vlbias_offline.py
─────────────────────────────────────────────────────────────────────────────
Phase 0.7 T1.1 — Offline reward evaluation on VLBiasBench responses.

Scores an existing (prompt, response) generation JSONL (produced by
`generate_vlbiasbench_answers.py`) with the SB-Bench reward base model
(`qwen2-vl-2b`) and a directory of DRM head .pth files (SVM or PCA), using
the SAME token-position embedding extraction the heads were built on
(default `post_letter` to match `generated_heads_letter_post_letter/` on
the Modal volume; override with --token_position if your heads were built
differently).

No PPO involved. No new training. Pure offline scoring.

What it answers (per (condition × correctness) cell, per head type):

    "When the base policy answered C on an ambig item where gold = C
     (correct unknown), how does the SVM reward score it vs the PCA reward?"
    "When the base policy answered the named option on a disambig (neg)
     item correctly, do the rewards still give it high reward?"

If PCA reward is high on `(disambig, model-chose-C)` and low on
`(disambig, model-chose-named-correctly)`, the reward itself prefers
"unknown" — the §5 mode-collapse is REWARD-side, not PPO-side.

Pre-requisites (paths on Modal volume by default):
  --gen_jsonl    : <variant>_vlbias_gen.jsonl  (per-record question_id, text, label, condition, bbq_axis)
  --parquet      : vlbiasbench_close_ended.parquet (for image_path join on question_id)
  --image_root   : root dir for image_path resolution
  --reward_base  : reward base model (default Qwen/Qwen2-VL-2B-Instruct)
  --heads_dir    : dir of *.pth component files (one per head)
  --kept_heads   : (optional) kept_heads_*.json from evaluate_drm_heads

Output JSON schema:
{
  "variant": "<tag from --variant>",
  "head_type": "<svm|pca>",
  "num_heads": K,
  "num_samples_scored": N,
  "by_cell": {
    "ambig::correct":    {"n": ..., "mean_reward": ..., "std_reward": ..., "mean_per_head": [K floats]},
    "ambig::incorrect":  {...},
    "neg::correct":      {...},
    "neg::incorrect":    {...},
    "non_neg::correct":  {...},
    "non_neg::incorrect":{...}
  },
  "by_bbq_axis_and_cell": { "<axis>::<cell>": {...}, ... },
  "global": {"mean_reward": ..., "std_reward": ...}
}
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from modules.training.drm_loader import load_pca_components
from modules.utils.model_architecture import create_custom_forward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gen_jsonl", required=True,
                   help="Path to <variant>_vlbias_gen.jsonl produced by generate_vlbiasbench_answers.py")
    p.add_argument("--parquet", required=True,
                   help="Path to vlbiasbench_close_ended.parquet (needed to recover image_path via question_id join)")
    p.add_argument("--image_root", required=True,
                   help="Root dir for image_path resolution")
    p.add_argument("--reward_base", default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="Reward base model HF id or local path (must match Phase 0.6 D3 head-build base)")
    p.add_argument("--heads_dir", required=True, nargs="+",
                   help="One or more directories containing component*.pth files. "
                        "Each gets its own output JSON. Pass multiple to score SVM + PCA in one forward pass.")
    p.add_argument("--kept_heads", default=None, nargs="*",
                   help="Optional kept_heads_<type>.json per heads_dir (positional match). "
                        "Use 'none' to skip filtering for a specific dir.")
    p.add_argument("--head_type", nargs="+", required=True,
                   help="Label per heads_dir (e.g. svm pca). Must match length of --heads_dir.")
    p.add_argument("--variant", required=True,
                   help="Tag for the policy that produced gen_jsonl (e.g. base, svm_ep1-50pct)")
    p.add_argument("--output_dir", required=True,
                   help="Directory for output JSONs (named <variant>__<head_type>__offline_reward.json)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Embedding extraction batch size. Matches generate_vlbiasbench_answers default region.")
    p.add_argument("--max_pixels", type=int, default=0,
                   help="Optional cap on image pixels (Qwen2.5-VL native arg). 0 = uncapped, matching the "
                        "generator pipeline. Set e.g. 200704 (256*28*28) only if you hit OOM at larger batches.")
    p.add_argument("--min_pixels", type=int, default=0,
                   help="Optional floor on image pixels (Qwen2.5-VL native arg). 0 = unset.")
    p.add_argument("--max_samples", type=int, default=0,
                   help="If >0, score only the first N rows (smoke test). Prefer --max_per_cell for stratified subsampling.")
    p.add_argument("--max_per_cell", type=int, default=0,
                   help="If >0, stratified subsample to N records per (bbq_axis x condition) cell. "
                        "10 axes x 3 conditions x N = 30N total records. Recommended: 20 (600 total).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--token_position", default="post_letter",
                   choices=["post_letter", "pre_letter"],
                   help="Embedding extraction position. Must match how the heads were built.")
    p.add_argument("--completion_format", default="letter",
                   choices=["letter"],
                   help="Reserved; only 'letter' is currently supported (assistant turn = single A/B/C).")
    return p.parse_args()


def build_prompt_text(context: str, question: str, ans0: str, ans1: str, ans2: str) -> str:  # noqa: F811
    """Re-exported from generate_vlbiasbench_answers.build_prompt (imported above).
    Defined here as a no-op shim so callers that import this name still work."""
    from modules.inference.generate_vlbiasbench_answers import build_prompt
    return build_prompt(context, question, ans0, ans1, ans2)


def resolve_letter_token_ids(tokenizer) -> List[int]:
    """Same multi-encoding resolution as extract.py — keep any single-token
    encoding plus the last-token of multi-token encodings."""
    ids = set()
    for letter in ("A", "B", "C"):
        for prefix in ("", " ", "\n"):
            enc = tokenizer.encode(f"{prefix}{letter}", add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
            elif len(enc) > 1:
                ids.add(enc[-1])
        for t_id in tokenizer.encode(letter, add_special_tokens=False):
            ids.add(t_id)
    return sorted(ids)


def load_records(gen_jsonl: str, parquet: str, max_samples: int,
                 max_per_cell: int = 0, seed: int = 42) -> List[dict]:
    """Load JSONL records and left-join image_path from the parquet on question_id.

    If max_per_cell > 0, stratified subsample to N records per
    (bbq_axis x condition) cell — produces a balanced sample for cell-wise
    reward comparisons. Otherwise honour max_samples (first-N).
    """
    rows = []
    with open(gen_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.read_parquet(parquet)
    qid_to_image = dict(zip(df["id"].astype(str), df["image_path"].astype(str)))

    merged = []
    missing_image = 0
    for r in rows:
        qid = str(r.get("question_id", ""))
        img = qid_to_image.get(qid)
        if not img:
            missing_image += 1
            continue
        r["image_path"] = img
        merged.append(r)

    if missing_image:
        print(f"⚠  {missing_image} JSONL rows had no matching image_path in parquet; skipped.")

    if max_per_cell and max_per_cell > 0:
        import random as _random
        rng = _random.Random(seed)
        by_cell: Dict[str, List[dict]] = defaultdict(list)
        for r in merged:
            # Match eval_vlbiasbench.py: fall back bbq_axis -> category -> 'Unknown'
            key = f"{r.get('bbq_axis', r.get('category', 'Unknown'))}::{r.get('condition', 'unknown')}"
            by_cell[key].append(r)
        sampled = []
        for key, items in by_cell.items():
            rng.shuffle(items)
            sampled.extend(items[:max_per_cell])
        rng.shuffle(sampled)
        print(f"→ stratified subsample: {len(sampled)} records across {len(by_cell)} cells (max {max_per_cell}/cell)")
        return sampled

    if max_samples and len(merged) > max_samples:
        merged = merged[:max_samples]
    return merged


# Import the EXACT prompt/parse helpers used by the generator + eval so the
# offline scorer sees byte-identical (text, image, letter) inputs.
from modules.inference.generate_vlbiasbench_answers import build_prompt as build_prompt_text
from modules.evaluation.eval_vlbiasbench import parse_choice as _eval_parse_choice


def parse_choice_letter(text: str) -> int:
    """Delegate to eval_vlbiasbench.parse_choice — must stay byte-identical so
    offline reward cells (correct/incorrect, ambig/neg/non_neg) line up with
    the PPO-eval headline numbers."""
    return _eval_parse_choice(text)


def correctness_cell(rec: dict) -> tuple[str, str]:
    """Return (condition, 'correct'|'incorrect') for a record — matches
    eval_vlbiasbench.main()'s is_correct definition: pred_idx == int(label)."""
    cond = rec.get("condition", "unknown")
    label = int(rec.get("label", -1))
    pred = parse_choice_letter(rec.get("text", ""))
    correct = (pred == label) and (pred != -1)
    return cond, ("correct" if correct else "incorrect")


def batched_iter(items: List[dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Validate parallel lists
    n_head_sets = len(args.heads_dir)
    if len(args.head_type) != n_head_sets:
        raise ValueError(f"--head_type ({len(args.head_type)}) must match --heads_dir ({n_head_sets})")
    kept_heads_list: List[str | None] = [None] * n_head_sets
    if args.kept_heads:
        for i, kh in enumerate(args.kept_heads):
            if i >= n_head_sets:
                break
            kept_heads_list[i] = None if kh.lower() == "none" else kh

    print(f"▶ Loading records from {args.gen_jsonl}")
    records = load_records(args.gen_jsonl, args.parquet, args.max_samples,
                           max_per_cell=args.max_per_cell, seed=args.seed)
    print(f"  → {len(records)} records ready to score")

    print(f"▶ Loading reward base model: {args.reward_base}")
    # Mirror generate_vlbiasbench_answers.py processor init: no trust_remote_code
    # (the generator omits it), padding_side='left' (left-padding is required for
    # last-token / post_letter embedding extraction in batched runs anyway), and
    # optional pixel caps only when explicitly requested.
    proc_kwargs: dict = {}
    if args.min_pixels and args.min_pixels > 0:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels and args.max_pixels > 0:
        proc_kwargs["max_pixels"] = args.max_pixels
    if proc_kwargs:
        print(f"  image pixel cap: {proc_kwargs}")
    else:
        print("  image pixel cap: none (matches generator pipeline)")
    processor = AutoProcessor.from_pretrained(args.reward_base, **proc_kwargs)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Match generator's model init: flash-attn-2 on CUDA, device_map='auto'.
    model = AutoModelForImageTextToText.from_pretrained(
        args.reward_base,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    model.config.num_labels = 1
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    letter_ids = resolve_letter_token_ids(processor.tokenizer)
    print(f"  → letter_token_ids = {letter_ids}")

    forward_fn = create_custom_forward(
        model, dtype=torch.bfloat16,
        token_position=args.token_position,
        letter_token_ids=letter_ids,
    )
    model.forward = forward_fn.__get__(model, type(model))

    # Load ALL head sets up front — we apply them all to the same embeddings.
    all_heads: List[torch.Tensor] = []
    for i in range(n_head_sets):
        print(f"▶ Loading heads [{args.head_type[i]}] from {args.heads_dir[i]}")
        h = load_pca_components(
            heads_dir=args.heads_dir[i],
            num_heads=0,
            device=device,
            kept_heads_filter=kept_heads_list[i],
        )
        all_heads.append(h)
        print(f"  → loaded {h.shape[0]} heads of hidden_dim={h.shape[1]}")

    # ─── Single embedding pass, score all head sets ────────────────────────
    # per_head_rewards[i] = list of (K_i,) arrays, one per scored record
    per_head_rewards: List[List[np.ndarray]] = [[] for _ in range(n_head_sets)]
    scored_records: List[dict] = []

    pbar = tqdm(total=len(records), desc="Embedding+scoring")
    for batch in batched_iter(records, args.batch_size):
        images = []
        chats = []
        ok_indices_in_batch = []
        for j, rec in enumerate(batch):
            # Mirror generate_vlbiasbench_answers.py image loading: try the
            # full image_path, then basename, finally fall back to a gray 224x224
            # placeholder so the record isn't silently dropped (and stays aligned
            # with what the generator scored).
            img_rel = rec["image_path"]
            img_full = os.path.join(args.image_root, img_rel)
            if not os.path.exists(img_full):
                img_full = os.path.join(args.image_root, os.path.basename(img_rel))
            try:
                if os.path.exists(img_full):
                    img = Image.open(img_full).convert("RGB")
                else:
                    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            except Exception as e:
                print(f"⚠  qid={rec.get('question_id')} image load failed ({e}); using gray placeholder")
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))

            prompt_text = build_prompt_text(
                rec["context"], rec["question"], rec["ans0"], rec["ans1"], rec["ans2"]
            )
            pred_idx = parse_choice_letter(rec.get("text", ""))
            if pred_idx not in (0, 1, 2):
                img.close()
                continue
            letter = "ABC"[pred_idx]

            msg = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": letter}]},
            ]
            images.append(img)
            chats.append(msg)
            ok_indices_in_batch.append(j)

        if not chats:
            pbar.update(len(batch))
            continue

        chat_texts = [processor.apply_chat_template(m, tokenize=False) for m in chats]
        # Match generator's processor() call: padding=True, return_tensors='pt'.
        # No truncation / max_length — a single-letter assistant turn keeps the
        # sequence well within model max anyway, and skipping truncation removes
        # one more divergence from the generator pipeline.
        inputs = processor(
            text=chat_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        for img in images:
            img.close()

        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                out = model(**inputs, return_dict=True)
            emb = out.hidden_states  # (B, hidden_dim)
            if emb.dim() != 2:
                raise RuntimeError(f"Expected 2D embeddings, got shape {tuple(emb.shape)}")
        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            print(f"⚠  CUDA OOM on batch of {len(chats)} (skipping, not falling back to text-only): {oom}")
            pbar.update(len(batch))
            continue

        # Score with ALL head sets in one go (cheap matmuls).
        for i, heads in enumerate(all_heads):
            rewards = (emb.to(heads.dtype) @ heads.t()).float().cpu().numpy()
            for row in rewards:
                per_head_rewards[i].append(row)

        for local_j in ok_indices_in_batch:
            scored_records.append(batch[local_j])

        pbar.update(len(batch))
    pbar.close()

    print(f"▶ Scored {len(scored_records)} / {len(records)} records")

    # ─── Aggregate per head set and write output ───────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(n_head_sets):
        ht = args.head_type[i]
        K = all_heads[i].shape[0]
        rewards_list = per_head_rewards[i]

        by_cell: Dict[str, dict] = defaultdict(_empty_cell)
        by_axis_cell: Dict[str, dict] = defaultdict(_empty_cell)
        per_head_sum_by_cell: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(K, dtype=np.float64))
        per_head_n_by_cell: Dict[str, int] = defaultdict(int)
        global_rewards: List[float] = []

        for rec, head_rewards in zip(scored_records, rewards_list):
            mean_r = float(head_rewards.mean())
            global_rewards.append(mean_r)

            cond, corr = correctness_cell(rec)
            cell = f"{cond}::{corr}"
            by_cell[cell]["rewards"].append(mean_r)
            by_cell[cell]["n"] += 1
            per_head_sum_by_cell[cell] += head_rewards.astype(np.float64)
            per_head_n_by_cell[cell] += 1

            axis = rec.get("bbq_axis", rec.get("category", "Unknown"))
            axis_cell = f"{axis}::{cell}"
            by_axis_cell[axis_cell]["rewards"].append(mean_r)
            by_axis_cell[axis_cell]["n"] += 1

        def _finalise(d: Dict[str, dict], include_per_head: bool):
            out = {}
            for k, v in d.items():
                arr = np.asarray(v["rewards"], dtype=np.float64)
                entry = {
                    "n": int(v["n"]),
                    "mean_reward": float(arr.mean()) if arr.size else None,
                    "std_reward": float(arr.std(ddof=0)) if arr.size else None,
                }
                if include_per_head and per_head_n_by_cell.get(k, 0) > 0:
                    entry["mean_per_head"] = (per_head_sum_by_cell[k] / per_head_n_by_cell[k]).tolist()
                out[k] = entry
            return out

        output_json = os.path.join(args.output_dir, f"{args.variant}__{ht}__offline_reward.json")
        summary = {
            "variant": args.variant,
            "head_type": ht,
            "heads_dir": args.heads_dir[i],
            "kept_heads": kept_heads_list[i],
            "num_heads": int(K),
            "num_samples_scored": len(scored_records),
            "num_samples_input": len(records),
            "by_cell": _finalise(by_cell, include_per_head=True),
            "by_bbq_axis_and_cell": _finalise(by_axis_cell, include_per_head=False),
            "global": {
                "mean_reward": float(np.mean(global_rewards)) if global_rewards else None,
                "std_reward": float(np.std(global_rewards)) if global_rewards else None,
            },
        }

        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ wrote {output_json}")

        print(f"\n── Decisive cells [{ht}] ──────────────────────────────────────")
        for cell in ("ambig::correct", "ambig::incorrect",
                     "neg::correct", "neg::incorrect",
                     "non_neg::correct", "non_neg::incorrect"):
            e = summary["by_cell"].get(cell)
            if e and e["mean_reward"] is not None:
                print(f"  {cell:<22} n={e['n']:>4}  mean_reward={e['mean_reward']:+.4f}  std={e['std_reward']:.4f}")


def _empty_cell():
    return {"n": 0, "rewards": []}


if __name__ == "__main__":
    main()
