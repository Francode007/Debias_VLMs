"""
probe_layers.py
─────────────────────────────────────────────────────────────────────────────
Phase 0.8 A1 — Per-layer linear probes on Qwen2.5-VL-3B hidden states.

Tests the multi-layer hypothesis (Phase0.8.md §0): does a layer L* < penultimate
encode the (correctness, good-C-vs-bad-C) distinction more cleanly than the
penultimate hidden state that current SVM / PCA DRM heads are built on?

What it does
────────────
1. Loads a base-policy generation JSONL produced by
   `generate_vlbiasbench_answers.py` (gold labels + model-emitted letter).
2. For each record: rebuilds the SAME (image + chat + assistant=letter) tensor
   the offline scorer uses, then runs a single model forward with
   `output_hidden_states=True`.
3. For EVERY LM decoder layer (embeddings + transformer blocks, ~37 for
   Qwen2.5-VL-3B), captures the hidden state at the `post_letter` token
   position — the same position the SB-Bench heads were built on.

   NOTE on vision-encoder layers: the `post_letter` position lives in the
   *text* token stream of the LM decoder; vision-encoder layers operate on
   image patches and have no comparable per-record probe target. This A1
   therefore probes LM-decoder layers only (≈37). Vision-side bias probing
   is a separate experiment deferred to Tier-B if A1+A2 fall through.

4. Trains three binary logistic regressions per layer with 5-fold stratified
   CV (sklearn):
     P1: condition ∈ {ambig, disambig}            (sanity)
     P2: correctness on disambig items only       (does the LM "know" the right answer?)
     P3: is_good_C — among model-emitted-C records,
         gold == C  ("legit unknown")  vs  gold == named option ("bias-driven C").
         **Decisive probe.** If no layer separates these cleanly, the
         multi-layer direction is falsified.

5. Optional holdout (`--holdout_gen_jsonl`): if a separate gen file with
   `qformat=text` records is provided, P3 is trained on the primary file and
   evaluated on the holdout. This is the bias-blind selection criterion
   (Phase0.8.md §1, §2). Without a holdout, only in-distribution CV scores
   are reported and the script prints a warning.

Output
──────
{
  "variant": "base",
  "n_records_scored": int,
  "letter_token_ids": [...],
  "token_position": "post_letter",
  "layer_count": int,
  "hidden_dim": int,
  "per_layer": [
    {"layer_idx": 0, "layer_type": "lm",
     "p1_acc_cv": float, "p1_acc_cv_std": float, "n_p1": int,
     "p2_acc_cv": float, "p2_acc_cv_std": float, "n_p2": int,
     "p3_acc_cv": float, "p3_acc_cv_std": float, "n_p3": int,
     "p3_acc_holdout": float | null, "n_p3_holdout": int},
    ...
  ]
}

Also writes an optional `<variant>_layerwise_probe.png` curve.

Usage
─────
python -m modules.evaluation.probe_layers \
    --gen_jsonl /mnt/data/phase07_vlbiasbench/base_vlbias_gen.jsonl \
    --parquet   /mnt/data/vlbiasbench_data/vlbiasbench_close_ended.parquet \
    --image_root /mnt/data/vlbiasbench_data/unpacked/close_ended/images \
    --reward_base Qwen/Qwen2.5-VL-3B-Instruct \
    --variant base \
    --output_dir /mnt/data/phase08_probe_results \
    --batch_size 4 --max_per_cell 30 \
    --holdout_gen_jsonl /mnt/data/phase07_vlbiasbench/base_qformat_text_vlbias_gen.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# Reuse the canonical helpers — keeps inputs byte-identical to the scorer.
from modules.inference.generate_vlbiasbench_answers import build_prompt as build_prompt_text
from modules.evaluation.eval_vlbiasbench import parse_choice as _eval_parse_choice
from modules.evaluation.score_vlbias_offline import (
    resolve_letter_token_ids,
    load_records,
    correctness_cell,
    batched_iter,
)


# ─── Arg parsing ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gen_jsonl", required=True,
                   help="Primary base-policy generation JSONL "
                        "(qformat ∈ {base,scene,scene_text} expected).")
    p.add_argument("--holdout_gen_jsonl", default=None,
                   help="Optional second generation JSONL whose records form the "
                        "P3 holdout (recommended: qformat=text base-policy gen).")
    p.add_argument("--parquet", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--reward_base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--variant", default="base")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_pixels", type=int, default=0)
    p.add_argument("--min_pixels", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_per_cell", type=int, default=0,
                   help="Stratified subsample per (bbq_axis × condition) cell on the PRIMARY file. "
                        "0 = use all primary records.")
    p.add_argument("--max_holdout", type=int, default=0,
                   help="Cap on holdout records (0 = use all). "
                        "Phase0.8.md targets ~100 records.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--token_position", default="post_letter",
                   choices=["post_letter", "pre_letter"])
    p.add_argument("--cache_npz", default=None,
                   help="Optional .npz path to cache per-layer hidden states (saves rerun cost).")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--logreg_C", type=float, default=1.0,
                   help="sklearn LogisticRegression C (inverse regularisation strength).")
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--save_plot", action="store_true",
                   help="Render PNG of per-layer p1/p2/p3 accuracies.")
    return p.parse_args()


# ─── Position lookup ────────────────────────────────────────────────────────
def find_letter_positions(input_ids: torch.Tensor, letter_ids: torch.Tensor,
                          token_position: str) -> torch.Tensor:
    """Return per-row index of the post_letter (or pre_letter) token in
    `input_ids`. Mirrors the logic baked into create_custom_forward to keep
    extraction byte-identical to the SB-Bench reward heads.

    Falls back to the last non-pad position (sequence_length - 1) for rows
    with no letter token; those rows will be filtered out by `--filter_letter_only`
    handling in main() before probing.
    """
    device = input_ids.device
    is_letter = torch.isin(input_ids, letter_ids.to(device))
    T = input_ids.shape[-1]
    rev = torch.flip(is_letter.int(), dims=[-1])
    any_letter = is_letter.any(dim=-1)
    offset_from_end = rev.argmax(dim=-1).long()
    letter_pos = (T - 1) - offset_from_end
    # Default fallback = last token index (T-1). Rows with no letter are
    # filtered upstream, but we keep a safe index here.
    fallback = torch.full_like(letter_pos, T - 1)
    letter_pos = torch.where(any_letter, letter_pos, fallback)
    if token_position == "post_letter":
        return letter_pos
    return (letter_pos - 1).clamp(min=0)


# ─── Extraction loop ────────────────────────────────────────────────────────
def extract_hidden_states(model, processor, records: List[dict], image_root: str,
                          letter_ids_list: List[int], token_position: str,
                          batch_size: int, device: torch.device,
                          ) -> tuple[np.ndarray, List[dict]]:
    """Run the model on each record and return:
       hs:  ndarray (N, L, D)  — hidden state at chosen position from every layer
       kept_records: List[dict] — records that survived (had a valid letter, no OOM)

    Layer count L includes the input-embeddings entry that HF returns as
    hidden_states[0]; so for Qwen2.5-VL-3B (36 LM blocks) L = 37.
    """
    letter_ids_tensor = torch.tensor(sorted(set(int(x) for x in letter_ids_list)),
                                     dtype=torch.long)
    all_hs: List[np.ndarray] = []
    kept_records: List[dict] = []

    pbar = tqdm(total=len(records), desc="Forward+collect")
    for batch in batched_iter(records, batch_size):
        images = []
        chats = []
        batch_kept_idx: List[int] = []

        for j, rec in enumerate(batch):
            img_rel = rec["image_path"]
            img_full = os.path.join(image_root, img_rel)
            if not os.path.exists(img_full):
                img_full = os.path.join(image_root, os.path.basename(img_rel))
            try:
                if os.path.exists(img_full):
                    img = Image.open(img_full).convert("RGB")
                else:
                    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            except Exception as e:
                print(f"⚠  qid={rec.get('question_id')} image load failed ({e})")
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))

            prompt_text = build_prompt_text(
                rec["context"], rec["question"], rec["ans0"], rec["ans1"], rec["ans2"]
            )
            pred_idx = _eval_parse_choice(rec.get("text", ""))
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
            batch_kept_idx.append(j)

        if not chats:
            pbar.update(len(batch))
            continue

        chat_texts = [processor.apply_chat_template(m, tokenize=False) for m in chats]
        inputs = processor(
            text=chat_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        for img in images:
            img.close()
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        try:
            with torch.no_grad():
                out = model(**inputs,
                            return_dict=True,
                            output_hidden_states=True,
                            use_cache=False)
        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            print(f"⚠  CUDA OOM on batch of {len(chats)} (skipped): {oom}")
            pbar.update(len(batch))
            continue

        hidden_states = out.hidden_states  # tuple of (B, T, D), length = L
        if hidden_states is None:
            raise RuntimeError("output_hidden_states=True returned None; "
                               "model may not propagate hidden states from the wrapper.")

        input_ids = inputs["input_ids"]
        slice_idx = find_letter_positions(input_ids, letter_ids_tensor, token_position)
        slice_idx = slice_idx.to(hidden_states[0].device)

        B = input_ids.shape[0]
        L = len(hidden_states)
        D = hidden_states[0].shape[-1]
        batch_arange = torch.arange(B, device=hidden_states[0].device)
        # Stack to (L, B, D) → (B, L, D)
        per_layer = torch.stack(
            [layer_h[batch_arange, slice_idx] for layer_h in hidden_states],
            dim=0,
        )  # (L, B, D)
        per_layer = per_layer.transpose(0, 1).contiguous()  # (B, L, D)
        all_hs.append(per_layer.float().cpu().numpy())

        for local_j in batch_kept_idx:
            kept_records.append(batch[local_j])

        # Free the big GPU tensors before the next batch.
        del out, hidden_states, per_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.update(len(batch))
    pbar.close()

    if not all_hs:
        raise RuntimeError("No records survived extraction.")
    hs = np.concatenate(all_hs, axis=0)  # (N, L, D)
    return hs, kept_records


# ─── Label builders ─────────────────────────────────────────────────────────
def _condition_binary(rec: dict) -> Optional[int]:
    cond = rec.get("condition", "")
    if cond == "ambig":
        return 0
    if cond in ("neg", "non_neg"):
        return 1
    return None


def _is_disambig_correct(rec: dict) -> Optional[int]:
    cond = rec.get("condition", "")
    if cond not in ("neg", "non_neg"):
        return None
    label = int(rec.get("label", -1))
    pred = _eval_parse_choice(rec.get("text", ""))
    if pred not in (0, 1, 2) or label < 0:
        return None
    return 1 if pred == label else 0


def _is_good_C(rec: dict) -> Optional[int]:
    """1 = legit C (gold == C), 0 = bias-driven C (gold == named option).
    Returns None if model didn't emit C, regardless of condition."""
    pred = _eval_parse_choice(rec.get("text", ""))
    if pred != 2:                       # only model-emitted-C records
        return None
    label = int(rec.get("label", -1))
    if label == 2:
        return 1
    if label in (0, 1):
        return 0
    return None


def _build_labels(records: List[dict]):
    p1 = [_condition_binary(r) for r in records]
    p2 = [_is_disambig_correct(r) for r in records]
    p3 = [_is_good_C(r) for r in records]
    return p1, p2, p3


# ─── Probing ────────────────────────────────────────────────────────────────
def _probe_one_layer(X: np.ndarray, y_list: List[Optional[int]],
                     cv_folds: int, C: float, max_iter: int, seed: int
                     ) -> tuple[float, float, int]:
    """Stratified k-fold logistic regression accuracy on (X, y).
    Returns (mean_acc, std_acc, n)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    mask = np.array([y is not None for y in y_list])
    if mask.sum() < 2 * cv_folds:
        return (float("nan"), float("nan"), int(mask.sum()))
    Xv = X[mask]
    yv = np.array([y for y in y_list if y is not None], dtype=np.int64)
    classes = np.unique(yv)
    if classes.size < 2:
        # Degenerate label distribution — report the trivial majority baseline.
        return (1.0, 0.0, int(yv.size))
    min_class = np.min(np.bincount(yv))
    if min_class < cv_folds:
        # Not enough minority samples for stratified folds — fall back to a
        # single 80/20 split.
        rng = np.random.default_rng(seed)
        n = yv.size
        perm = rng.permutation(n)
        cut = max(1, int(n * 0.2))
        te, tr = perm[:cut], perm[cut:]
        clf = LogisticRegression(C=C, max_iter=max_iter,
                                 class_weight="balanced", n_jobs=1)
        clf.fit(Xv[tr], yv[tr])
        acc = float(clf.score(Xv[te], yv[te]))
        return (acc, 0.0, int(yv.size))

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    accs = []
    for tr_idx, te_idx in skf.split(Xv, yv):
        clf = LogisticRegression(C=C, max_iter=max_iter,
                                 class_weight="balanced", n_jobs=1)
        clf.fit(Xv[tr_idx], yv[tr_idx])
        accs.append(clf.score(Xv[te_idx], yv[te_idx]))
    return (float(np.mean(accs)), float(np.std(accs)), int(yv.size))


def _probe_holdout(X_tr: np.ndarray, y_tr: List[Optional[int]],
                   X_ho: np.ndarray, y_ho: List[Optional[int]],
                   C: float, max_iter: int) -> tuple[Optional[float], int]:
    from sklearn.linear_model import LogisticRegression

    mask_tr = np.array([y is not None for y in y_tr])
    mask_ho = np.array([y is not None for y in y_ho])
    if mask_tr.sum() < 10 or mask_ho.sum() < 5:
        return (None, int(mask_ho.sum()))
    Xtr = X_tr[mask_tr]
    ytr = np.array([y for y in y_tr if y is not None], dtype=np.int64)
    if np.unique(ytr).size < 2:
        return (None, int(mask_ho.sum()))
    Xho = X_ho[mask_ho]
    yho = np.array([y for y in y_ho if y is not None], dtype=np.int64)
    clf = LogisticRegression(C=C, max_iter=max_iter,
                             class_weight="balanced", n_jobs=1)
    clf.fit(Xtr, ytr)
    return (float(clf.score(Xho, yho)), int(yho.size))


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load primary records ────────────────────────────────────────
    print(f"▶ Loading PRIMARY records from {args.gen_jsonl}")
    primary = load_records(args.gen_jsonl, args.parquet,
                           max_samples=args.max_samples,
                           max_per_cell=args.max_per_cell,
                           seed=args.seed)
    print(f"  → {len(primary)} primary records")

    holdout = []
    if args.holdout_gen_jsonl and os.path.exists(args.holdout_gen_jsonl):
        print(f"▶ Loading HOLDOUT records from {args.holdout_gen_jsonl}")
        holdout = load_records(args.holdout_gen_jsonl, args.parquet,
                               max_samples=args.max_holdout,
                               max_per_cell=0, seed=args.seed)
        if args.max_holdout > 0 and len(holdout) > args.max_holdout:
            import random as _r
            _r.Random(args.seed).shuffle(holdout)
            holdout = holdout[:args.max_holdout]
        print(f"  → {len(holdout)} holdout records")
    else:
        print("⚠  No holdout_gen_jsonl provided — p3_acc_holdout will be null. "
              "Layer pre-registration MUST then use a held-out criterion you "
              "decide separately (Phase0.8.md §2). Recommended fix: regen "
              "qformat=text and pass it via --holdout_gen_jsonl.")

    # ── 2. Load model + processor (mirror score_vlbias_offline) ────────
    print(f"▶ Loading reward base model: {args.reward_base}")
    proc_kwargs: dict = {}
    if args.min_pixels and args.min_pixels > 0:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels and args.max_pixels > 0:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.reward_base, **proc_kwargs)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

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
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    letter_ids = resolve_letter_token_ids(processor.tokenizer)
    print(f"  → letter_token_ids = {letter_ids}")

    # ── 3. Extract per-layer hidden states ─────────────────────────────
    cache_path = args.cache_npz
    if cache_path and os.path.exists(cache_path):
        print(f"▶ Loading cached hidden states from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        hs_primary = cached["hs_primary"]
        kept_primary = list(cached["kept_primary"])
        hs_holdout = cached["hs_holdout"] if "hs_holdout" in cached.files else None
        kept_holdout = list(cached["kept_holdout"]) if "kept_holdout" in cached.files else []
        if not isinstance(kept_primary[0], dict):
            # numpy stored dicts as 0-d arrays; recover
            kept_primary = [k.item() if hasattr(k, "item") else k for k in kept_primary]
            kept_holdout = [k.item() if hasattr(k, "item") else k for k in kept_holdout]
    else:
        hs_primary, kept_primary = extract_hidden_states(
            model, processor, primary, args.image_root,
            letter_ids, args.token_position, args.batch_size, device,
        )
        hs_holdout = None
        kept_holdout = []
        if holdout:
            hs_holdout, kept_holdout = extract_hidden_states(
                model, processor, holdout, args.image_root,
                letter_ids, args.token_position, args.batch_size, device,
            )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            save_dict = {"hs_primary": hs_primary,
                         "kept_primary": np.array(kept_primary, dtype=object)}
            if hs_holdout is not None:
                save_dict["hs_holdout"] = hs_holdout
                save_dict["kept_holdout"] = np.array(kept_holdout, dtype=object)
            np.savez_compressed(cache_path, **save_dict)
            print(f"  cached → {cache_path}")

    # Free model — probing is CPU-bound from here.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    N, L, D = hs_primary.shape
    print(f"▶ Hidden states shape: primary={hs_primary.shape}  "
          f"holdout={None if hs_holdout is None else hs_holdout.shape}")

    # ── 4. Labels ──────────────────────────────────────────────────────
    p1_tr, p2_tr, p3_tr = _build_labels(kept_primary)
    p3_ho = _build_labels(kept_holdout)[2] if kept_holdout else []

    n_p1 = sum(1 for x in p1_tr if x is not None)
    n_p2 = sum(1 for x in p2_tr if x is not None)
    n_p3 = sum(1 for x in p3_tr if x is not None)
    n_p3_ho = sum(1 for x in p3_ho if x is not None)
    print(f"  labels: P1 n={n_p1}  P2 n={n_p2}  P3 n={n_p3}  P3-holdout n={n_p3_ho}")

    if n_p3 < 20:
        print(f"⚠  Only {n_p3} model-emitted-C records — P3 will be statistically thin. "
              "Consider increasing --max_per_cell or running on a larger gen JSONL.")

    # qformat masks for per-qformat P3 CV breakdown. The primary file mixes
    # qformat ∈ {base, scene, scene_text} unevenly (Phase 0.7 G4a generates
    # all three; base ~87%, scene ~7%, scene_text ~6% on the n=3000 sample).
    # We report p3_acc_cv per qformat so layer rankings can be sanity-checked
    # for qformat-dependence before pre-registration (Phase0.8.md §2 guard).
    qformat_primary = [str(r.get("qformat", "?")) for r in kept_primary]
    qformats_seen = sorted({q for q in qformat_primary if q})
    qformat_counts = {q: qformat_primary.count(q) for q in qformats_seen}
    print(f"  primary qformat distribution: {qformat_counts}")

    def _p3_for_qformat(qf: str) -> list:
        return [
            p3_tr[i] if qformat_primary[i] == qf else None
            for i in range(len(p3_tr))
        ]

    # ── 5. Per-layer probing ───────────────────────────────────────────
    per_layer_results = []
    for layer_idx in tqdm(range(L), desc="Probe layers"):
        X = hs_primary[:, layer_idx, :]
        p1_mean, p1_std, p1_n = _probe_one_layer(
            X, p1_tr, args.cv_folds, args.logreg_C, args.max_iter, args.seed)
        p2_mean, p2_std, p2_n = _probe_one_layer(
            X, p2_tr, args.cv_folds, args.logreg_C, args.max_iter, args.seed)
        p3_mean, p3_std, p3_n = _probe_one_layer(
            X, p3_tr, args.cv_folds, args.logreg_C, args.max_iter, args.seed)

        p3_ho_acc, p3_ho_n = (None, 0)
        if hs_holdout is not None and n_p3_ho > 0:
            Xho = hs_holdout[:, layer_idx, :]
            p3_ho_acc, p3_ho_n = _probe_holdout(
                X, p3_tr, Xho, p3_ho, args.logreg_C, args.max_iter)

        # Per-qformat P3 CV (in-distribution slice of the primary set).
        p3_by_qformat: dict = {}
        for qf in qformats_seen:
            qf_labels = _p3_for_qformat(qf)
            m, s, n = _probe_one_layer(
                X, qf_labels, args.cv_folds, args.logreg_C, args.max_iter, args.seed)
            p3_by_qformat[qf] = {"p3_acc_cv": m, "p3_acc_cv_std": s, "n_p3": n}

        per_layer_results.append({
            "layer_idx": layer_idx,
            "layer_type": "embedding" if layer_idx == 0 else "lm",
            "p1_acc_cv": p1_mean, "p1_acc_cv_std": p1_std, "n_p1": p1_n,
            "p2_acc_cv": p2_mean, "p2_acc_cv_std": p2_std, "n_p2": p2_n,
            "p3_acc_cv": p3_mean, "p3_acc_cv_std": p3_std, "n_p3": p3_n,
            "p3_acc_holdout": p3_ho_acc, "n_p3_holdout": p3_ho_n,
            "p3_by_qformat": p3_by_qformat,
        })

    # ── 6. Write output ────────────────────────────────────────────────
    summary = {
        "variant": args.variant,
        "n_records_scored": int(N),
        "n_records_holdout": int(0 if hs_holdout is None else hs_holdout.shape[0]),
        "letter_token_ids": letter_ids,
        "token_position": args.token_position,
        "layer_count": int(L),
        "hidden_dim": int(D),
        "cv_folds": args.cv_folds,
        "logreg_C": args.logreg_C,
        "primary_qformat_counts": qformat_counts,
        "per_layer": per_layer_results,
    }
    out_json = os.path.join(args.output_dir, f"{args.variant}_layerwise_probe.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {out_json}")

    # ── 7. Optional plot ───────────────────────────────────────────────
    if args.save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [r["layer_idx"] for r in per_layer_results]
            p1 = [r["p1_acc_cv"] for r in per_layer_results]
            p2 = [r["p2_acc_cv"] for r in per_layer_results]
            p3 = [r["p3_acc_cv"] for r in per_layer_results]
            p3h = [r["p3_acc_holdout"] for r in per_layer_results]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(xs, p1, label="P1 condition (sanity)", marker=".")
            ax.plot(xs, p2, label="P2 disambig correctness", marker=".")
            ax.plot(xs, p3, label="P3 is_good_C (CV, all qf)", marker="o")
            if any(v is not None for v in p3h):
                ax.plot(xs, [v if v is not None else float("nan") for v in p3h],
                        label="P3 is_good_C (holdout = qformat=text)",
                        marker="x", linestyle="--")
            # Per-qformat P3 CV overlays — thin lines, no markers, to keep
            # the headline curves readable.
            for qf in qformats_seen:
                ys = [r["p3_by_qformat"].get(qf, {}).get("p3_acc_cv")
                      for r in per_layer_results]
                ys = [v if v is not None else float("nan") for v in ys]
                ax.plot(xs, ys, label=f"P3 (CV, qformat={qf})",
                        linewidth=0.8, alpha=0.7)
            ax.axhline(0.5, color="grey", linewidth=0.5)
            ax.set_xlabel("LM layer index (0 = input embeddings)")
            ax.set_ylabel("Probe accuracy")
            ax.set_title(f"Phase 0.8 A1 — per-layer probe accuracy ({args.variant})")
            ax.legend()
            ax.grid(alpha=0.3)
            png = os.path.join(args.output_dir, f"{args.variant}_layerwise_probe.png")
            fig.tight_layout()
            fig.savefig(png, dpi=140)
            print(f"✅ Wrote {png}")
        except ImportError:
            print("matplotlib not installed; skipping plot.")


if __name__ == "__main__":
    main()
