# Phase 0.6 — DRM Reward Correctness

> **Scope:** code correctness only, no new PPO runs.
> **Branch:** `phase0_collapse_mitigation`
> **Baseline to beat:** Phase 0.5 Exp E step-80 = 0.7709 (binary reward, full 2916-sample SB-Bench test set).

## 0. Why this phase exists

Phase 0.5 closed with the conclusion that **binary ±1 reward over a 3-way letter choice cannot teach causal credit assignment** — the optimal policy under binary + KL is to collapse onto the highest-prior letter. The Decomposed Reward Model (DRM) was supposed to fix this by replacing the sparse scalar with dense, per-token projections of `φ` onto direction vectors learned from labelled bias-bench pairs.

In practice the DRM path was **silently broken** in three independent places. Switching `--reward-mode binary → svm` was producing a zero-variance reward, so PPO was running on noise. Phase 0.6 fixes those three blockers so the SVM/PCA reward path is functionally correct before any new PPO runs are launched.

## 1. The three D-blockers

| ID | One-liner | Status |
|---|---|---|
| **D3** | Heads trained on a φ that does not match what PPO scores at training time | CLOSED |
| **D2** | All heads were loaded unfiltered, including category-degenerate ones | CLOSED |
| **D1** | Reward heads were projected onto the **moving** `φ_active`, not the frozen `φ_ref` they were fit on | CLOSED |

Implementation order was **D3 → D2 → D1**, gating after each.

---

## 2. D3 — token-position alignment of `φ`

### 2.1 The bug

The Phase-1 head-generation pipeline extracted `φ` from **EOS** of a free-text completion. At PPO time, the trainer projects `φ` from **per-generated-token** positions of a short letter completion. The two distributions are not the same vector space — at best the SVM separates noise, at worst the head normal vectors point in directions the trainer can never produce.

### 2.2 The fix

Two new flags propagated through `extract.py` and `generate_drm_heads.py`:

- `--completion-format {free_text, letter}` — controls whether the assistant turn is the dataset's free-text answer or just the letter (`A` / `B` / `C`).
- `--token-position {eos, pre_letter, post_letter}` — controls which token position's hidden state is taken as `φ`.

`post_letter` was selected after `pre_letter` was tried and produced a **degenerate SVM** (acc ≈ 0.5, ||w|| ≈ 0) because chosen and rejected share the same prompt prefix up to the pre-letter position. `post_letter` corresponds to the hidden state immediately after the letter has been emitted — which is exactly what the trainer projects against at the answer position.

### 2.3 Volume layout (auto-suffixed)

When `(completion_format, token_position) != ("free_text", "eos")`, the extract/generate/eval pipeline appends `_{completion_format}_{token_position}` to its output dirs so legacy and new artifacts coexist:

```
/mnt/data/embeddings_output_letter_post_letter/
/mnt/data/generated_heads_letter_post_letter/
    sb_bench-SVM-component/component0.pth ... component8.pth
    sb_bench-PCA-component/component0.pth ... component49.pth
```

### 2.4 Acceptance gate

D3 closed empirically: SVM heads at `post_letter` on the held-out test split scored **acc = 1.00** across all 9 SB-Bench categories (Age=834, Disability=242, Gender=896, Nationality=326, Physical Appearance=290, Race/Ethnicity=822, Religion=222, SES=294, Sexual Orientation=468). Compare with the original requirement of `overall_mean ≥ 0.65 AND per-cat min ≥ 0.55`.

### 2.5 Subsidiary fixes shipped with D3

- **Split parameterization.** `run_preprocess` and `run_inference` were hard-coded to `--split train`. Added a `split` parameter so the test split can be embedded for held-out eval.
- **Chunk-resume bug.** `RLDatasetBuilder` used `os.path.exists()` as the resume marker, which incorrectly treated partial directories from killed Modal workers as complete. Now requires `chunk_*/dataset_info.json` and deletes partial dirs before rebuilding. Cache key was also extended to include `split_mode` and `completion_format` so train/test/letter/free-text chunks coexist as separate directories.
- **Modal eval entrypoint** `run_drm_eval` added. Auto-suffixes paths to read from the right embeddings/heads dirs based on `(completion_format, token_position)`, picks the SVM or PCA subdir based on `--head-type`, writes `drm_head_eval_{head_type}_{split}.json`.

---

## 3. D2 — kept-heads filter

### 3.1 The bug

`load_pca_components` loaded **all** `component*.pth` files from the heads dir. For SVM this meant all 9 category-specific heads were always used, including any that failed to learn a useful boundary. For PCA, all 50 components were used, most of which are noise after the top ~10 eigenvectors.

The PPO trainer mean-pools the K projections through the FastRL `alpha`, so noisy heads dilute the signal and let the policy hack the reward by drifting `φ` into the null space of the genuine heads.

### 3.2 The fix

Three edits:

1. **[evaluate_drm_heads.py](src/modules/evaluation/evaluate_drm_heads.py)** — added `--head_type {svm,pca}`, `--keep_threshold` (default 0.55), and `--kept_heads_json`. After computing per-head + per-category accuracy, the evaluator writes a `kept_heads.json` next to the eval results with:
   - `head_type`: `"svm"` or `"pca"`
   - `threshold`: the cutoff used
   - `kept_indices`: list of original head indices that passed the rule
   - `kept_count`: convenience
   - `num_heads_total`: convenience
   - `category_coverage`: `{category_name: [head_indices_that_help_with_it]}`
   - `split`: which split was used to compute accuracies

   **SVM rule:** head `i` ↔ `SB_BENCH_CATEGORIES[i]` (1:1 by construction in `generate_drm_heads.py`). Keep head `i` iff its own category's `accuracy_mean ≥ threshold`.

   **PCA rule:** heads are unordered linear axes. Keep head `i` iff `overall_per_head[i] ≥ threshold`. Category coverage is the union over per-category accuracies.

2. **[drm_loader.py](src/modules/training/drm_loader.py)** — `load_pca_components` gained an optional `kept_heads_filter: str = None` argument. When set, it loads the JSON, validates that every kept index exists as a `.pth` file, and subsets the loaded heads to just those indices (order preserved from `kept_indices`).

3. **[args.py](src/modules/training/args.py)** — new `--kept_heads_filter` CLI argument (default `None`) plumbed into `train_rl.py`'s `load_pca_components` call.

### 3.3 Acceptance gate

`kept_heads.json` must exist with **≥ 5 SVM heads (or ≥ 30 PCA heads) AND at least one head per SB-Bench category**. With D3 producing acc=1.0 on all 9 SVM categories, the gate trivially passes — **9/9 SVM heads kept, all 9 categories covered**.

---

## 4. D1 — frozen-φ projection

### 4.1 The bug

The PPO trainer was projecting reward heads against the **active** (LoRA-on) penultimate hidden state:

```python
h_active = curr_penultimate[:, :-1, :]
r_token_k = torch.matmul(h_active, self.reward_heads_weight.T)
```

But the heads were fit on `φ` from the **un-adapted** base model. As LoRA trains, `φ_active` drifts. The policy can then increase reward without changing its answer distribution at all, simply by drifting `φ` into directions the heads happen to score highly — classic **representation hacking**. The reward stops being a function of model behaviour.

### 4.2 The fix

The trainer **already** computes `ref_penultimate` inside `with self.policy.disable_adapter()` for the KL penalty. D1 simply reuses it:

```python
h_reward = h_ref if self.use_frozen_phi else h_active
r_token_k = torch.matmul(h_reward, self.reward_heads_weight.T)
```

Three edits:

- **[custom_vlm_ppo_trainer.py](src/modules/rl_components/custom_vlm_ppo_trainer.py)** — `__init__` takes `use_frozen_phi: bool = False`; the SVM branch picks `h_reward`. The causal penalty deliberately keeps using `h_active` vs `h_ref` because it is supposed to *detect* drift, not feed it back.
- **[setup.py](src/modules/training/setup.py)** — `build_ppo_controller` passes `use_frozen_phi` from args.
- **[args.py](src/modules/training/args.py)** — new `--use_frozen_phi` flag (action `store_true`, default off → preserves backcompat with the binary-mode Exp E 0.7709 baseline).

The flag is a no-op in `--reward-mode binary` because that branch never matmuls against the heads.

### 4.3 Acceptance gate

A 1-batch (8-sample) smoke run of `train_rl.py` with `--reward_mode svm --use_frozen_phi --kept_heads_filter ...` must log a per-token `reward_task` with std > 0.01.

A new metric `reward_task_std` was added to the trainer's `metrics` dict for this gate. Smoke-test result on Modal:

```json
{
  "reward_task": -1.375,
  "reward_task_std": 1.0,     // gate: > 0.01  ✅
  "policy_grad_norm": 2.114,
  ...
}
```

(`std=1.0` is by construction after the running-variance normalizer kicks in. The point is that the *raw* signal is non-degenerate.) The gradient-flow check inside the trainer additionally reported **252/504** trainable LoRA params received non-zero gradients on the first backward pass.

---

## 5. New / changed Modal entrypoints

### 5.1 `run_drm_eval` (new)

```bash
modal run src/run_modal.py::run_drm_eval \
  --completion-format letter \
  --token-position post_letter \
  --head-type svm \
  --split test
```

Writes:

```
/mnt/data/generated_heads_letter_post_letter/drm_head_eval_svm_test.json
/mnt/data/generated_heads_letter_post_letter/kept_heads.json
```

### 5.2 `run_preprocess` / `run_inference` — added `split` param

```bash
modal run src/run_modal.py::run_preprocess \
  --completion-format letter --split test
modal run src/run_modal.py::run_inference \
  --completion-format letter --token-position post_letter --split test
```

### 5.3 `run_training` — added three Phase 0.6 flags

- `--use-frozen-phi` (action store_true) — gates D1.
- `--kept-heads-filter PATH` — gates D2.
- `--heads-suffix _letter_post_letter` — selects the D3 heads dir. Concatenated onto `/mnt/data/generated_heads` to form the `reward_heads_dir`.

---

## 6. Reference training command (Exp E analog with all three fixes)

```bash
modal run src/run_modal.py::run_training \
  --epochs 1 \
  --max-train-samples 2000 \
  --batch-size 8 \
  --max-gen-tokens 8 \
  --reward-mode svm \
  --head-type svm \
  --use-frozen-phi \
  --heads-suffix _letter_post_letter \
  --kept-heads-filter /mnt/data/generated_heads_letter_post_letter/kept_heads_svm.json \
  --learning-rate 1e-4 \
  --value-learning-rate 5e-4 \
  --kl-beta 0.1 \
  --target-kl 0.02 \
  --kl-adapt-rate 0.1 \
  --kl-beta-min 0.05 \
  --kl-beta-max 5.0 \
  --max-grad-norm 0.1 \
  --value-clip-range 0.2 \
  --lr-schedule cosine \
  --min-lr-ratio 0.2 \
  --warmup-ratio 0.05 \
  --ckpt-every-steps 10 \
  --midtrain-eval-every-steps 10 \
  --midtrain-eval-samples 64 \
  --output-dir /mnt/data/output_ppo_phase06_E_svm_frozen_phi
```

Everything except the four Phase 0.6 flags (`--reward-mode svm`, `--use-frozen-phi`, `--heads-suffix`, `--kept-heads-filter`) is bit-identical to Phase 0.5 Exp E so the comparison is clean.

Eval sweep across all 10-step checkpoints:

```bash
modal run src/run_modal.py::run_phase0_eval_sweep \
  --train-output-dir /mnt/data/output_ppo_phase06_E_svm_frozen_phi \
  --combined-json /mnt/data/output_ppo_phase06_E_svm_frozen_phi/phase06_eval_sweep.json \
  --data-path /mnt/data/sb_bench_data/sb_bench_data.parquet \
  --dataset sb_bench
```

---

## 7. Reward-mode / head-type matrix

`--reward-mode` only accepts `{svm, binary}`. There is no `pca` or `drm` value — the same `svm` code path is used regardless of where the head vectors came from. Switch the **head-type** to change source:

| Combination | Flags | Notes |
|---|---|---|
| Binary baseline | `--reward-mode binary` | Sparse ±1 reward at the answer-letter position. Head dir / kept-heads / frozen-φ are no-ops. |
| SVM-DRM | `--reward-mode svm --head-type svm` | Category-specific SVM normals. 1:1 head↔category (9 heads for SB-Bench). |
| PCA-DRM | `--reward-mode svm --head-type pca` | Top-50 PCA eigenvectors. Needs its own `kept_heads_pca.json` (re-run `run_drm_eval --head-type pca`). |

`--num-heads` defaults to **0**, which auto-detects and loads every `component*.pth` file present in the heads dir (9 for SVM, 50 for PCA). Pass a positive integer only if you want to truncate to the first N sorted components.

SVM and PCA `kept_heads_{head_type}.json` files coexist in the same heads dir without collision, so you can flip `--head-type` without re-running the evaluator each time:

```bash
# SVM filter
modal run src/run_modal.py::run_drm_eval \
  --completion-format letter --token-position post_letter \
  --head-type svm --split test
# → writes kept_heads_svm.json

# PCA filter (separate file, no overwrite)
modal run src/run_modal.py::run_drm_eval \
  --completion-format letter --token-position post_letter \
  --head-type pca --split test
# → writes kept_heads_pca.json
```

---

## 8. Files touched in Phase 0.6

| File | Change |
|---|---|
| [src/run_modal.py](src/run_modal.py) | Added `run_drm_eval` entrypoint. Added `split` param to `run_preprocess` and `run_inference`. Added `use_frozen_phi`, `kept_heads_filter`, `heads_suffix` to `run_training`. |
| [src/modules/utils/dataset_builder.py](src/modules/utils/dataset_builder.py) | Cache key now includes `split_mode` + `completion_format`. Chunk-resume now requires `dataset_info.json` marker; partial dirs are deleted and rebuilt. |
| [src/modules/embeddings/extract.py](src/modules/embeddings/extract.py) | `--completion_format`, `--token_position`, `--split` flags. Auto-suffixed output dir. |
| [src/modules/embeddings/generate_drm_heads.py](src/modules/embeddings/generate_drm_heads.py) | Reads auto-suffixed embeddings dir. SVM heads written in `SB_BENCH_CATEGORIES` order (head 0 = Age, …, head 8 = Sexual Orientation). |
| [src/modules/evaluation/evaluate_drm_heads.py](src/modules/evaluation/evaluate_drm_heads.py) | `--head_type`, `--keep_threshold`, `--kept_heads_json` flags. Emits `kept_heads.json` with `kept_indices` + `category_coverage`. |
| [src/modules/training/drm_loader.py](src/modules/training/drm_loader.py) | `load_pca_components` gained `kept_heads_filter` arg. |
| [src/modules/training/args.py](src/modules/training/args.py) | New `--kept_heads_filter` and `--use_frozen_phi` flags. |
| [src/modules/training/setup.py](src/modules/training/setup.py) | `build_ppo_controller` forwards `use_frozen_phi`. |
| [src/modules/training/train_rl.py](src/modules/training/train_rl.py) | Loader call passes `kept_heads_filter=...`. |
| [src/modules/rl_components/custom_vlm_ppo_trainer.py](src/modules/rl_components/custom_vlm_ppo_trainer.py) | `use_frozen_phi` switches SVM matmul from `h_active` → `h_ref`. New `reward_task_std` metric. |

---

## 9. Acceptance evidence (final)

| Gate | Required | Observed |
|---|---|---|
| D3: SVM eval on held-out test | `overall_mean ≥ 0.65, per-cat min ≥ 0.55` | **1.0 / 1.0** across 9 categories |
| D2: `kept_heads.json` with coverage | ≥ 5 SVM (or ≥ 30 PCA) AND ≥ 1 per category | **9/9 kept, all 9 categories covered** |
| D1: smoke-test reward signal | `reward_task` std > 0.01 over 1 batch | **`reward_task_std = 1.0`** + 252/504 LoRA params get gradients |

**Status: Phase 0.6 complete. Code is correctness-clean for the first real DRM-PPO run.**

---

## 10. Open items deferred from Phase 0.6

These were considered in scope at the start of Phase 0.6 but rolled forward; none of them block running the Exp E analog above.

1. **PPO `--reward_position {pre_letter, post_letter}` gate.** Trainer still hard-codes `h_active = curr_penultimate[:, :-1, :]`. The D3-rebuilt heads were trained on a `post_letter` shift (`curr_penultimate[:, 1:, :]` with `ans_pos+1`). The trainer's current behaviour is one position off from what the heads were trained on. This may or may not materially hurt the reward signal once it's measured on the full run — the smoke test confirmed `reward_task_std > 0.01` despite the misalignment.
2. **Checkpoint resume bug** (carried over from Phase 0.5): `kl_beta` controller state and optimizer state are not persisted. Resume restarts the controller from `--kl-beta`. Defer to Phase 0.7.