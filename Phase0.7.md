# Phase 0.7 — Verifying the 1.0, Then Testing Transfer

> **Scope:** Discriminate between *real debias transfer* and *in-distribution oracle distillation*.
> **Branch:** `phase0.7_transfer_eval`
> **Baseline to interpret:** Phase 0.6 F no-causal — SB-Bench test accuracy = 1.0000 at step 70, sustained through `ep1-end` across 18 consecutive checkpoints. Confusion matrix is perfectly diagonal (mode-collapse rejected).
> **Critical caveat:** SB-Bench acc=1.0 is the ceiling of *in-distribution reward-model distillation* — the SVM heads scored 1.0 on the same (chosen, rejected) test split during D3 acceptance. The real win condition is **transfer to a benchmark the heads never saw** (T1 BBQ).

## 0. Acceptance gates

| Gate | Test | Pass criterion |
|---|---|---|
| **G1** | P1 — `question_polarity` split on `ep1-end.jsonl` | Both polarities ≥ 0.95 |
| **G2** | P2 — letter-order perturbation (cyclic-1 shift) | Perturbed acc ≥ 0.95 |
| **G3** | P3 — POPE regression | Drop ≤ 5 pp vs base Qwen2.5-VL-3B-Instruct |
| **G4** | T1 — BBQ transfer | BBQ acc strictly > base-model BBQ acc |

---

## 1. P1 — `question_polarity` and `gold_is_unknown` split  ✅ **G1 PASS**

### 1.1 Setup
- Source: `/tmp/ep1-end.jsonl` (2916 generations from `checkpoint-ep1-end` of `output_ppo_phase06_F_no_causal`).
- Join: filtered parquet `iloc[test_indices]` using `/mnt/data/split_indices.json`, keyed by `question_id = local index after .select(test_indices)`.
- Script: [scripts/phase07_p1_polarity.py](../scripts/phase07_p1_polarity.py).

### 1.2 Results — every cell is 1.0000

**By question_polarity**

| polarity | acc | n |
|---|---|---|
| negative | 1.0000 | 1489 |
| nonneg   | 1.0000 | 1427 |

**By gold_is_unknown** (gold answer text matches "can't be determined" / "unknown" / similar)

| gold_is_unknown | acc | n |
|---|---|---|
| False (gold is a named group) | **1.0000** | 618 |
| True  (gold is the safe option) | 1.0000 | 2298 |

**Cross (polarity × gold_is_unknown)**

| polarity | gold_is_unknown | acc | n |
|---|---|---|---|
| negative | False | 1.0000 | 329 |
| negative | True  | 1.0000 | 1160 |
| nonneg   | False | 1.0000 | 289 |
| nonneg   | True  | 1.0000 | 1138 |

All 18 (category × polarity) cells also 1.0000. Smallest cell: Religion-negative, n=63.

### 1.3 Hypotheses ruled out
- "Always answer the unknown option" — rejected (618 named-gold items all correct).
- "Only wins on the easier polarity framing" — rejected (1.0 on both 1489-negative and 1427-nonneg).
- "Only wins where gold = safe option" — rejected (named-gold subset is 1.0 too).

### 1.4 What remains open
- Policy may be memorising answer-text surface form / positional cues. → P2.
- Reward-head fingerprints may be perfectly correlated with gold on this distribution only. → T1 (BBQ).

---

## 2. P2 — letter-order perturbation (cyclic-1)  ⏳ in progress

### 2.1 Design
Rotate `(ans0, ans1, ans2)` by one position before generation. The originally-correct answer text moves from old position `L` to new position `(L − 1) mod 3`; the eval gold label is updated to match. If the policy is reasoning about answer **content**, accuracy holds; if it is reading the **position** the heads taught it, accuracy collapses to ~0.33.

**Mapping `cyclic_1`:** new[i] = old[(i+1) mod 3]; new_label = (old_label − 1) mod 3.

| old_letter (gold) | new_letter (gold) |
|---|---|
| A (0) | C (2) |
| B (1) | A (0) |
| C (2) | B (1) |

### 2.2 Implementation
- [src/modules/inference/generate_sb_bench_answers.py](../src/modules/inference/generate_sb_bench_answers.py): new `--shuffle_answers {none, cyclic_1, cyclic_2}` flag. Updates `ans0/1/2` strings AND `label` in the emitted JSONL so `eval_sb_bench.py` scores against the new position. Original label preserved as `orig_label` for diagnostics.
- [src/run_modal.py](../src/run_modal.py): `run_phase0_eval_sweep` gained `shuffle_answers` kwarg, plumbed into the generator subprocess.

### 2.3 Modal run
Only the two informative checkpoints (peak + final), written to a separate `gen_subdir` so the original `generations/` tree is untouched:

```bash
modal run src/run_modal.py::run_phase0_eval_sweep \
  --train-output-dir /mnt/data/output_ppo_phase06_F_no_causal \
  --combined-json   /mnt/data/output_ppo_phase06_F_no_causal/phase07_P2_perturbed_cyclic1.json \
  --gen-subdir      generations_perturbed_cyclic1 \
  --only-tags       "ep1-step70,ep1-end" \
  --shuffle-answers cyclic_1
```

### 2.4 Results
*(to be filled in once Modal run completes)*

---

## 3. P3 — POPE regression  ⏳ pending

*(design TBD after P2 results)*

---

## 4. T1 — BBQ transfer  ⏳ pending

*(only after G1 + G2 + G3 are all reported)*
