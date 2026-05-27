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

## 2. P2 — letter-order perturbation (cyclic-1)  ✅ **G2 PASS**

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

```bash
modal run src/run_modal.py::run_phase0_eval_sweep \
  --train-output-dir /mnt/data/output_ppo_phase06_F_no_causal \
  --combined-json   /mnt/data/output_ppo_phase06_F_no_causal/phase07_P2_perturbed_cyclic1.json \
  --gen-subdir      generations_perturbed_cyclic1 \
  --only-tags       "ep1-step70,ep1-end" \
  --shuffle-answers cyclic_1
```

### 2.4 Results

| ckpt | perturbed acc | acc if scored vs ORIGINAL label |
|---|---|---|
| `ep1-step70` | **1.0000** (n=2916) | 0.0000 |
| `ep1-end`    | **1.0000** (n=2916) | 0.0000 |

Per-category: all 9 categories at 1.0000 on both checkpoints.

**Letter distribution sanity** — predicted letters track the rotated gold positions exactly:

| | A | B | C |
|---|---|---|---|
| orig gold (pre-rotation) | 959 | 1008 | 949 |
| new gold (post-rotation) | 1008 | 949 | 959 |
| **predicted (both ckpts)** | **1008** | **949** | **959** |

The exact 0.0000 against original positions confirms the policy followed the gold text into its new position, not the previously-correct letter slot.

### 2.5 Hypotheses ruled out
- "Policy memorised letter-position cues from the SVM heads" — rejected. Acc would have collapsed to ~0.33 (chance under permutation) if true.
- "Policy ignores the answer strings and reads only A/B/C distribution learned from training" — rejected.
- Combined with P1: the policy is reading the **answer text** and routing it to the correct letter even under permutation. This is the strongest in-distribution evidence achievable without a held-out benchmark.

### 2.6 What remains open
- Capability tax: did fitting the SVM reward break general VQA? → P3 (POPE).
- True transfer: does this hold off-distribution (BBQ)? → T1.

---

## 3. P3 — POPE regression  ✅ **G3 PASS**

### 3.1 Design
Run POPE (yes/no object-hallucination probe, 9000 items) on:
1. **Base** Qwen2.5-VL-3B-Instruct (one-time baseline).
2. `ep1-step70` (peak SB-Bench checkpoint).
3. `ep1-end` (final checkpoint).

Compare accuracy / F1 to base. **G3 passes** if drop ≤ 5 pp on both.

### 3.2 Implementation
New Modal entrypoint [run_pope_eval in src/run_modal.py](../src/run_modal.py): loads (optional) LoRA checkpoint onto base Qwen2.5-VL-3B-Instruct, generates against `/mnt/data/pope_data/pope_data.parquet`, scores against `/mnt/data/pope_data/pope_data.jsonl` via `modules.evaluation.eval_pope`, writes `{tag}_pope_results.json` to `/mnt/data/phase07_pope/`.

### 3.3 Modal commands

```bash
# (a) base model baseline — pass empty --checkpoint-dir for vanilla
modal run src/run_modal.py::run_pope_eval --checkpoint-dir "" --tag base

# (b) peak SB-Bench checkpoint
modal run src/run_modal.py::run_pope_eval \
  --checkpoint-dir /mnt/data/output_ppo_phase06_F_no_causal/checkpoint-ep1-step70 \
  --tag phase06F_step70

# (c) final checkpoint
modal run src/run_modal.py::run_pope_eval \
  --checkpoint-dir /mnt/data/output_ppo_phase06_F_no_causal/checkpoint-ep1-end \
  --tag phase06F_end
```

Pull results locally with:

```bash
modal volume get debias-vlm-persistent-storage phase07_pope/base_pope_results.json            /tmp/pope_base.json --force
modal volume get debias-vlm-persistent-storage phase07_pope/phase06F_step70_pope_results.json /tmp/pope_step70.json --force
modal volume get debias-vlm-persistent-storage phase07_pope/phase06F_end_pope_results.json    /tmp/pope_end.json --force
```

### 3.4 Results

| ckpt | acc | Δacc | F1 | ΔF1 | precision | recall | yes_prop | unknown |
|---|---|---|---|---|---|---|---|---|
| **base** Qwen2.5-VL-3B-Instruct | 0.8714 | — | 0.8746 | — | 0.8538 | 0.8964 | 0.5200 | 0.0068 |
| `ep1-step70` | 0.8636 | **−0.79 pp** | 0.8693 | −0.53 pp | 0.8340 | 0.9078 | 0.5433 | 0.0016 |
| `ep1-end`    | 0.8668 | **−0.47 pp** | 0.8718 | −0.28 pp | 0.8400 | 0.9062 | 0.5388 | 0.0012 |

All n=9000. Both checkpoints far inside the 5 pp tolerance.

### 3.5 Interpretation
- **Capability tax is negligible** (< 1 pp acc / F1 on both checkpoints). The Phase 0.6 F policy did *not* trade general VQA for SB-Bench reward fit — POPE is a different task family (yes/no object grounding, no multiple-choice letters) and the LoRA delta is small enough to be transparent to it.
- The slight shifts (recall +1 pp, precision −1 to −2 pp, `yes_proportion` 0.52 → 0.54) indicate the trained policy is marginally more likely to say "yes". This is a known side-effect of PPO with KL-anchoring around a chatty base — small enough to be cosmetic. Unknown-rate **dropped** (0.68% → 0.12 / 0.16%), which is a strict improvement (model is more decisive without sacrificing accuracy).
- `ep1-end` (Δacc −0.47 pp) is slightly better than `ep1-step70` (Δacc −0.79 pp), consistent with the SB-Bench eval-sweep observation that the policy continues quiescent fine-tuning past the saturation point without drift.

### 3.6 What G3 PASS implies for Phase 0.8
- The rectified causal penalty (deferred to Phase 0.8) is **not needed for capability preservation** at the current LoRA rank / KL budget. It remains worth implementing for the *reward-hacking* concern (D1 only blocks one form of representation drift), but the capability-tax motivation is now weak.
- If T1 BBQ comes back negative (oracle-only), revisit the reward-head training corpus rather than the causal penalty.

---

## 4. T1 — BBQ transfer  ⏳ pending

*(only after G1 + G2 + G3 are all reported)*

---

## 5. Phase 0.7 running summary

| Gate | Test | Status | Observed |
|---|---|---|---|
| G1 | P1: polarity & gold-is-unknown splits | ✅ PASS | 1.0000 on all 4 cells incl. 618 named-gold items |
| G2 | P2: cyclic-1 letter perturbation | ✅ PASS | 1.0000 perturbed, 0.0000 vs original letter — content-routing confirmed |
| G3 | P3: POPE regression | ✅ PASS | Δacc = −0.79 / −0.47 pp (step70 / end), ΔF1 ≤ 0.53 pp — capability tax negligible |
| G4 | T1: BBQ transfer | ⏳ pending | next experiment |

**Combined P1+P2 verdict:** the Phase 0.6 F policy is genuinely reading the (image, context, question, answer-text) input and producing the gold answer — not pattern-matching on letter position or question polarity. This rules out all *in-distribution* shortcut hypotheses but does not yet demonstrate transferable debiasing (the SVM heads were fit on the same distribution as eval).

