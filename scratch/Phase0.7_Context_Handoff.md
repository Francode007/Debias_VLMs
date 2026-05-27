# Phase 0.7 — Context Handoff

> **Audience:** the next Copilot Chat conversation that will execute Phase 0.7.
> **Branch:** `phase0_collapse_mitigation` (working tree has uncommitted Phase 0.6 changes — see §8).
> **One-line summary:** Phase 0.6 ended with a *suspiciously perfect* SB-Bench test accuracy of **1.0** sustained across 18 consecutive checkpoints. Phase 0.7 must prove (or disprove) that this is genuine debiasing rather than reward-model oracle distillation, then move to transfer evals.

---

## 1. State at the start of Phase 0.7

### 1.1 What works

- **Three D-blockers (D1/D2/D3) closed** (full detail in `Phase0.6.md` at repo root). The DRM reward path is functionally correct.
- **Phase 0.6 F (no-causal-penalty)** training run completed end-to-end on 2000 SB-Bench train samples:
  - Peak **1.0000** SB-Bench test accuracy first hit at **step 70**, sustained through step 240 and `ep1-end`.
  - **No collapse** anywhere across 18 checkpoints (vs. Phase 0.6 with-causal which peaked at 0.9791 step 30 then collapsed to 0.336).
  - `v_loss`, `kl_beta`, and `policy_grad_norm` all stable.
- **Pipeline plumbing verified end-to-end:** `run_preprocess --split test`, `run_inference`, `run_drm_eval`, `run_training` with all Phase 0.6 flags, `run_phase0_eval_sweep`.

### 1.2 What the 1.0 actually means

The Phase 0.6 follow-up analysis on `ep1-end.jsonl` produced a **perfectly diagonal 2916×2916 confusion matrix**:

```
Predictions:  A=959 (32.9%)   B=1008 (34.6%)   C=949 (32.5%)
Gold labels:  A=959 (32.9%)   B=1008 (34.6%)   C=949 (32.5%)
Unknown-opt position:  A=756   B=768   C=774   (≈ uniform — SB-Bench randomizes)

Confusion (pred × gold):
            gold→   A     B     C
       pred A:    959     0     0
       pred B:      0  1008     0
       pred C:      0     0   949
```

This **rules out** the mode-collapse / "always-answer-C" shortcut hypothesis. The policy is producing the gold letter on every item, balanced across letter positions and categories. It does **not** mean debiasing has been demonstrated — see §1.3.

### 1.3 The remaining open question

D3 made the SVM reward heads near-perfect on the (chosen, rejected) pair task at `post_letter` (test acc = 1.0 across all 9 categories). PPO then distilled that signal into the policy.

So the SVM heads are effectively a **gold-label oracle on the same SB-Bench distribution** as the eval set. The 1.0 is therefore the *ceiling of in-distribution reward-model distillation*, not evidence of transferable debiasing. Possible failure modes still consistent with 1.0:

1. The policy may be *memorising positional / letter-token cues* the SVM heads exploit, rather than reasoning about the image+context.
2. The policy may have *lost general VQA capability* (POPE / captioning) as a tax for the reward-fit.
3. The policy may *only* win on items whose distribution matches SB-Bench train, and *fail* on related-but-different bias benchmarks (BBQ, StereoSet, etc.).

Phase 0.7 is designed to discriminate between "real debias transfer" and "in-distribution oracle distillation".

---

## 2. Phase 0.7 scope

### 2.1 Priority-1 (must-do, cheap, falsifiable)

| # | Experiment | What it tests | Estimated cost |
|---|---|---|---|
| **P1** | **`question_polarity` breakdown on existing `ep1-end.jsonl`** | Whether accuracy holds when gold is a *named group* (negative polarity) vs. when gold is "unknown" (non-negative polarity, typically). If acc drops sharply on one split, the policy isn't actually using the context — it's pattern-matching on the question framing. | 1 min, local Python (no Modal) |
| **P2** | **Letter-order perturbation eval** | Swap the (ans0, ans1, ans2) order at inference, re-generate, re-evaluate. If acc drops, the policy memorised positional cues from the SVM head; if acc holds, it's reasoning about content. | ~30 min on Modal (one new generation pass on test set + eval) |
| **P3** | **POPE regression eval** on `checkpoint-ep1-step70` AND `checkpoint-ep1-end` | Capability tax check. POPE entry point already exists in `run_modal.py` (`run_setup` downloads it). Compare to base-model POPE accuracy. A >5pp drop = the model traded general VQA for SB-Bench reward — needs the rectified causal penalty to mitigate. | ~20 min/ckpt on Modal |

These three must all be answered before launching any new training run.

### 2.2 Priority-2 (closer of Phase 0.7 — transfer evals)

Run **only after** the P1/P2/P3 results are in and discussed.

| # | Experiment | What it tests |
|---|---|---|
| **T1** | **BBQ eval** (Bias Benchmark for QA) | Closest analog to SB-Bench, different items, ambiguous + disambiguated splits. Real test of debiasing transfer. |
| **T2** | **VLStereoSet** or **PAIRS** or **FACET** (one vision-bias benchmark of choice) | Whether the multimodal nature of debiasing transferred. Pick one based on availability/license; VLStereoSet is the safest first choice. |
| **T3** | (Optional) **StereoSet / CrowS-Pairs** sentence-level scoring | Text-only bias score — sanity check that we didn't break the text head. |

### 2.3 Out of scope for Phase 0.7

- **Reintroducing a rectified causal penalty.** Plan: do this in **Phase 0.8**, informed by P3 (capability-tax magnitude tells us how much pressure we need).
- **Checkpoint resume bug** (kl_beta controller state, optimizer state not persisted on resume).
- **Trainer's `--reward_position {pre_letter, post_letter}` flag.** Trainer still hard-codes `h_active = curr_penultimate[:, :-1, :]`; D3-rebuilt heads were trained on a `post_letter` shift. The smoke test showed reward signal works despite this, and the 1.0 result confirms it isn't blocking — leave it for a separate cleanup.
- **PCA-DRM** comparison. SVM is already saturated; PCA can wait.
- **Multi-epoch training.** One epoch already saturates; no point sweeping epochs until we know whether the 1.0 is real.

---

## 3. Acceptance gates for Phase 0.7

| Gate | Required outcome to proceed |
|---|---|
| **G1** | P1 produces per-polarity accuracy on `ep1-end.jsonl`. If both splits ≥ 0.95 → proceed to G2. If one split < 0.80 → diagnose before any further runs. |
| **G2** | P2 (letter-perturbation) reports accuracy. If perturbed acc ≥ 0.95 → policy is robust to position. If < 0.80 → policy is using positional cues; flag for Phase 0.8 mitigation. |
| **G3** | P3 (POPE) drop ≤ 5pp vs base Qwen2.5-VL-3B-Instruct POPE accuracy → proceed to T1. If > 5pp → record and proceed to T1 anyway (transfer answer is still informative), but flag capability tax as a Phase 0.8 priority. |
| **G4** | T1 (BBQ) accuracy strictly above base-model BBQ accuracy on the same eval protocol → we have **evidence of transferable debiasing**. Anything else = oracle distillation only. |

Phase 0.7 ends after G4 with a written summary mapping each gate to the empirical outcome.

---

## 4. Files / artifacts already on disk

### 4.1 Local

| Path | Contents |
|---|---|
| `Phase0.6.md` | Full D-blocker writeup, training & eval commands, deferred items. |
| `scratch/Phase0.5_Context_Handoff.md` | Phase 0.5 baseline (Exp E 0.7709), reward-shape diagnosis. |
| `scratch/Phase0.6_Kickoff_Prompt.md` | Reference for the format of this handoff. |
| `/tmp/phase06_F_eval_sweep.json` | The 27 checkpoint × 9 category accuracy matrix (peak 1.0 step 70+). |
| `/tmp/ep1-end.jsonl` | 2916 SB-Bench test generations for the final checkpoint. Used to confirm letter distribution is balanced. |

### 4.2 Modal volume (`debias-vlm-persistent-storage` at `/mnt/data/`)

| Path | Contents |
|---|---|
| `/mnt/data/output_ppo_phase06_F_no_causal/checkpoint-ep1-step{10..240}/` | Phase 0.6 F LoRA adapters at every 10-step. |
| `/mnt/data/output_ppo_phase06_F_no_causal/checkpoint-ep1-{24,50,74}pct/`, `checkpoint-ep1-end/` | Same run, percentage-tagged + final. |
| `/mnt/data/output_ppo_phase06_F_no_causal/generations/ep1-*.jsonl` | Constrained-generation outputs from each ckpt's sweep. |
| `/mnt/data/output_ppo_phase06_F_no_causal/phase06_F_eval_sweep.json` | Combined sweep JSON (same as `/tmp/...`). |
| `/mnt/data/output_ppo_phase06_F_no_causal/metrics.jsonl` | Per-step training metrics (loss, KL, v_loss, reward, grad norms, etc.). |
| `/mnt/data/generated_heads_letter_post_letter/sb_bench-SVM-component/component{0..8}.pth` | D3 SVM heads (1:1 ↔ SB_BENCH_CATEGORIES). |
| `/mnt/data/generated_heads_letter_post_letter/kept_heads_svm.json` | D2 filter (all 9 kept; all categories covered). |
| `/mnt/data/sb_bench_data/sb_bench_data.parquet` | SB-Bench dataset. |
| `/mnt/data/pope_data/pope_data.jsonl` | POPE eval data (already downloaded via `run_setup`). |

### 4.3 SB-Bench schema reminder

`sb_bench_data/data/test-*.parquet` columns:

```
file_name (image bytes), id, category, additional_metadata,
question_polarity, context, question,
ans0, ans1, ans2, label
```

Note: SB-Bench uses **`question_polarity`** (`negative` / `nonneg`) — *not* BBQ-style `context_condition` (`ambig` / `disambig`). Whether SB-Bench tags ambiguous vs disambiguated contexts at all is **unknown to the previous agent** and the new agent must inspect `additional_metadata` to find out. If SB-Bench items are *all* ambiguous, the P1 split should be `question_polarity` (the closest available proxy); if a context_condition field exists in `additional_metadata`, use it directly.

---

## 5. Code touchpoints for Phase 0.7 work

### 5.1 P1 — `question_polarity` breakdown (purely local, no code changes needed)

`/tmp/ep1-end.jsonl` only contains the model-side fields (question_id, category, label, ans0..ans2, context, question, text). It does NOT contain `question_polarity`. Two options:

- **(preferred)** Join `/tmp/ep1-end.jsonl` against the parquet on `question_id`, group by `question_polarity`, compute accuracy.
- **(alternative)** Re-extend the generator (`src/modules/inference/generate_sb_bench_answers.py`) to pass `question_polarity` through to the jsonl. Cleaner long-term, requires a fresh generation pass.

### 5.2 P2 — letter-order perturbation

Needs a new flag or a new script. Cleanest approach: add `--shuffle_answers {none, cyclic_1, cyclic_2, random_seed}` to `src/modules/inference/generate_sb_bench_answers.py`. For each row, rotate `(ans0, ans1, ans2)` by `shuffle_amount` and re-derive `label` accordingly. Then sweep the same checkpoints (or just `ep1-step70` and `ep1-end`) with the perturbed generator. Acceptance: if the policy is robust, perturbed-eval accuracy ≈ original (~1.0). If it crashes to ~0.33 (random), the policy was reading the position, not the answer text.

### 5.3 P3 — POPE eval

`run_modal.py` already has POPE plumbing via `run_setup`. Check for an existing `run_pope_eval` entrypoint; if absent, write a thin wrapper that:

1. Loads a checkpoint adapter onto the base policy model.
2. Runs `modules.inference.generate_answers --data-path /mnt/data/pope_data/pope_data.jsonl ...`.
3. Evaluates against POPE gt with the standard yes/no scoring.

The existing `run_phase0_eval_sweep` already has a `dataset == "pope"` branch for the generation step but explicitly skips the eval step (no gt_file). Phase 0.7 must close that hole.

### 5.4 T1 — BBQ eval

Larger lift. BBQ data download + loader + generator. Plan to write `src/modules/data/load_bbq.py` + `src/modules/inference/generate_bbq_answers.py` + `src/modules/evaluation/eval_bbq.py`. Mirror the SB-Bench structure for minimum surprise.

---

## 6. Training/eval commands ready to use

### 6.1 Re-running the Phase 0.6 F sweep on a perturbed eval set (after P2 implementation)

```bash
modal run src/run_modal.py::run_phase0_eval_sweep \
  --train-output-dir /mnt/data/output_ppo_phase06_F_no_causal \
  --combined-json   /mnt/data/output_ppo_phase06_F_no_causal/phase07_P2_perturbed_sweep.json \
  --gen-subdir      generations_perturbed_cyclic1 \
  --only-tags       "ep1-step70,ep1-end"
```

(After adding `--shuffle_answers cyclic_1` plumb-through to the generator.)

### 6.2 POPE eval (skeleton — entrypoint TBD)

```bash
modal run src/run_modal.py::run_pope_eval \
  --checkpoint-dir /mnt/data/output_ppo_phase06_F_no_causal/checkpoint-ep1-step70 \
  --output-json    /mnt/data/output_ppo_phase06_F_no_causal/phase07_P3_pope_step70.json
```

### 6.3 Base model POPE baseline (must be measured once, then cached)

Same command with no `--checkpoint-dir` (or pointing at the base policy dir). Record the number — it's needed by every G3 evaluation forever after.

---

## 7. Open risks / things that could surprise the next agent

1. **`question_polarity` may not partition cleanly into "named gold" vs "unknown gold".** SB-Bench's polarity is a *question framing* axis, not an *answer-type* axis. The new agent should look at sample items in each polarity bucket and verify whether the 1.0 accuracy hypothesis decomposition still makes sense before reporting G1. If it doesn't, the right cut is whatever field marks `gold = "unknown" option`. Build that field by joining label index against `is_unknown(ansN)`.
2. **The trainer still projects against `curr_penultimate[:, :-1, :]`** (off by one from how heads were trained). The 1.0 happened anyway, so this is benign in practice, but if P2 (letter perturbation) reveals positional sensitivity, this is the first place to look.
3. **18 perfect checkpoints means PPO converged before step 70 and then stopped learning anything new.** That's *not* a bug — the reward signal becomes degenerate (every emitted letter is "correct") once the policy matches the SVM. Don't try to "improve" past step 70; analyse what was learned at step 70.
4. **`reward_models_sb_bench/` (Qwen2-VL-2B reward model) is still unused.** Phase 0.7 doesn't need it; flag for later.
5. **`scripts/phase05_ablation.sh` references Phase 0.5 paths.** Don't reuse blindly.

---

## 8. Git state at the start of Phase 0.7

```
branch: phase0_collapse_mitigation
HEAD:   e230aa4  Enhance model training and evaluation pipeline with new features and fixes
uncommitted (Phase 0.6 work — to be committed before starting 0.7):
  M .gitignore
  M src/modules/evaluation/evaluate_drm_heads.py
  M src/modules/rl_components/custom_vlm_ppo_trainer.py
  M src/modules/training/args.py
  M src/modules/training/drm_loader.py
  M src/run_modal.py
  ?? Phase0.6.md
```

**Recommended first action for the new agent:** commit the Phase 0.6 work as a single `feat(phase0.6): close D1/D2/D3 + drop causal penalty from loss` commit, push to `origin/phase0_collapse_mitigation`, then branch `phase0.7_transfer_eval` off of it. Phase 0.7 work should NOT mix with the Phase 0.6 commits.

---

## 9. Critical caveat that must survive the handoff

**SB-Bench accuracy = 1.0 is not the win condition for the project.** It is the *ceiling of in-distribution reward-model distillation* given that the SVM heads were trained on the same dataset. The real win condition is transfer evidence on a benchmark the SVM heads have never seen (T1/T2). Phase 0.7 must end with one of:

- ✅ "Transfer holds — debiasing learned." → commit, push, plan Phase 0.8 (rectified causal penalty).
- ⚠ "In-distribution-only — no transfer." → diagnose whether the reward model is too narrow, or the policy overfit it, or both. Phase 0.8 = different reward-head training corpus (cross-dataset).
- 🟥 "Capability tax broke POPE." → Phase 0.8 = reintroduce rectified causal penalty + retrain.

Whichever it is, the next phase plan emerges from the P1/P2/P3/T1 evidence — not from speculation.
