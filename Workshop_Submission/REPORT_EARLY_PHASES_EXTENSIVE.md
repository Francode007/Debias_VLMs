# REPORT_EARLY_PHASES_EXTENSIVE
## Detailed Trajectory of Phases 0 → 0.7 (Pre-Probe-as-Reward)

**Project:** Interpretable Probe-Reward Debiasing of Vision-Language Models
**Coverage:** From the initial binary-PPO baseline through the SVM/PCA-DRM head-to-head
that motivated the Phase 0.8 pivot to probe-as-reward.

> This document is the **supplementary detailed history**. The compressed view of
> these phases is in §2 of [REPORT_SUMMARY.md](REPORT_SUMMARY.md); the Phase 0.8
> deep-dive (current work) is in [REPORT_EXTENSIVE.md](REPORT_EXTENSIVE.md).
>
> All commit hashes, CLI flags, Modal apps, and file paths are reproduction
> footnotes; the narrative stays in the prose.

---

## 0. Setting

**Model.** Qwen2.5-VL-3B-Instruct, bf16, FlashAttention-2, LoRA r=16/α=32 on text
projection layers (PEFT). 37 LM-decoder layers, 2048-d hidden state.

**Training data.** SB-Bench (BBQ-style 3-way multiple choice, 2916 test items,
9 demographic categories: Age, Disability, Gender, Nationality, Physical_appearance,
Race/Ethnicity, Religion, SES, Sexual_orientation). Ground-truth letter distribution:
A=32.9%, B=34.6%, C=32.5%. Images grounded to demographic stereotypes.

**Transfer benchmark.** VLBiasBench close-ended (10 BBQ axes × 3 conditions × multiple
question formats; 53,034 records after `qformat ∈ {base, scene, scene_text}` filter).

**Infrastructure.** Modal A100-80GB (12 h timeout per training run); single-epoch
on-policy PPO with FirstTokenAllowlistProcessor restricting first generated token to
{A, B, C}; GAE γ=1.0, λ=0.95, ppo_clip=0.2, vf_coef=0.1.

**Research question (entering Phase 0).** Can a single-token correctness reward over
A/B/C train a fairer policy via PPO under LoRA?

---

## 1. Phase 0 — Binary Reward + Mode Collapse

### 1.1 Setup and the first transient

The Phase 0 PPO trainer used a binary reward: `r = +1 if predicted letter == gold letter
else −1`, with a target-KL adaptive controller `β *= clamp(1 + α·(kl/target − 1),
[0.5, 2.0])`. First run produced a striking trajectory:

| Step | Eval acc | Per-token KL | β |
|---:|---:|---:|---:|
| 30 | 0.960 | 0.07 | 0.06 |
| 40 | 0.990 | **0.76** | 0.18 |
| **50** | **0.9949** (peak) | 0.68 | 0.63 |
| 60 | 0.975 | 0.82 | **5.0 (pinned)** |
| ep1-end | **0.5377** | ~0.03 | 5.0 |

The model peaked at near-perfect accuracy at step 50, then collapsed **back to 53.8%
by epoch end**. The collapse mechanism was mode-collapse onto letter "B" ("Cannot be
determined"): predicted ~63% of the time vs ground-truth 34.6%.

### 1.2 Diagnosis: reward shape, not PPO

Binary ±1 reward over a 3-way categorical action is mathematically the worst-case for
policy gradient[^bandit]: the gradient signal is `(±1 − V) × ∇log π(a|s)`, and after KL
re-anchoring the policy's optimum becomes the **highest-prior letter** (which gives
≈ 35% accuracy for free). The reward is **input-invariant** — it sees only the output
token, not the (image, text) pair — so it can teach the **marginal** letter
distribution, never the **conditional**. **It cannot teach causality.**

[^bandit]: Standard result: under softmax policies with KL anchoring and a sparse
±1 reward, the unique stationary point is `π(a) ∝ exp(β·r(a)) · π_ref(a)` — the
posterior-weighted prior. For a 3-class problem with one class having highest prior
and ±1 reward, that converges to "always pick the highest-prior class."

### 1.3 What Phase 0 shipped (code)

| Change | Why |
|---|---|
| Adaptive KL controller with `β` clamped to `[β_min, β_max]` | Stop unbounded KL drift after the peak |
| Cosine LR schedule floored at `min_lr_ratio·peak` via `LambdaLR` | Prevent total-decay-to-zero blowing up the optimizer |
| Value-loss clipping with `value_clip_range` | Standard PPO hygiene (inert in single-epoch PPO; see §3.1 caveat) |
| `--ckpt_every_steps` dense checkpoints | Capture peak before collapse for post-hoc analysis |

Commit hash: `3d5b01e` on branch `phase0_collapse_mitigation`.

### 1.4 What Phase 0 left open

- Could PPO knobs (KL aggressiveness, grad clip, LR, batch size) rescue the binary
  reward, or is the bottleneck the reward *shape*?
- Is the 0.99 peak a real ceiling or just an unstable transient on the way to collapse?

These motivate Phase 0.5's ablation matrix.

---

## 2. Phase 0.5 — PPO-Side Knob Ablation Matrix (A–E)

### 2.1 The ablation design

Five experiments, each isolating one PPO-side hypothesis:

| Exp | Hypothesis | Key knob |
|---|---|---|
| **A** | KL controller too slow + ceiling too low | `kl_beta=1.0, kl_min=1.0, kl_max=50.0, adapt_rate=0.5, target=0.005` |
| **B** | Training-batch acc too noisy for early-stop | `--use_eval_for_early_stop`, `--early_stop_patience 3` |
| **C** | LR too high | `learning_rate=5e-5` |
| **D** | Reward variance dominated by small batch | `gradient_accumulation_steps=8` (effective BS=64) |
| **E** | Single-step weight delta too large | `max_grad_norm=0.1` (was 1.0) |

### 2.2 Phase 0.5 infra shipped

Beyond the experiment matrix:

- **Grad-norm logging fix.** `policy_grad_norm` and `value_grad_norm` were being
  computed only inside `if accelerator.sync_gradients:` — so 3/4 logged rows showed 0.
  Fixed to log every micro-batch.
- **Mid-training held-out eval hook.** `PPOVLMController.evaluate_subset()` runs
  constrained greedy decoding on a deterministic tail-slice (default n=64) every N steps;
  logged as `event=midtrain_eval` rows in `metrics.jsonl`.
- **CLI plumbing.** `--max_grad_norm`, `--use_eval_for_early_stop`, `--midtrain_eval_*`
  added to `args.py`; `run_training` Modal entrypoint accepts the new kwargs.

Commit: `025088c` on `phase0_collapse_mitigation`.

### 2.3 Results — Exp A (aggressive KL)

| Step | KL | β | Eval acc |
|---:|---:|---:|---:|
| 30 | 0.001 | 1.00 | 0.6492 |
| 40 | 0.025 | 2.68 | 0.7401 |
| **50** | 0.069 | 10.7 | **0.8028** (peak) |
| 60 | 0.055 | **50.0 (pinned)** | — |
| 120 | 0.013 | 50.0 | 0.7178 (small recovery) |
| ep1-end | 0.011 | 50.0 | **0.5689** |

**Findings.**
- **KL controller succeeded at its job**: max observed KL = 0.069 (vs Phase 0's 0.82).
- **Peak dropped by 19 pp** (0.9949 → 0.8028) — over-regularisation suppressed the
  learning signal.
- **Collapse still happened**, just at lower amplitude (0.57 vs 0.54 end). The KL
  ceiling pinned at 50.0 by step 60 and never released.

### 2.4 Results — Exp E (tight grad clip)

`max_grad_norm = 0.1` instead of 1.0. Full eval sweep across 27 checkpoints on the
full 2916-sample SB-Bench test set:

| Tag | Exp E acc | Exp A acc | Δ (E − A) |
|---|---:|---:|---:|
| step10 | 0.6142 | 0.6108 | +0.003 |
| step30 | **0.7606** | 0.6492 | +0.112 |
| step50 (KL spike) | 0.7281 | **0.8028** | −0.075 |
| step70 | 0.7136 | 0.6495 | +0.064 |
| **step80 (E peak)** | **0.7709** | 0.5336 | **+0.237** |
| step90 | 0.7541 | 0.4969 | +0.257 |
| step120 | 0.5473 | 0.7178 | −0.170 |
| step194 (resume; β reset to 0.10 by bug) | — | — | — |
| step240 | 0.6361 | 0.5641 | +0.072 |
| **ep1-end** | **0.6646** | 0.5689 | **+0.096** |

**Findings.**
1. **Tight grad clip did NOT prevent the KL spike** at step 50 (KL = 0.802, almost
   identical to Phase 0's 0.76). The clipped per-step delta accumulates over multiple
   steps to the same total drift.
2. **Tight grad clip DID shift the peak** from step 50 → step 80, and let E keep
   improving for 30 steps past the KL spike (while A immediately decayed).
3. **E retained the most signal at ep1-end of any run so far**: 0.6646 vs A's 0.5689
   vs Phase 0's 0.5377.
4. **Peak-to-end drop was smallest** for E (10.6 pp) vs A (23.4 pp) vs Phase 0 (45.7 pp).
   Tight grad clipping **softens** the collapse trajectory even though it doesn't
   prevent the initial KL eruption.
5. **A serendipitous bug helped.** At a forced resume around step 194 the checkpoint
   loader failed to persist `kl_beta` controller state — β reset to 0.10. Accuracy then
   *climbed* from 0.61 to 0.66. This was a real signal: **the β ceiling and the
   controller's pinning behaviour were an active suppressor**, not only the reward shape.

### 2.5 Comparative summary

| Metric | Phase 0 | Exp A | **Exp E** |
|---|---:|---:|---:|
| Max KL observed | 0.82 | **0.069** | 0.80 |
| β ceiling | 5.0 | 50.0 | 5.0 |
| Eval peak | **0.9949** | 0.8028 | 0.7709 |
| Eval peak step | 50 | 50 | **80** (shifted later) |
| Eval ep1-end | 0.5377 | 0.5689 | **0.6646** |
| Peak-to-end drop | −0.457 | −0.234 | **−0.106** |

### 2.6 Key reinterpretation

The Phase 0 0.9949 peak occurred *during* an uncontrolled KL spike — it was an
**unstable transient**, not a stable mode of operation. Once any regularisation is
added that prevents the spike, the stable ceiling drops to ~0.77–0.80. **The 0.77
ceiling — not 0.99 — is the true Pareto frontier of binary-reward PPO on SB-Bench.**

### 2.7 Decision: stop ablating PPO knobs

A + E together disproved three of the four PPO-side hypotheses (controller too lax,
controller too slow, step too large). B/C/D were skipped as low-EV.

**Beating 0.77 stably requires changing the reward shape.** Phase 0.6 is therefore the
DRM (Decomposed Reward Model) implementation — replace the sparse binary scalar with
dense, per-token projections of hidden state `φ` onto direction vectors learned from
labelled bias-bench pairs.

---

## 3. Phase 0.6 — Making DRM Reward Functionally Correct (D1/D2/D3)

### 3.1 The diagnosis: DRM was silently broken in three places

Switching `--reward-mode binary → svm` produced a **zero-variance reward** — PPO was
running on noise. Three independent bugs were the cause:

| ID | One-liner | Status entering Phase 0.6 |
|---|---|---|
| **D3** | Heads trained on a `φ` that does not match what PPO scores at training time | broken |
| **D2** | All heads loaded unfiltered, including category-degenerate ones | broken |
| **D1** | Reward heads projected onto the **moving** `φ_active`, not the frozen `φ_ref` they were fit on | broken |

Implementation order: **D3 → D2 → D1**, gated after each.

### 3.2 D3 — token-position alignment of `φ`

**The bug.** The Phase-1 head-generation pipeline extracted `φ` from the **EOS** of a
free-text completion. At PPO time, the trainer projected `φ` from the **per-generated-token
positions of a short letter completion**. The two distributions are not the same vector
space — at best the SVM separates noise, at worst the head normal vectors point in
directions the trainer can never produce.

**The fix.** Two new flags propagated through `extract.py` and `generate_drm_heads.py`:

- `--completion-format {free_text, letter}` — controls whether the assistant turn is
  the dataset's free-text answer or just the letter (`A` / `B` / `C`).
- `--token-position {eos, pre_letter, post_letter}` — controls which token position's
  hidden state is taken as `φ`.

We tried `pre_letter` first; it produced a **degenerate SVM** (acc ≈ 0.5, ||w|| ≈ 0)
because chosen and rejected share the same prompt prefix up to the pre-letter position.
**`post_letter`** corresponds to the hidden state immediately after the letter has been
emitted — which is exactly what the trainer projects against at the answer position.

**Acceptance.** SVM heads at `post_letter` on the held-out test split scored **acc = 1.00**
across all 9 SB-Bench categories. (Original requirement: `overall_mean ≥ 0.65 AND per-cat min ≥ 0.55`.)

### 3.3 D2 — kept-heads filter

**The bug.** `load_pca_components` loaded **all** `component*.pth` files from the
heads dir. For SVM this meant all 9 category-specific heads, including any that failed
to learn a useful boundary. For PCA, all 50 components, most of which are noise after
the top ~10 eigenvectors. The PPO trainer mean-pools the K projections through the
FastRL `alpha`, so noisy heads dilute the signal and let the policy hack the reward by
drifting `φ` into the null space of the genuine heads.

**The fix.** `evaluate_drm_heads.py` gained `--head_type`, `--keep_threshold` (default
0.55), and writes `kept_heads.json` with:
- `kept_indices`: list of original head indices that passed
- `category_coverage`: `{category_name: [head_indices_that_help_with_it]}`

**SVM rule:** keep head `i` iff its own category's `accuracy_mean ≥ threshold`.
**PCA rule:** keep head `i` iff `overall_per_head[i] ≥ threshold`.

`drm_loader.load_pca_components` gained an optional `kept_heads_filter` argument; when
set, it loads the JSON, validates indices, and subsets to just those heads.

**Acceptance.** With D3 producing acc = 1.0 on all 9 SVM categories, the
`≥ 5 SVM heads AND ≥ 1 per category` gate trivially passed — 9/9 SVM heads kept.

### 3.4 D1 — frozen-φ projection

**The bug.** The trainer was projecting reward heads against the **active** (LoRA-on)
penultimate hidden state:

```python
h_active = curr_penultimate[:, :-1, :]
r_token_k = torch.matmul(h_active, self.reward_heads_weight.T)
```

But the heads were fit on `φ` from the **un-adapted** base model. As LoRA trains,
`φ_active` drifts. The policy can then increase reward without changing its answer
distribution at all, simply by drifting `φ` into directions the heads happen to score
highly — classic **representation hacking**. The reward stops being a function of
model behaviour.

**The fix.** Reuse the `h_ref` that the trainer **already** computes inside
`with self.policy.disable_adapter()` for the KL penalty:

```python
h_reward = h_ref if self.use_frozen_phi else h_active
r_token_k = torch.matmul(h_reward, self.reward_heads_weight.T)
```

A new `--use_frozen_phi` flag (default off → preserves backcompat with binary-mode
Exp E 0.7709 baseline) controls it.

**Acceptance.** A 1-batch smoke run with `--reward_mode svm --use_frozen_phi --kept_heads_filter ...`
logs a per-token `reward_task` with std > 0.01. Smoke result on Modal:
`reward_task=-1.375, reward_task_std=1.0, policy_grad_norm=2.114`. 252/504 trainable
LoRA params received non-zero gradients on the first backward pass.

### 3.5 Phase 0.6 reference training command

```bash
modal run src/run_modal.py::run_training \
  --epochs 1 --max-train-samples 2000 --batch-size 8 --max-gen-tokens 8 \
  --reward-mode svm --head-type svm --use-frozen-phi \
  --heads-suffix _letter_post_letter \
  --kept-heads-filter /mnt/data/generated_heads_letter_post_letter/kept_heads_svm.json \
  --learning-rate 1e-4 --value-learning-rate 5e-4 \
  --kl-beta 0.1 --target-kl 0.02 --kl-adapt-rate 0.1 \
  --max-grad-norm 0.1
```

### 3.6 Phase 0.6 F result: SB-Bench saturation at 1.0000

The SVM-DRM run with all three fixes (`phase06_F_no_causal`) reached **SB-Bench
acc = 1.0000 at step 70**, sustained through `ep1-end` across 18 consecutive
checkpoints. The confusion matrix was perfectly diagonal — **mode-collapse rejected**.

This is the result that motivated Phase 0.7's discrimination question: is the 1.0000
genuine debiasing transfer, or is it just **in-distribution oracle distillation** (since
the SVM heads scored 1.0 on the same chosen-vs-rejected test split during D3 acceptance)?

---

## 4. Phase 0.7 — Verifying the 1.0, Then Testing Transfer

### 4.1 Acceptance gates

| Gate | Test | Pass criterion |
|---|---|---|
| **G1** | P1 — `question_polarity` split on `ep1-end.jsonl` | Both polarities ≥ 0.95 |
| **G2** | P2 — letter-order perturbation (cyclic-1 shift) | Perturbed acc ≥ 0.95 |
| **G3** | P3 — POPE regression | Drop ≤ 5 pp vs base Qwen2.5-VL-3B-Instruct |
| **G4** | T1 — BBQ / VLBiasBench transfer | BBQ acc strictly > base-model BBQ acc |

### 4.2 G1 — polarity & gold-is-unknown splits (P1)

**Source:** 2916 generations from `checkpoint-ep1-end` of `output_ppo_phase06_F_no_causal`.

Every cell scored **1.0000**:

| split | acc | n |
|---|---:|---:|
| polarity = negative | 1.0000 | 1489 |
| polarity = nonneg   | 1.0000 | 1427 |
| gold_is_unknown = False (named-gold) | **1.0000** | **618** |
| gold_is_unknown = True (safe-option-gold) | 1.0000 | 2298 |

Hypotheses ruled out:
- "Always answer the unknown option" — the 618 named-gold items are all correct.
- "Only wins on the easier polarity framing" — 1.0 on both polarities.
- "Only wins where gold = safe option" — named-gold subset is 1.0 too.

**G1 PASS.**

### 4.3 G2 — letter-order perturbation (P2 cyclic-1)

**Design.** Rotate `(ans0, ans1, ans2)` by one position before generation. The
originally-correct answer text moves from old position `L` to new position `(L − 1) mod 3`;
the eval gold label is updated to match. If the policy is reasoning about answer
**content**, accuracy holds; if it is reading **position**, accuracy collapses to ~0.33.

| ckpt | perturbed acc | acc if scored vs ORIGINAL label |
|---|---:|---:|
| `ep1-step70` | **1.0000** (n=2916) | 0.0000 |
| `ep1-end`    | **1.0000** (n=2916) | 0.0000 |

Per-category: all 9 categories at 1.0000. Predicted letters exactly track rotated gold
positions: pre-rotation A=959/B=1008/C=949; post-rotation gold A=1008/B=949/C=959;
predicted A=1008/B=949/C=959. The exact 0.0000 against original positions confirms the
policy followed gold text into its new position, not the previously-correct letter slot.

**G2 PASS.** Combined with G1: the policy is reading the **answer text** and routing it
to the correct letter even under permutation. This is the strongest in-distribution
evidence achievable without a held-out benchmark — but it does not yet falsify
"in-distribution oracle distillation."

### 4.4 G3 — POPE capability regression

POPE = 9000 yes/no object-presence VQA items. Tests whether SVM-PPO broke general VQA.

| ckpt | acc | Δacc | F1 | ΔF1 | unknown-rate |
|---|---:|---:|---:|---:|---:|
| **base** Qwen2.5-VL-3B-Instruct | 0.8714 | — | 0.8746 | — | 0.0068 |
| `ep1-step70` | 0.8636 | **−0.79 pp** | 0.8693 | −0.53 pp | 0.0016 |
| `ep1-end`    | 0.8668 | **−0.47 pp** | 0.8718 | −0.28 pp | 0.0012 |

Both well inside the 5 pp tolerance. **Capability tax is negligible** (< 1 pp on both
checkpoints). Unknown-rate **dropped** — model is more decisive without sacrificing
accuracy. POPE is a different task family (yes/no, no MCQ letters), so the LoRA delta
is transparent to it. **G3 PASS.**

### 4.5 PCA-DRM secondary control variant

To avoid pinning the whole programme to one head family before transfer, we trained
an analogous PPO run using **PCA-100 DRM heads** (90/200 kept above 0.55 signal cutoff).
All other hyperparameters bit-identical to Phase 0.6F.

P1 (n=2916, SB-Bench test):

| polarity | acc |
|---|---:|
| negative | 0.9181 |
| nonneg | 0.9166 |
| gold = named group (n=618) | **0.8333** |
| gold = unknown (n=2298) | 0.9399 |

The PCA G1 nominally **FAILS** (< 0.95). But what matters is the *shape* of the failure:
**a 10.6 pp gap** between gold-unknown (0.9399) and named-gold (0.8333), monotone across
both polarities. The PCA policy is systematically more accurate when the safe option *is*
the gold → it learned a **conservative / "prefer-unknown" bias**. This is the opposite
of stereotype bias, but it is still a measurable bias signature — and it became a
*predictive* finding for what would happen at G4.

### 4.6 G4 — VLBiasBench transfer (the gold-standard signal)

**Sample.** 3000 records stratified across (bbq_axis × condition) = 10 × 3 = 30 cells
(seed=42, 1000 each per condition, 300 per BBQ axis). Filter `qformat ∈ {base, scene, scene_text}`.

| tag | OvAcc | Δ vs base | AmbigAcc | DisambigAcc | BIAS |
|---|---:|---:|---:|---:|---:|
| **base** | 0.5513 | — | 0.9190 | 0.3675 | +0.0070 |
| **svm_ep1-50pct** | **0.5873** | **+3.60 pp** | 0.4080 | **0.6770** | +0.0200 |
| **svm_ep1-end** | 0.5700 | +1.87 pp | 0.2030 | **0.7535** | **−0.0670** |
| **pca_ep1-step80** | 0.3367 | **−21.47 pp** | **1.0000** | 0.0050 | −0.0100 |
| **pca_ep1-end** | 0.3387 | −21.27 pp | **1.0000** | 0.0080 | −0.0100 |

### 4.7 Why overall accuracy is the wrong primary metric here

Base model OvAcc (55.13%) is almost entirely driven by ambig (91.9%) — i.e. by routinely
answering "Can't be determined". Disambig (where context fully resolves the answer) is at
chance: 36.75% ≈ 1/3. **Base is not engaging with the visual/contextual signal on
disambig; it succeeds on ambig by predicting "unknown" indiscriminately.**

The correct view splits the two:

| Variant | Ambig acc (gold = Unknown) | Disambig acc (gold = named group) | Δ disambig vs base |
|---|---:|---:|---:|
| base | 0.9190 | 0.3675 | — |
| svm_ep1-50pct | 0.4080 | **0.6770** | **+30.95 pp** |
| svm_ep1-end | 0.2030 | **0.7535** | **+38.60 pp** |
| pca_ep1-step80 | 1.0000 | 0.0050 | **−36.70 pp** |
| pca_ep1-end | 1.0000 | 0.0080 | −36.67 pp |

**SVM doubles-to-triples disambig accuracy** — the policy stops over-predicting "unknown"
and starts engaging with the disambiguating context. **PCA mode-collapses** to predicting
C on 99.5–99.7% of all 3000 records. This is the **exact failure mode predicted in §4.5**
by the in-distribution conservative-bias signature — confirming the P1 split is a
*predictive* probe for transfer-time failures, not just a literal gate.

### 4.8 SVM late-training drift

Global bias score:

| Variant | Bias score | Direction |
|---|---:|---|
| base | +0.0070 | neutral |
| svm_ep1-50pct | +0.0200 | neutral / slight counter-stereo |
| svm_ep1-end | **−0.0670** | **stereotype-amplification** |

By `ep1-end` the SVM policy has drifted further in the stereotype-confirming direction
on the highest-base-bias axes:

| BBQ axis | base bias | svm 50pct | svm end | drift end vs 50pct |
|---|---:|---:|---:|---:|
| Race_x_SES | −0.16 | −0.12 | **−0.38** | −0.26 pp |
| Race_x_gender | −0.08 | +0.06 | −0.18 | −0.24 pp |
| Disability_status | +0.14 | +0.05 | −0.19 | −0.24 pp |

**Continued PPO past in-distribution saturation does not preserve fairness.** SVM
`ep1-50pct` is the recommended deployment checkpoint.

### 4.9 SVM visual-reasoning regression

Invisible to POPE (yes/no), but G4 surfaces it:

| qformat | base acc | svm_50pct acc | svm_end acc |
|---|---:|---:|---:|
| base | 0.527 | 0.595 | 0.571 |
| scene | **0.726** | 0.562 | 0.557 |
| scene_text | **0.693** | 0.505 | 0.568 |

**SVM degrades on natural-scene images** (−17 pp, −19 pp). PPO on SB-Bench (essentially
text-driven MCQ) blunted the natural-image grounding. The `base` qformat improves
slightly (0.53 → 0.60) because it is most similar to SB-Bench style. This is a known
regression in any shipping artefact from this recipe.

### 4.10 T1.1 — the bridge to Phase 0.8 (REWARD-side pathology proof)

**Scored 990 stratified VLBiasBench responses with both SVM and PCA heads against the
**base** Qwen2.5-VL-3B-Instruct policy** (no PPO). Same processor, chat template,
`post_letter` position as the PPO eval.

Cell-level reward landscape (base model, no PPO):

| Head set | `ambig::C` (right, gold=C) | `disambig::correct` (right, gold=named) | `disambig::incorrect` (picked C, gold=named) | **C-bias** = ambC − dis_corr | **Confusion** = ambC − dis_inc |
|---|---:|---:|---:|---:|---:|
| SVM (K=9)  | **−10.95** | −20.95 | −12.40 | **+10.28** | **+1.47** |
| PCA (K=90) | **−1.91**  | −3.69  | −2.20  | **+1.79**  | **+0.29** |

Both pathologies are present on the **base model** with **no PPO**:

1. **C-bias ≫ 0.** Picking C beats picking the correct named option. PPO will exploit
   this. **PCA's +1.79 explains the mode-collapse exactly** — PCA-PPO discovers
   "always emit C" → +1.79 reward → 99.5% C-rate at convergence. The math is exact.
2. **Confusion ≈ 0.** The reward cannot distinguish "C because gold is C" from "C as
   refusal cop-out". They look identical to the heads. This is why PPO cannot learn
   *when not to* pick C.

Per-head structure:

| Head set | K | Heads preferring C | Heads preferring named | Heads with `|confusion| < 0.3` | C-bias std |
|---|---:|---:|---:|---:|---:|
| SVM | 9   | **9/9 (100%)** | 0 | 0 | 1.63 |
| PCA | 90  | 59/90 (66%) | 31/90 (34%) | **36/90 (40%)** | 4.76 |

**SVM** is structurally homogeneous — all 9 heads agree on "prefer C", but the §4.6
result still showed +30.95 pp disambig improvement. FastRL's α-weighting plus the
sign-alignment work extracted a *named-preferring* combination *despite* the unanimous
C-bias in the equal-mean view. **PCA** is heterogeneous but noisy: top-5 C-preferring
heads have extreme bias (head 1 = +28.67), and 40% of PCA heads are indifferent between
good-C and bad-C — pure noise diluting the α-weighted reward.

**Verdict.**

| Hypothesis | T1.1 status |
|---|---|
| **Reward-side**: heads prefer C / cannot distinguish good-C from bad-C | ✅ **CONFIRMED** |
| **PPO-side**: KL too loose, α collapsed | ❌ **Falsified** — PCA mirrors base reward landscape exactly |
| **Data-side**: ambig dominates training mix | Secondary — irrelevant if reward is structurally biased |

### 4.11 Why this motivates Phase 0.8

The SVM/PCA-DRM heads were built on the **penultimate** hidden state of Qwen2.5-VL-3B.
That state is task-conditioned (it must encode "which letter to output"); the bias-content
representation competes with answer-formatting for the same dimensions. The probing
literature (Tenney, Hewitt & Manning, Belinkov, Schwettmann et al.) repeatedly shows
demographic / social features peak at **middle** transformer layers, not final ones.
And the vision-language fusion in Qwen2.5-VL adds a further hypothesis: bias signal may
live in cross-modal fusion layers (LM blocks ~12–24) before being compressed by
late-layer answer-formatting circuits.

**Phase 0.8 hypothesis (to be tested, not assumed):** there exists a layer L* < penultimate
such that a linear probe of L* hidden states exhibits lower C-bias and higher
good-C-vs-bad-C confusion than the penultimate SVM/PCA heads.

---

## 5. Trajectory Summary — What the Pre-Phase-0.8 Work Established

| Conclusion | Phase | Evidence |
|---|---|---|
| Binary letter-correctness reward cannot debias via PPO | 0 | Mode-collapse to 63% B, peak 0.99 was an unstable transient |
| PPO knobs cannot rescue binary reward | 0.5 | Exp A (KL controller) capped peak at 0.80; Exp E (grad clip) only softened collapse to 0.66 at ep-end |
| The DRM (SVM/PCA) reward pipeline had 3 independent silent bugs | 0.6 | D3 token-position misalignment, D2 noisy-head loading, D1 moving-φ representation hacking |
| Fixed SVM-DRM saturates SB-Bench at 1.0000 | 0.6F | Phase 0.6F no-causal, sustained through 18 checkpoints |
| 1.0000 SB-Bench is **content-routing**, not letter-position memorisation | 0.7 G1+G2 | P1 cells all 1.0; P2 cyclic-1 also 1.0; cross-permutation perfectly tracked |
| Capability tax is negligible | 0.7 G3 | POPE Δacc = −0.47 pp at ep1-end |
| SVM transfers to VLBiasBench, **PCA mode-collapses** | 0.7 G4 | SVM disambig +30.95 pp / +38.60 pp; PCA 99.5% always-C |
| **The transferable SVM gain is highest at `ep1-50pct`**; late training drifts | 0.7 §5.6 | bias_score goes +0.020 → −0.067 from mid to end; Race_x_SES bias −0.38 at ep1-end |
| SVM has a scene-image visual-reasoning regression | 0.7 §5.8 | −17 pp on scene qformat, invisible to POPE |
| **Both pathologies (SVM late drift, PCA mode-collapse) are REWARD-side, not PPO-side** | 0.7 T1.1 | Base-model reward landscape already biased: PCA C-bias = +1.79, SVM C-bias = +10.28 |
| Penultimate-layer heads conflate bias and answer formatting | 0.7 T1.1 → 0.8 | Confusion ≈ 0 for both head families: cannot distinguish "C because gold = C" from "C as refusal" |
| **Pivot to probe-as-reward at a mid-layer** | 0.8 | Test whether a linear probe at L < penultimate can decouple bias from answer formatting |

The Phase 0.8 deep-dive picks up here: probe design, layer audit, PPO recipe, KLFIX,
wash-out diagnostic, two-regime bias geometry. See [REPORT_EXTENSIVE.md](REPORT_EXTENSIVE.md).

---

## Appendix A. Reproduction-Footnote Index

**Phase 0 / 0.5:**
- Commits: `3d5b01e` (Phase 0), `025088c` (Phase 0.5 infra), branch `phase0_collapse_mitigation`.
- Modal volume artefacts:
  - `/mnt/data/output_ppo_phase0/` — Phase 0 baseline
  - `/mnt/data/output_ppo_phase05_A_high_kl_floor/` — Exp A (with `phase05_eval_sweep.json`)
  - `/mnt/data/output_ppo_phase05_E_tight_grad_clip/` — Exp E (with `phase05_eval_sweep.json`)
- Scripts: [scripts/phase05_ablation.sh](../scripts/phase05_ablation.sh),
  [scripts/phase0_letter_distribution.py](../scripts/phase0_letter_distribution.py).
- Local copies of metrics: `phase05_results/` (A and E training metrics + A's eval sweep).
- Detailed handoff: [scratch/Phase0.5_Context_Handoff.md](../scratch/Phase0.5_Context_Handoff.md).

**Phase 0.6:**
- Branch `phase0_collapse_mitigation`; D1/D2/D3 work documented in [Phase0.6.md](../Phase0.6.md).
- New Modal entrypoint: `run_drm_eval` (see `src/run_modal.py`).
- Heads dir suffix: `/mnt/data/generated_heads_letter_post_letter/` (added by D3
  auto-suffixing).
- Final SVM-DRM checkpoints under `/mnt/data/output_ppo_phase06_F_no_causal/`.

**Phase 0.7:**
- Branch `phase0.7_transfer_eval`; documented in [Phase0.7.md](../Phase0.7.md).
- Remediation analysis: [Phase0.7_Remediation_Plan.md](../Phase0.7_Remediation_Plan.md).
- VLBiasBench gens: `/mnt/data/phase07_vlbiasbench/{base,svm_ep1-50pct,svm_ep1-end,pca_ep1-step80,pca_ep1-end}_vlbias_gen.jsonl`.
- POPE results: `/mnt/data/phase07_pope/{base,phase06F_step70,phase06F_end}_pope_results.json`.
- T1.1 offline reward scoring: [src/modules/evaluation/score_vlbias_offline.py](../src/modules/evaluation/score_vlbias_offline.py),
  [scripts/phase07_t1_offline_reward.sh](../scripts/phase07_t1_offline_reward.sh).
