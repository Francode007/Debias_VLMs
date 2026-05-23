# Critical Review — C-DeFR-L v4 (Qwen2.5-VL-3B Debiasing)

> Result under review: **66.67 % vs 66.56 %** vanilla on SB-Bench (Δ = +0.11 pp, well inside SE).
> This document records the full deep audit performed against the actual implementation,
> the four flaws sufficient on their own to explain the null result, and a five-phase
> remediation plan. References are file/line links into the live codebase.

---

## 0. TL;DR

| Rank | ID | Flaw | Why fatal | Fix phase |
|---|---|---|---|---|
| 1 | **B1** | PPO ratio uses `π_active / π_ref` **and** KL-to-ref is *also* in the reward | Two trust regions toward the base policy → ratio≈1, advantages cancel, gradient collapses. Matches observed `logit_r≈0`, `ratio≈1`. | Phase 1 |
| 2 | **A2** | SVM heads trained on the EOS of `(prompt + full answer text)`, then queried on the EOS of `(prompt + single letter)` | Reward function evaluated **out of distribution** of its training set → projection magnitude / sign no longer meaningful. | Phase 0 → fix in Phase 2 |
| 3 | **B2** | Reward normalized by EMA z-score **and** advantages renormalized batch-wise | Double-centering removes directional information; surrogate loses sign consistency across batches. | Phase 3 |
| 4 | **C1** | `max_new_tokens=256` for a *single-letter* answer prompt | Effective generation length T≈1–3 valid tokens, most of the sequence is padding → dense GAE is theatrical. | Phase 3 |

Any one of these alone is enough to nullify learning. All four are present.

---

## 1. Architecture, as actually implemented

| Component | Location | Reality |
|---|---|---|
| Base policy | [src/modules/training/setup.py](src/modules/training/setup.py#L66-L78) | Qwen2.5-VL-3B-Instruct, bf16, FA2, left-padded |
| LoRA | [setup.py](src/modules/training/setup.py#L70-L82) | r=16, α=32, targets `{q,k,v,o,gate,up,down}_proj`. **No assertion that vision tower is excluded.** |
| Reference policy | [custom_vlm_ppo_trainer.py](src/modules/rl_components/custom_vlm_ppo_trainer.py#L262-L268) | `policy.disable_adapter()` context — correct, no second model copy |
| Reward heads | [generate_drm_heads.py](src/modules/embeddings/generate_drm_heads.py) | 9 × `LinearSVC(C=1.0)` on penultimate hidden state at last non-pad of `(prompt + full answer text)`. All heads reach **1.0 train accuracy** in 2048-D → memorization, not generalization. |
| Logprobs in surrogate | [custom_vlm_ppo_trainer.py:364](src/modules/rl_components/custom_vlm_ppo_trainer.py#L364) | `ratio = exp(curr_logprobs - ref_logprobs.detach())` — **uses ref instead of old policy** |
| KL term | [custom_vlm_ppo_trainer.py:316-322](src/modules/rl_components/custom_vlm_ppo_trainer.py#L316-L322) | Schulman k3 KL(π‖π_ref), clamped ≤10, subtracted from reward |
| Reward composition | [custom_vlm_ppo_trainer.py:350](src/modules/rl_components/custom_vlm_ppo_trainer.py#L350) | `r_task + logit_reward + causal − β·kl`, all masked to generated positions |
| Eval generation | [generate_sb_bench_answers.py:174](src/modules/inference/generate_sb_bench_answers.py#L174) | `generate(..., max_new_tokens=10)` — **no constrained decoding to {A,B,C}** |
| Stat reporting | README §Results | One-pass accuracy, SE assumes independence; **no McNemar paired test** |

---

## 2. Critical flaws (sufficient alone)

### B1 — Double trust region toward the base policy
**File:** [custom_vlm_ppo_trainer.py:364](src/modules/rl_components/custom_vlm_ppo_trainer.py#L364)

```python
ratio = torch.exp(curr_logprobs - ref_logprobs.detach())   # ← ref, not old
```

PPO requires the surrogate to use **π_old**: the snapshot of the policy at the time the
rollout was sampled. Here, "old" has been silently replaced with **π_ref** (the
base policy with LoRA disabled). Simultaneously, a KL-to-ref penalty
(`token_kl`) is added inside the reward at
[line 350](src/modules/rl_components/custom_vlm_ppo_trainer.py#L350). The
surrogate therefore minimises drift toward π_ref **twice**:

* once via ratio clipping in objective space
* once via KL penalty in reward space

Net effect: every gradient step that improves task reward is countered by an
equal-and-opposite pressure to remain identical to the un-adapted base model.
The empirical fingerprint is the observed `ratio≈1`, `|Δlogp|≈0`, `logit_r≈0`
even though `r_task` moves. The policy is allowed to *think* but not to *act*.

### A2 — Distribution mismatch between SVM training and reward use
**Training:** [reward_trainer.py:280-314](src/modules/utils/reward_trainer.py#L280-L314)
+ [dataset_builder.py:290-300](src/modules/utils/dataset_builder.py#L290-L300).
The SVM is fit on `h_{-2}(EOS)` where the input is `prompt + apply_chat_template(full-answer text)`.

**PPO inference:** [custom_vlm_ppo_trainer.py:282](src/modules/rl_components/custom_vlm_ppo_trainer.py#L282).
The very same projection is queried on the policy's *generated continuation*,
which (because the prompt ends with "Answer with only the letter (A, B, or C):")
is typically **one letter token** followed by `<|im_end|>`.

Linear classifiers in 2048-D that hit 1.0 train accuracy in 9 categories on
a few hundred samples each are almost certainly latching onto lexical artifacts
of the chosen text (length, punctuation, specific n-grams). The single-letter
hidden state at inference contains none of those artifacts → the decision
function evaluates in a region the SVM never saw.

### B2 — Double normalization of the reward signal
[custom_vlm_ppo_trainer.py:304-312](src/modules/rl_components/custom_vlm_ppo_trainer.py#L304-L312)
applies an EMA z-score to `r_task_dense`. Then
[lines 358-360](src/modules/rl_components/custom_vlm_ppo_trainer.py#L358-L360)
re-normalize advantages batch-wise. After the second centering, the sign of an
advantage no longer reliably reflects the sign of the underlying reward
relative to baseline — it only reflects the rank within the *current batch*.
With small batches (B=12) and per-token granularity, the resulting advantages
are dominated by within-batch sampling noise.

### C1 — Sequence-length theatre
`max_new_tokens=256` ([custom_vlm_ppo_trainer.py:167](src/modules/rl_components/custom_vlm_ppo_trainer.py#L167))
for a prompt that explicitly demands a single letter. The effective valid
trajectory is ~1 task token + `<|im_end|>`. GAE over a length-256 window with
λ=0.95 then propagates *noise* from positions that are not even generated by
the model (they were stopped by EOS) into the credit assignment of the
single useful token. The dense reward architecture's whole motivation collapses.

---

## 3. High-severity flaws (each materially harmful)

* **A4 — Lexical artifact SVMs.** All 9 heads hit 1.0 train accuracy at 2048-D
  with C=1.0 ([generate_drm_heads.py:195-360](src/modules/embeddings/generate_drm_heads.py)).
  This is a memorization signature; the heads cannot be trusted as causal bias detectors
  without a held-out / activation-steering validation.
* **A3 — Text-only reward on a visual-bias benchmark.** SVMs are trained from
  `(prompt + text-only chat template)` hidden states; no vision tokens
  contribute to the training point. SB-Bench bias is image-conditioned.
* **B6 — KL on sampled tokens punishes exploration.** Because the KL term sits
  inside the reward (not as an external regularizer) and is applied to the
  *sampled* tokens, the gradient pushes the policy back toward the action
  distribution of π_ref — i.e. the biased base policy is implicitly used as a
  conservatism prior.
* **B5 — Value head shares no gradient path with reward.** The Linear value
  head reads `hidden_states[-1]` of the policy
  ([line 99](src/modules/rl_components/custom_vlm_ppo_trainer.py#L99)) while the
  reward is computed on `hidden_states[-2]`; mismatched layers make the
  value bootstrap biased.

## 4. Medium-severity / numerical hygiene

* **D1 — Off-by-one in hidden state alignment.** `h_active = curr_penultimate[:, 1:, :]`
  ([line 282](src/modules/rl_components/custom_vlm_ppo_trainer.py#L282)). The logprob shift
  produces predictions for positions `1..T-1` *from* positions `0..T-2`, so the
  contextual hidden states that justified those predictions are
  `penultimate[:, :-1, :]`, not `[:, 1:, :]`.
* **D2 — Unbounded dispersive loss.** `-log(d+eps)` blows up as `d → 0`
  ([caa_feedback.py:75-83](src/modules/rl_components/caa_feedback.py#L75-L83)).
  Switch to a bounded form (RBF or hinge).
* **D4 — No gradient clipping.** Nowhere in the training stack
  ([train_rl.py:80-110](src/modules/training/train_rl.py#L80-L110)).
* **D5 — No LR warmup.** `lr=1e-5` flat from step 0 on top of bf16 +
  custom losses → first batches dominate Adam state.
* **E2 — Penultimate may be too late.** Bias direction extraction at layer
  `-2` (≈ layer 34/36) sits past the model's representational refactor;
  earlier layers (12–24) tend to carry more linearly-decodable concept
  directions.
* **E5 — `_left_pad_generated` misclassifies generated pad tokens.**
  `(seq != pad_id)` ([line 188-211](src/modules/rl_components/custom_vlm_ppo_trainer.py#L188-L211))
  treats any generated `pad` token id as padding, which can truncate legitimate
  continuations.

## 5. Statistical / evaluation flaws

* **F1 — SE underestimated.** 14 578 samples, 9 categories, no clustering
  correction. Realistic CI on the headline 66.67 % is ±~1.2 pp, not ±0.87.
* **F2 — No McNemar / paired test.** Comparing two policies on the *same*
  test set demands a paired test, not two independent binomials.
* **F3 — No constrained decoding at eval.** `max_new_tokens=10` then regex parse
  ([generate_sb_bench_answers.py:174](src/modules/inference/generate_sb_bench_answers.py#L174));
  any sample where the model emits "Cannot…" or "The image…" before a letter
  becomes random noise.

---

## 6. DRM-paper alignment (paper 2502.13131v2, Luo et al. 2025)

The DRM method proposes:
1. Train a **reward model** (Bradley-Terry head) on preference pairs.
2. Build the dataset of **`emb_chosen − emb_rejected`** vectors from that reward model's penultimate state.
3. Apply **PCA** on the difference vectors → orthogonal preference axes.
4. At **inference** time, linearly combine the PCs to match a target user / objective.

This framework deviates in three concrete ways:

* Uses **raw embeddings**, not their *differences* (no PCA contrast). The
  `pca` mode in [generate_drm_heads.py](src/modules/embeddings/generate_drm_heads.py)
  takes PCA over raw hidden states, which captures *task* variance, not
  *preference* variance.
* Substitutes **SVM separating hyperplanes** for PCA components. SVM normals
  in 2048-D under perfect train separability are not orthogonal,
  not interpretable, and not what the paper validates.
* Substitutes **base-VLM hidden states** for a *trained reward model*'s
  hidden states. The DRM paper relies on the reward model already having
  preference-aligned representations; the bare Qwen base model carries
  task representations, not preference representations.

DRM is therefore **not IRL** (IRL recovers a reward from optimal-demonstration
*trajectories*; DRM decomposes an *already-trained* reward into orthogonal
axes). This codebase's use of "IRL" is a misnomer — it is closer to
*linear probe reward design* than to either DRM or IRL.

---

## 7. Five-phase remediation plan

### Phase 0 — Sanity gates (no training change)
Validate the reward signal before spending compute on it.

* **0a — Activation-steering sweep**:
  [scripts/phase0_activation_steering.py](scripts/phase0_activation_steering.py).
  Inject `h_L ← h_L + λ w_k` at layer L for λ ∈ {−3, −1.5, 0, 1.5, 3} and
  L ∈ {12, 18, 24, 30, 34}; measure SB-Bench accuracy delta per (k, L, λ).
  If no head moves the metric monotonically, the heads do not encode a
  causally usable direction → do not run PPO.
* **0b — Reward / metric correlation**:
  [scripts/phase0_reward_correlation.py](scripts/phase0_reward_correlation.py).
  For each test sample, score the SVM head at `(prompt + "A")`, `(prompt + "B")`,
  `(prompt + "C")`. Compare the projection of the *correct* letter vs the
  *wrong* letter on the *unbiased* split; check whether correct > wrong.
* **0c — McNemar paired test**:
  [scripts/phase0_mcnemar.py](scripts/phase0_mcnemar.py).
  Paired test on vanilla vs v4 answer JSONLs.

### Phase 1 — PPO ratio fix
* Snapshot `old_logprobs` *immediately after* generation, under no-grad.
* Surrogate: `ratio = exp(curr_logprobs − old_logprobs.detach())`.
* Move KL-to-ref **out of the dense reward** and into an optional
  explicit penalty (or drop it entirely once the ratio is correct).
* Remove `logit_reward` (no longer needed; was compensating for the broken ratio).

### Phase 2 — *Deferred by user.* SVM training distribution match.
Re-train heads at the same distribution at which they will be queried:
hidden states at the **single-letter EOS** of a `(prompt + letter)` continuation,
*or* aggregate over the entire generated sequence and use a sequence-pooled
projector at inference too.

### Phase 3 — Numerical hygiene
* Replace EMA z-score with **scale-only** (divide by running std, keep mean) and keep advantage normalization.
* Add `accelerator.clip_grad_norm_(..., 1.0)` before each `optimizer.step()`.
* Add LR warmup (5–10 % of total steps) via `transformers.get_linear_schedule_with_warmup`.
* Drop the dead causal penalty (δ_margin=1.0 is never crossed) **or** switch to
  quadratic `λ‖drift‖²` with measured drift quantile.
* Replace dispersive loss with bounded form: `mean(exp(−d²/2σ²))`.
* Assert `target_modules` contain no vision-tower substrings in
  [setup.py](src/modules/training/setup.py#L70-L82).
* Reduce `max_gen_tokens` from 256 → 8 for SB-Bench (matches eval).
* Constrained decoding at eval — restrict the first generated logit to the
  `{A, B, C}` token ids.

### Phase 4 — *Deferred by user.* Architecture upgrade.
Consider GRPO (no value head, group-relative baseline) and a layer sweep
for reward head extraction.

### Phase 5 — *Deferred by user.* Close the DRM loop properly.
Train an actual preference reward model on SB-Bench pairs; take PCA on the
difference embeddings *from that reward model*; combine top-K PCs as the
reward function.

---

## 8. Q&A appendix (live questions from user)

### A1 — Does SVM/DRM head creation count as IRL?
**No.** IRL recovers a reward function from observed (near-)optimal
*trajectories*. DRM (and this codebase's SVM variant) recovers reward
*structure* from a labelled preference dataset by decomposing or fitting
a linear function in embedding space. DRM is closest to *interpretable
linear reward modelling*; the SVM variant used here is even further from
IRL because no PCA contrast and no reward model is involved.

### A2 — Deeper read on the distribution mismatch
The SVM has only ever seen hidden states whose preceding text token sequence
is `…assistant\n{full chosen answer text}<|im_end|>`. At RL time it sees
`…assistant\n A<|im_end|>` (or B / C). The two hidden-state distributions
share a prefix but the final-token context is structurally different
(one is a content-bearing word ending a paragraph; the other is a single capital
letter). With 1.0 train accuracy at C=1.0 in 2048-D, the head almost
certainly relies on artifacts that disappear at inference. The
**diagnostic** is Phase 0a + 0b — if accuracy does not move under
activation steering and if the head cannot rank `(prompt + correct letter)`
above `(prompt + wrong letter)`, the head provides no usable reward signal,
no matter how PPO is fixed.

### A3 — Involving image components in the reward
* Train heads at a *multimodal-fused* layer, with image tokens present in
  the training input (not the text-only chat template).
* Project the reward at *image-token positions* too, not only at text positions.
* Add an auxiliary reward term that uses a small vision-language reward
  model (or CLIP similarity to a debiased caption) so the reward depends
  on the image content, not just the text continuation.

### B1 — What is the "old policy" supposed to be?
The policy **at the moment the rollout was sampled**, **before** any
optimizer step in the current update. In single-epoch on-policy PPO with
regeneration every batch, π_old ≡ π_θ right before `generate()` in this batch.
The fix is to compute `old_logprobs = log π_θ(generated_tokens)` in the same
no-grad block as generation, store them, and use them in the surrogate.
After the first optimizer step inside multi-epoch PPO, π_θ has shifted but
π_old stays fixed → the ratio meaningfully diverges from 1 → clipping does
something. Today the codebase silently substitutes π_ref for π_old, which
turns the surrogate into a KL-to-base regularizer.

---

*Last updated: deep audit pass after the +0.11 pp null result.*
