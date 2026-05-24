# DRM / CAA / FastRL Adoption Plan for VLM Debiasing PPO

**Scope.** Consolidated technical report covering (1) the empirical state of the current PPO pipeline, (2) collapse-mitigation playbook for the binary baseline that is already working, (3) line-by-line comparison of the current DRM / CAA / FastRL implementations against the source papers, and (4) a phased roadmap that uses each method correctly. All paper references are to the PDFs read end-to-end during this session:

- **DRM** — Luo et al., *Rethinking Diverse Human Preference Learning through Principal Component Analysis*, arXiv:2502.13131v2 (June 2025).
- **CAA** — Xia et al., *Aligning as Debiasing: Causality-Aware Alignment via Reinforcement Learning with Interventional Feedback*, NAACL 2024 (Long), pp. 4684–4695.
- **FastRL** — Li et al., *Optimizing Language Models with Fair and Stable Reward Composition in Reinforcement Learning*, EMNLP 2024, pp. 10122–10140.

**Companion document.** Read alongside [CRITICAL_REVIEW_v4.md](../CRITICAL_REVIEW_v4.md), which logs the v4 audit (the +0.11 pp null result on the SVM/DRM path) and lists the bugs B1/A2/B2/C1 that were fixed afterwards. This report extends v4 with paper-level alignment and a phased adoption roadmap; it does **not** re-derive the bug list.

---

## 1. Empirical recap — PPO is healthy, training duration is the problem

Per-checkpoint greedy SB-Bench accuracy (binary reward, Qwen2.5-VL-3B, max_train=2000, lr=1e-4, kl_beta=0.1):

| Checkpoint | SB-Bench accuracy | Notes |
| --- | --- | --- |
| Vanilla (no PPO) | 0.6190 | argmax over {A,B,C} |
| ep1 — 24% (~step 60) | **0.9472** | **peak** |
| ep1 — 50% (~step 125) | 0.7006 | first sign of overshoot |
| ep1 — 74% (~step 185) | 0.4633 | collapse |
| ep1 — end (~step 250) | 0.4472 | degraded below vanilla |

**Diagnosis.** PPO machinery is correct. Binary ±1 reward is informative enough to drive a +33 pt jump in <60 steps. The collapse is the classic RLHF over-optimization pattern: KL drifts past the sustainable region, the value head and the policy stop being mutually consistent, and the policy begins exploiting the reward in ways that hurt held-out accuracy. The fact that we observe this on a *2000-sample* subset means it will be even sharper on the full 11,662-sample train set — the policy reaches the cliff later in absolute steps but with more momentum.

**Implication for the rest of this report.** Any method that adds extra reward signal (DRM heads, FastRL composition) or extra loss terms (CAA) on top of the same un-controlled KL drift will just collapse faster. KL stabilisation is the foundation, not an afterthought.

---

## 2. Collapse-mitigation playbook (apply *before* enabling DRM/CAA/FastRL)

These are independent of the reward source. Apply them to the binary baseline first, confirm that the peak (0.9472) is *sustained* rather than just *visited*, then layer on the new mechanisms.

### 2.1 Adaptive KL controller (Schulman PPO recipe)
Replace the fixed `kl_beta=0.1` with an adaptive controller targeting a fixed per-token KL budget (Ouyang et al. 2022; both CAA and FastRL keep the standard token-level KL penalty):

```python
# after each PPO update step
kl_obs = token_kl.detach().mean().item()
target = 0.02                                # per-token nats; typical RLHF range
err = kl_obs / target - 1.0
self.kl_beta *= float(np.clip(1.0 + 0.1 * err, 0.5, 2.0))
self.kl_beta = float(np.clip(self.kl_beta, 0.05, 5.0))
```

This is exactly the mechanism the canonical PPO-RLHF papers use; CAA's eq. (2) and FastRL's eq. (7) both write `r_final = r − β · KL(πθ ‖ πref)` and assume β is hand-tuned per task. Adaptive control removes that tuning.

### 2.2 Learning-rate schedule
`lr=1e-4` (LoRA) is fine for the first 50–80 steps but is the proximate cause of the collapse later. Use cosine decay to `2e-5` over the full run, or linear decay with `min_lr=1e-5`. Pair with the value-head `lr=5e-4` decayed by the same factor.

### 2.3 Dense checkpoints + early stopping in the first epoch
Save every 10–15 steps for steps 30–150, then every 25 steps. Run greedy SB-Bench eval on a 256-sample held-out subset on each save; early-stop when the held-out metric drops for two consecutive checkpoints.

### 2.4 Value-loss clipping
The current trainer does not clip value updates. Add:

```python
v_pred_clipped = v_old + (v_pred - v_old).clamp(-eps_v, eps_v)
vf_loss = 0.5 * torch.max((v_pred - returns)**2, (v_pred_clipped - returns)**2)
```
with `eps_v ≈ 0.2`. This is in the original Schulman PPO and prevents the value head from chasing transient reward spikes (a known precursor to policy collapse).

### 2.5 KL-tripwire rollback
If `token_kl.mean() > 5 × target` for two consecutive steps, restore the last-saved checkpoint and halve the LR. This is a poor-person's trust region, but cheap and very effective for the kind of cliff we are seeing.

**Acceptance gate before moving to §3:** A binary-PPO run that holds ≥0.90 SB-Bench accuracy for ≥3 consecutive checkpoints across ≥150 PPO steps.

---

## 3. DRM — alignment with Luo et al. (arXiv:2502.13131)

### 3.1 What the paper actually does (Sections 3.1–3.4, Appendix A.3)
1. Take a pre-trained reward-model **or** any LM/IT-LM as feature extractor `φ(x, y) ∈ R^d` (penultimate hidden state). The paper uses `Gemma-2B-RM` (d=2048) and `Llama-3-8B-RM` (d=4096) as `φ`.
2. Build the embedding-difference dataset `z_i = φ(x, y_chosen_i) − φ(x, y_rejected_i)`.
3. Run PCA on `{z_i}` to obtain orthonormal basis `W = [w_1, …, w_d]`.
4. Each `(w_j, φ)` is a reward model: `r_j(x, y) = w_j^T φ(x, y)`. Because PCA signs are arbitrary and preferences are directional, both `+w_j` and `−w_j` are kept (Section 3.2 "Discussion"). The released model uses 50 components ⇒ 100 heads.
5. **Use-case in the paper: test-time adaptation, not RLHF training.** Given a small adaptation set, fit coefficients `k_j` and use `w' = Σ k_j w_j` as a single reward for either ranking or downstream alignment. No PPO experiment is run with DRM heads as in-the-loop rewards (Section 4 + Appendix A.1).

### 3.2 What our codebase does
- `src/modules/embeddings/generate_drm_heads.py`: PCA on `(h_chosen − h_rejected)` differences ✅, emits both ±w ✅, stores per-head `.pth` files ✅.
- `src/modules/training/drm_loader.py`: stacks to `(num_heads, hidden_dim)` ✅.
- `src/modules/rl_components/custom_vlm_ppo_trainer.py` (SVM mode, lines ~515–595): scores `r_k = h_active @ heads.T` per token, aggregates with FastRL.
- `src/modules/evaluation/evaluate_drm_heads.py`: per-head chosen>rejected accuracy on a held-out set ✅.

### 3.3 Gaps and required fixes

| # | Gap | Severity | Fix |
| --- | --- | --- | --- |
| D1 | `φ` is the **policy LM's** hidden state, not a reward model's. The paper *can* use an IT-LM as `φ`, but only because that `φ` was *also* the embedding source for training the original RM. Using the policy's own evolving hidden state as both `φ` and the thing-being-updated creates a moving target. | High | Either freeze `φ` to the *initial* (reference) policy hidden state, **or** train a small reward head on top of `φ` once before PPO and use that frozen `φ` throughout. Paper's Appendix A.3 uses a frozen RM. |
| D2 | Each head's discriminative quality is unverified before being plugged into PPO. Paper's Figure 6/7 show only ~the first few components are sharply discriminative; the tail is mostly noise. | High | **Phase-0 gate**: each head must score chosen>rejected accuracy > 0.6 on a held-out 1k SB-Bench pair set, otherwise drop it. The existing `evaluate_drm_heads.py` already produces this metric — wire it as a hard filter before training. Expect the 2k head count to drop to 50–200 useful heads. |
| D3 | SVM heads are currently trained on the *full-answer EOS* embedding, but PPO scores at the *one-letter generated token*. Out-of-distribution at PPO time. | Critical | Re-extract embeddings at the same token position the PPO trainer queries (post-letter, before EOS). Even better: extract at every candidate-letter position and average, matching the FirstTokenAllowlistProcessor's actual decision moment. |
| D4 | DRM in the paper is **not** used as a per-step PPO reward signal. Plugging 50–200 noisy reward heads into a token-level PPO loop is a novel extension and is what the FastRL composition is meant to manage. This is fine, but it means "the paper validated this" is no longer the safety net — Phase-0 validation (D2) becomes load-bearing. | Informational | Treat the SVM/DRM PPO path as a research extension, not a paper-replication. |
| D5 | The SVM path was disabled during the Phase-1 fix (token-KL was removed) and never re-enabled or re-tested. | High | Re-enable `dense_rewards = (dense_rewards − kl_beta * token_kl) * loss_mask` in the SVM branch identically to the binary branch (line ~513). |

---

## 4. CAA — alignment with Xia et al. (NAACL 2024)

### 4.1 What the paper actually does (Section 3.3, eq. 1–3 and Figure 3)
1. Generate two outputs for the same prompt:
   - `y_init` from the **frozen initial** LLM `L_θ_init`,
   - `y` from the **current PPO-trained** LLM `L_θ`.
2. Score both with the **same reward model R**: `r_init = R(y_init)`, `r = R(y)`.
3. Interventional feedback (Eq. 1): `w = |r_init − r|`, then **min-max normalise across the batch** to get `ŵ ∈ [0, 1]`.
4. **Multiply** the standard PPO clipped surrogate loss by `ŵ`: `L̂(θ) = ŵ · L(θ)`. That is the entirety of the CAA modification. Reward is the *standard* `r_t,final = r_t − β·KL` (Eq. 2); KL is unchanged.

The semantic intent: down-weight samples where the intervention (the PPO update) did not actually change the reward, because those samples carry no causal information about what the policy gradient is doing.

### 4.2 What our codebase does (`src/modules/rl_components/caa_feedback.py`)
- `compute_causal_reward_penalty`: `penalty = −λ_causal · max(0, ‖h_active − h_ref‖₂ − δ_margin)`. **Historically** distributed per-token and added to the reward; **as of the post-v4 fix** this term has been pulled out of the main PPO loss because empirically it was suppressing the gradient signal and producing the +0.11 pp null result documented in CRITICAL_REVIEW_v4.md (flaw D-causal-dead-zone: with δ_margin=1.0 the hinge almost never activates, and when it does it cancels useful policy moves). Confirmed by user, May 2026.
- `compute_dispersive_loss`: pairwise Gaussian RBF over hidden states, added to the **loss**. Independent regulariser, kept on.

**Net state today:** the file still defines `compute_causal_reward_penalty`, but it is no longer wired into the PPO reward path. This is the *correct* short-term action — it stops the bleeding — but it also means the codebase currently has **no CAA mechanism at all**, despite the name. The right next step is not to re-wire the L2 hinge but to implement the paper's actual algorithm (§4.4).

### 4.3 Gap

**This is the largest implementation/paper divergence in the codebase.** The historical "CAA" (now disabled, see §4.2) is *not* the CAA from Xia et al. It was a hidden-state-drift hinge regulariser. The two mechanisms differ in:

| Axis | Paper | Our (now-disabled) code |
| --- | --- | --- |
| Signal source | Reward-model score difference `\|R(y_init) − R(y)\|` | Hidden-state L2 drift `‖h_active − h_ref‖₂` |
| Requires? | Frozen initial policy + rollout from it + reward call | Just LoRA disable_adapter, no extra rollout |
| Applied to | PPO loss as **multiplicative sample weight** | Reward as **additive penalty** (and a dispersive aux loss) |
| Sign | Always non-negative weight ∈ [0,1] | Negative penalty (discouraging drift) |
| Effect | Discards samples where the policy is *not yet differentiated* from init | Discouraged the policy from *being differentiated* at all → empirically suppressed learning, which is why it was removed |

These are not the same thing. The historical implementation was closer to a (badly-tuned) representation trust region than to causal-IV alignment, and its removal from the reward path is consistent with that diagnosis.

### 4.4 Required fix
Replace the current CAA module with the paper's algorithm:

```python
# At rollout: also generate from the frozen initial policy (LoRA-disabled) for the same prompts.
with model.disable_adapter():
    y_init = model.generate(prompt, **gen_kwargs)
r_init = reward_fn(prompt, y_init)         # same reward used by PPO
r      = reward_fn(prompt, y)              # already computed
w_raw  = (r_init - r).abs()                # per-sample scalar
w_hat  = (w_raw - w_raw.min()) / (w_raw.max() - w_raw.min() + 1e-8)

# Standard PPO loss, then:
loss = (w_hat.detach() * per_sample_ppo_loss).mean()
```

Two concrete extras worth keeping from our current code but moved off the "CAA" label:
- The dispersive RBF loss is an independent, defensible regulariser. Keep it as `dispersive_loss` (separate flag), not as part of CAA.
- The hidden-state drift hinge is a poor proxy for what CAA does but a reasonable trust-region surrogate. If we want to keep it, rename it `representation_trust_region_penalty` and document it as our own choice.

For the **binary reward path** the CAA paper's mechanism is trivially applicable: rollout one extra sequence from the LoRA-disabled model, score it ±1 the same way, take `|r_init − r|`, normalise, weight the loss. Negligible compute cost (one extra forward+generate per batch). This is the cleanest first experiment to validate the CAA mechanism honestly.

---

## 5. FastRL — alignment with Li et al. (EMNLP 2024)

### 5.1 What the paper actually does (Section 3.2, Algorithm 1)
Mirror-descent estimate for weights on the simplex:

```
w_i^cur = w_i^pre · exp(−λ · r_i)  /  Σ_j w_j^pre · exp(−λ · r_j)
```
with `w_i` initialised at `1/n` and **multiplicative** updates that *preserve history*. Optional smoothing & bias:
```
w_i^cur = w_i^pre · exp(−λ · smooth(r_i) + b_i) / Z
```
Composite: `r_com = Σ w_i^cur · r_i`. Then standard KL-penalised PPO. The negative sign in the exponent assigns *higher* weight to *lower*-performing rewards — fairness across reward dimensions.

### 5.2 What our codebase does (`src/modules/rl_components/fast_rl.py`)
```python
alpha = softmax( max(0, -r_norm.mean(0)) / tau )    # tau=1.0
# EMA with decay 0.99 toward this alpha
```

### 5.3 Gap
Both are deficit-driven multi-head softmax weightings, but structurally different:

| Axis | Paper | Our code |
| --- | --- | --- |
| Update rule | Multiplicative `w · exp(−λr)/Z` (true mirror descent on simplex) | Recompute-from-scratch `softmax(max(0,−r_norm)/τ)` then EMA(0.99) |
| Input to exp | Raw rewards `r_i` | Clamped-negative-only normalised rewards |
| History | Carried implicitly by `w_pre` | Carried by separate EMA (a hyperparameter that the paper does not have) |
| Hyper-params | `λ`, optional `b_i`, optional smooth() | `τ`, `ema_decay` |

### 5.4 Required fix
Replace the current update with the paper's exact rule (one line change):

```python
# r_i: mean reward of head i over the current batch, shape (num_heads,)
unnorm = w_prev * torch.exp(-self.lam * r_i)   # default lam=1.0
w_new = unnorm / unnorm.sum().clamp_min(1e-12)
self.w_prev = w_new.detach()
r_com = (w_new * r_per_head).sum(dim=-1)       # composite per-token reward
```
Drop the separate EMA — the multiplicative update *is* the history. Drop the `max(0, −r)` clamp — the paper does not have it; the exp handles sign symmetrically and that is precisely what makes the weights converge instead of oscillating. Keep `tau`/`lam` as a tunable temperature.

Anneal `lam` from low (≈0.1, near-uniform) to higher (≈1.0, focused) over training — this is a stability trick consistent with the paper's "stability over varying numbers of rewards" experiments (their Figure 3 shows their method dominates only when n ≥ 3 heads, and uses up to 4 heads; we are proposing 50–200, so annealing is essential to avoid early winner-take-all collapse).

---

## 6. Risks and open questions

1. **Dominant first-component reward hacking.** PCA's first eigenvector concentrates the most variance and so produces the strongest reward signal. Under FastRL, a successful first-component policy gets *down*-weighted, but the corresponding head still dominates the gradient when it has high magnitude. Watch for the case where 1–2 heads have ‖w_j‖·‖φ‖ orders of magnitude larger than the rest. Standardise heads by `r_j ← r_j / std(r_j over a calibration batch)` before composition.
2. **Sign ambiguity going the wrong way.** PCA produces `±w_j`. The paper's evaluation (Figure 6/7) shows that the +w and −w halves of the same component have systematically different reward-bench scores. Our `evaluate_drm_heads.py` already produces per-sign accuracy — *use it to drop signs that score below 0.5*, not just to log them.
3. **Vision-modality coverage.** The DRM, CAA, and FastRL papers are all text-only. The decomposition we run is on the LM hidden state *after* the vision encoder, which means the decomposition basis is anchored in the joint vision-text representation. There is no guarantee a PCA component corresponds to "bias along a visual attribute" rather than a text-only artefact. Phase-0 should include per-category chosen>rejected on SB-Bench (the existing `evaluate_drm_heads.py` does this) and any category where *no* head clears 0.6 should trigger a re-extraction with a different `φ`.
4. **δ-margin sensitivity in the kept dispersive/trust-region loss.** Embedding norms grow during PPO; a fixed margin will silently saturate. Track `‖h‖` mean over training and scale `δ_margin` with it, or use a relative threshold.
5. **One-letter generation is a poor reward signal granularity.** Both the CAA paper and FastRL operate on multi-token generations where dense intra-sequence rewards matter. Our 1-letter setting makes every per-token mechanism degenerate. Either (a) accept that CAA/FastRL operate at the *sample* level here (which is what the paper does anyway), or (b) lengthen generation to include rationale tokens before the letter, restoring the multi-token reward profile.

---

## 7. Recommended next experiments (phased)

Each phase has a single, falsifiable success criterion. Do not start phase `k+1` until phase `k` passes.

| Phase | What | Success criterion |
| --- | --- | --- |
| **0** | Re-run binary PPO with §2 fixes (adaptive KL, LR decay, dense ckpts, value clip, KL tripwire) on max_train=2000. | Held-out SB-Bench ≥ 0.90 sustained over ≥150 steps with no >5pt drop. |
| **1** | Same as Phase 0 on **full** train set (11,662 samples, ≥2 epochs). | Same criterion holds; final-checkpoint accuracy ≥ 0.90. |
| **2** | Rebuild DRM heads with frozen `φ` at the post-letter token position. Run `evaluate_drm_heads.py` Phase-0 gate; keep only heads with chosen>rejected > 0.6 *per category*. | ≥ 30 heads survive the gate, each SB-Bench category covered by ≥ 1 head. |
| **3** | Enable SVM/DRM mode with **paper-correct FastRL** composition (§5.4), no CAA. KL anchoring from Phase 1 still active. | Held-out SB-Bench ≥ Phase-1 result (no regression from added complexity). |
| **4** | Add **paper-correct CAA** (§4.4) on top: one extra rollout from LoRA-disabled model per batch, `\|r_init − r\|`-weighted PPO loss. | Held-out SB-Bench improves by ≥ 1pt over Phase 3, **per-category min** improves by ≥ 2pt. |
| **5** | Ablation: turn off each component (DRM, FastRL, CAA) one at a time and confirm each contributes. | Each ablation drops at least one metric. |

If any phase fails, the prior phase's checkpoint and hparams are the production baseline.

---

## 8. Specific code locations that need to change

| File | Change | Phase |
| --- | --- | --- |
| `src/modules/rl_components/custom_vlm_ppo_trainer.py` ~line 513 | Add adaptive-KL update after each step; expose `target_kl`. | 0 |
| same, SVM branch ~line 530 | Re-enable token-KL subtraction symmetrically with binary branch. | 3 |
| same, both branches | Add value-loss clipping. | 0 |
| same, both branches | Add KL-tripwire rollback hook. | 0 |
| `src/run_modal.py` | Expose `--target-kl`, `--lr-schedule {cosine,linear}`, `--min-lr`, `--ckpt-every`, `--eval-every`. | 0 |
| `src/modules/embeddings/generate_drm_heads.py` | Add `--phi-source {policy,reward_model,frozen_init}` and token-position selector. | 2 |
| `src/modules/evaluation/evaluate_drm_heads.py` | Add hard-filter mode that writes a `kept_heads.json` consumed by `drm_loader.py`. | 2 |
| `src/modules/rl_components/fast_rl.py` | Replace update rule with multiplicative mirror-descent (§5.4); drop EMA. | 3 |
| `src/modules/rl_components/caa_feedback.py` | Add new `compute_interventional_weight(r_init, r)` returning min-max-normalised `\|r_init − r\|`. Keep the existing dispersive loss but rename and document it. Leave `compute_causal_reward_penalty` undeleted but unwired (already the state today per v4 fix). | 4 |
| `src/modules/rl_components/custom_vlm_ppo_trainer.py` | At rollout time, generate one extra sequence under `model.disable_adapter()`, score it, compute `w_hat`, multiply into per-sample PPO loss. | 4 |

---

## 9. What I have *not* verified and you should check yourself

- Whether `evaluate_drm_heads.py` currently filters by category (skim suggests it logs per-category but does not gate).
- Whether the reward model used to score `y_init` vs `y` in the CAA fix should be the SVM/DRM composite reward or the binary letter reward. The paper assumes a single `R`. For the SVM path the natural choice is the FastRL composite `r_com`; for the binary path the ±1 letter reward.
- The exact embedding token position used by the current `generate_drm_heads.py`. Confirm against the position used at PPO scoring time (the FirstTokenAllowlistProcessor's first-generated token). If they differ, D3 is the single biggest blocker to the DRM path producing any useful signal.
- DRM paper Appendix A.3 says they used 50 components × 2 signs = 100 heads. Our current code emits 2k. Consider starting closer to the paper's regime (top-50 by eigenvalue, both signs) once Phase-0 gating is in place.
