# Phase 0.8 — Multi-Layer DRM: Where does bias actually live?

**Branch:** `phase0.8_multilayer_drm`
**Status:** scaffolding
**Prereq:** [Phase 0.7 §8 T1.1 results](Phase0.7.md) — reward-side pathology confirmed.

---

## 0. Motivation

T1.1 (Phase 0.7 §8) proved that PCA and SVM reward heads, both built on the **penultimate** hidden state of Qwen2.5-VL-3B, prefer "C" (unknown) over the correct named option on disambig items — *before* any PPO. The reward objective and the bias-encoding location are conflated:

- The penultimate state is task-conditioned (it must encode "which letter to output"). That competes with bias-content representation for the same dimensions.
- The probing literature (Tenney, Hewitt & Manning, Belinkov, Schwettmann et al.) repeatedly shows that demographic / social features peak at **middle** transformer layers, not final ones.
- The vision-language fusion in Qwen2.5-VL adds a further hypothesis: bias signal may live in cross-modal fusion layers (LM layers ~12–24) before being compressed by late-layer answer-formatting circuits.

**Hypothesis (to be tested, not assumed):** there exists a layer L* < penultimate such that DRM-style heads built on layer L* exhibit lower C-bias and higher good-C-vs-bad-C confusion than the penultimate heads.

## 1. Acceptance gates

| Gate | Criterion | Decides |
|---|---|---|
| **G-A1** | Linear probe accuracy for `(condition, correctness)` reported across **all** layers of LM + vision encoder, with bootstrap CIs. No layer pre-selected. | Whether a "bias-dominant" layer exists at all. |
| **G-A2** | SVM and PCA heads rebuilt on top-3 candidate layers from G-A1; T1.1 protocol re-applied; C-bias and confusion reported per layer. | Whether better probe accuracy translates to better DRM reward. |
| **G-A3** *(if A1+A2 pass)* | At least one layer L* satisfies `C-bias ≤ +0.5` AND `confusion ≥ +1.0` on a held-out VLBiasBench split. | Whether to proceed to Tier-B head reconstruction. |

**Eval hygiene:** all layer-selection decisions use VLBiasBench `qformat=text` (currently unused) as the **selection holdout**. Final reward eval uses `qformat ∈ {base, scene, scene_text}`. No selection criterion ever sees the final-eval split.

## 2. Bias-blind layer discovery — methodology

The user's constraint: **layers must not be cherry-picked from A1/A2 outputs.** Concretely:

1. **A1 covers every layer.** No layer is excluded a priori. Report the *full* per-layer curve.
2. **A1 metric is task-agnostic.** Probe target is `(condition, correctness)` over base-policy responses, derived from VLBiasBench labels — **not** from any DRM head output. Avoids the eval-leak that broke T1.2.
3. **Layer ranking is pre-registered.** Before running A2, write down the top-3 layers by A1 probe-accuracy on the **`qformat=text` holdout split**, *not* on the full set. This is the bias-blind selection step.
4. **A2 tests transfer.** Heads built on the pre-registered top-3 are scored on the full VLBiasBench (n=990 stratified). If C-bias does not drop monotonically across the ranking, the A1 metric is not predictive of DRM quality and we abandon the layer-probe direction.
5. **Negative result is publishable.** If no layer L* exists that satisfies G-A3, that itself is a useful finding — it means the bias representation is irreducibly entangled with the answer circuit in this architecture, and Tier-B/C must take a different shape (contrastive heads, supervised reward, etc.).

## 3. Tier-A experiments

### A1 — Per-layer probe accuracy (read-only, ~2 GPU-hrs)

- **Input:** the 5 existing `*_vlbias_gen.jsonl` from Phase 0.7 (covers all variants).
- **Forward pass:** Qwen2.5-VL-3B with `output_hidden_states=True`. Capture hidden state at the `post_letter` position from every **LM-decoder** layer (embeddings + 36 transformer blocks = 37 layers for the 3B model).
  - **Vision-encoder caveat:** the `post_letter` position lives in the text token stream; vision-encoder layers operate on image patches and have no comparable per-record probe target. A1 therefore probes LM-decoder layers only. Vision-side bias probing is a separate Tier-B experiment.
- **Probes:** three binary logistic regressions per layer, trained with 5-fold CV on the **base** variant only:
  - P1: `condition ∈ {ambig, disambig}` — sanity check (should be high everywhere)
  - P2: `correctness ∈ {correct, incorrect}` on disambig items only
  - P3: `is_good_C` — among model-emitted-C records, `gold = C` (good) vs `gold = named` (bad). **This is the decisive probe.**
- **Holdout:** P3 trained on `qformat ∈ {base, scene, scene_text}` excluding 100 random `qformat=text` records reserved for layer-selection.
- **Output:** `Phase0.8/probe_results/layerwise_probe.json` with `{layer_idx, layer_type, p1_acc, p2_acc, p3_acc, p3_acc_holdout}`. Also a PNG curve.

### A2 — Per-layer DRM head rebuild + T1.1 re-score (~4 GPU-hrs)

- **Pre-registration step:** open `Phase0.8/PREREG.md`, write down the top-3 layers from A1 ranked by `p3_acc_holdout`. Commit. Then proceed.
- **Head build:** rerun the SB-Bench head-extraction pipeline at each of the 3 layers (SVM + PCA, same hyperparameters as Phase 0.6 D3).
- **Score:** apply the existing [score_vlbias_offline.py](src/modules/evaluation/score_vlbias_offline.py) at each layer. Need to add `--layer_idx` argument to `create_custom_forward` to expose intermediate-layer extraction.
- **Output:** Phase 0.7 §8.1-style table per layer.

### A3 — Causal validation *(only if A2 shows a candidate L\*)*

- **Activation patching:** for 100 disambig items, patch the L*-layer hidden state at the answer-token position from a known `disambig::correct` example into a `disambig::incorrect` example. Measure C→named flip rate.
- **Passes if** flip rate > 30% (clean causal effect) — confirms L* carries the decision, not just a correlation.

## 4. Tier-B / Tier-C (conditional)

Held until A1+A2+A3 close. See [Phase0.7_Remediation_Plan.md](Phase0.7_Remediation_Plan.md) §"Tier B" for the contrastive-head and multi-layer aggregation options.

## 5. Out of scope for Phase 0.8

- New PPO runs (no policy training until reward is fixed).
- Architectural changes to the policy (CAA, frozen-φ, etc.).
- New benchmarks (PAIRS, FairFace-VQA) — those are Phase 0.9 verification.

## 6. File layout

```
Phase0.8.md                              ← this doc
Phase0.8/
  PREREG.md                              ← pre-registered top-3 layers (created after A1)
  probe_results/
    layerwise_probe.json
    layerwise_probe.png
  drm_per_layer/
    layer{L}__svm__offline_reward.json
    layer{L}__pca__offline_reward.json
scripts/
  phase08_a1_layer_probe.sh
  phase08_a2_per_layer_drm.sh
src/modules/evaluation/
  probe_layers.py                        ← new (A1)
```

## 7. Running log

*(append as experiments complete)*
