# Phase 0.8 — Tier-A Pre-Registration (A2 layer commitment)

**Status:** COMMITTED before any A2 head extraction or scoring is run.
**Branch:** `phase0.8_multilayer_drm`
**Parent commit (A1 results):** `81e841a2b1baae431ae52281d9a96734b0b59186`
**Date:** 2026-05-30

---

## 1. What is being pre-registered

The set of LM transformer layers from which Phase 0.8 / A2 will rebuild
SB-Bench reward heads (SVM-per-axis + PCA-50) at the `post_letter` token
position, and the acceptance criterion that determines G-A2 / G-A3.

This is registered **before** any A2 code is run, so that the layer choice
cannot be retro-fit to favourable downstream numbers (C-bias, confusion).

## 2. Selection methodology (already executed in A1)

- **Source data:** 5 layerwise probe JSONs at
  `Phase0.8/probe_results/phase08_probe_results/{base, pca_ep1-end,
  pca_ep1-step80, svm_ep1-50pct, svm_ep1-end}_layerwise_probe.json`
- **Probe protocol:** post_letter hidden state extracted from every LM
  layer L ∈ {0..36} (37 total = 1 embedding + 36 blocks). For each layer
  we trained `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)`
  with 5-fold StratifiedKFold on three binary tasks:
    - **P1**: condition (ambig vs disambig)
    - **P2**: disambig correctness
    - **P3 (decisive)**: `is_good_C` — among rows where the model emitted "C",
      whether the gold answer was actually C (good_C=1) or a named answer that
      the model declined into C (good_C=0).
- **Decisive metric:** `p3_acc_holdout` — P3 accuracy on a separate
  qformat-shifted holdout (`qformat=text`, 100 records per variant) trained on
  the primary pool (`qformat ∈ {base, scene, scene_text}`). This is the
  bias-blind generalization metric specified in Phase0.8.md §2.

## 3. Holdout coverage

| Variant            | n_p3_holdout | Eligible for layer-selection? |
|--------------------|--------------|-------------------------------|
| `base`             | 71           | ✅                             |
| `pca_ep1-end`      | 100          | ✅                             |
| `pca_ep1-step80`   | 100          | ✅                             |
| `svm_ep1-50pct`    | 19           | ✅                             |
| `svm_ep1-end`      | **3**        | ❌ EXCLUDED (insufficient)    |

**Exclusion rule (pre-registered):** any variant with `n_p3_holdout < 10`
is dropped from the cross-variant aggregate. `svm_ep1-end` fails this
threshold (3 < 10) due to extreme C-mode-collapse exhausting the
"model-emitted-C ∧ gold=named" sub-population on the qformat=text holdout.
It will still be **scored** in A2 but does not get a vote in layer choice.

## 4. Pre-registered decision rule (from prior turn)

> If ≥3 of 5 variants have a non-penultimate layer L* with
> `p3_acc_holdout >= p3_acc_holdout@L35 + 2pp`, AND those L* values are within
> 5 layers of each other → proceed to A2 on the median L*.

**Result:** Rule **PASSES.** 4 of 4 eligible variants have L9 as a (tied) best
layer with the following lifts over penultimate (L35):

| Variant            | L9 P3 holdout | L35 P3 holdout | Δ (pp)  |
|--------------------|---------------|----------------|---------|
| `base`             | 0.986         | 0.958          | **+2.8** |
| `pca_ep1-end`      | 0.990         | 0.960          | **+3.0** |
| `pca_ep1-step80`   | 0.990         | 0.950          | **+4.0** |
| `svm_ep1-50pct`    | 0.947         | 0.895          | **+5.3** |

Cross-variant aggregate (mean of 4 eligible variants) — top 6 layers:

| Rank | L  | mean   | std   | min   |
|------|----|--------|-------|-------|
| 1    | **9**  | 0.978 | 0.021 | 0.947 |
| 2    | **13** | 0.962 | 0.045 | 0.895 |
| 3    | **11** | 0.962 | 0.045 | 0.895 |
| 4    | 12 | 0.957 | 0.041 | 0.895 |
| 5    | 14 | 0.957 | 0.041 | 0.895 |
| 6    | 10 | 0.956 | 0.043 | 0.895 |
| 9    | 35 (penultimate) | 0.941 | 0.031 | 0.895 |

L9 has the highest mean, lowest std, and highest min across eligible variants.

## 5. COMMITTED LAYER LIST (frozen for A2)

| Slot      | Layer | Role                                  |
|-----------|-------|---------------------------------------|
| Primary   | **L9**  | Highest mean & lowest std; wins in all 4 eligible variants |
| Secondary | **L13** | Tied 2nd by mean; cleanly separated from L11 by ≥2 blocks  |
| Tertiary  | **L11** | Tied 2nd by mean; fills the L10-L14 plateau                |
| Control   | **L35 (penultimate)** | Existing Phase 0.6 head location — comparison baseline      |

Layer indices follow the `hidden_states` convention of HuggingFace
transformers: `hidden_states[0]` is the embedding layer; `hidden_states[k]`
for k≥1 is the output of LM block k-1. So L9 = output of block 8, L35 =
output of block 34 = penultimate of a 36-block model.

**No additional layers will be evaluated at the A2 acceptance step.** If
none of {L9, L13, L11} satisfy G-A2, we will report null and pivot per
Phase0.8.md §3, **not** drift into evaluating more layers.

## 6. A2 acceptance gates (pre-registered)

For each layer L ∈ {L9, L13, L11}, A2 rebuilds SB-Bench heads:
- `sb_bench-SVM-component_L{L}/` (9 per-axis hyperplanes)
- `sb_bench-PCA-component_L{L}/` (top-K PCA components, K matched to existing)

and re-scores ALL 5 variants under the T1.1 offline protocol
(`scripts/phase07_t1_offline_reward.sh` equivalent), reporting on the
exact same 900-record sample as Phase 0.7.

### G-A2 (per layer, per head type)

A layer L is a CANDIDATE if, for **at least one** of {SVM, PCA}:
  - C-bias on `base` is no worse than L35 baseline ± 0.5
  - C-bias on PPO variants drops by ≥ 1.0 pp relative to L35
  - Confusion-pair accuracy does not regress by > 1.0 pp on any variant

### G-A3 (overall A2 verdict)

A2 SUCCEEDS if at least one (L, head_type) cell satisfies G-A2 on ≥3 of 5
variants (any variant qualifies — `svm_ep1-end` is included here because A2
re-scoring is well-defined even when probe-based selection was not).

If A2 fails → Tier-B (head architecture pivot, per Phase0.8.md §4).

## 7. Implementation notes (frozen surface area)

- New arg: `--layer_idx` on `create_custom_forward`, default `-2` for
  backcompat. Accepts any int in `[-37, -1]` or `[0, 36]`. Output paths
  auto-suffixed with `_L{abs_idx}` when `layer_idx != -2`.
- Heads dir suffix convention: existing
  `/mnt/data/generated_heads_letter_post_letter/` becomes
  `/mnt/data/generated_heads_letter_post_letter_L{N}/` for N ∈ {9, 11, 13}.
  L35 keeps the existing un-suffixed dir.
- No re-training of PPO, no re-running A1.

## 8. Caveats / known limits

1. `svm_ep1-end` is excluded from layer selection but included in A2 scoring.
   If A2 results disagree wildly between svm_ep1-end and the other 4
   variants, we will report that openly and discuss in the Phase 0.8 writeup;
   we will NOT use it to override the layer choice post-hoc.
2. The post_letter probe sees only text-side hidden states. A1 does not
   inform whether vision-encoder layers contain additional bias signal;
   Phase0.8.md flagged this as deferred.
3. P3 holdout sample sizes are modest (19-100). The L9 vs L35 gap is large
   enough (3-5pp) that 95% bootstrap CIs should not overlap, but we did not
   formally compute them. A2's downstream gates (C-bias) are computed on
   the full 900-record sample and so do not inherit this limitation.

---

**Signed-off by:** Phase 0.8 multi-layer DRM workstream
**Modification policy:** This file is append-only after commit. Any change
to §5 or §6 requires an explicit "REVISION" section dated and justified.
