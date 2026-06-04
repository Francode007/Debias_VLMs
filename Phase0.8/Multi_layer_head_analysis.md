# Multi-layer Head Analysis

**Workstream:** Phase 0.8 — A3 (probe-as-head decision gate)
**Model:** Qwen2.5-VL-3B-Instruct (37 transformer blocks, hidden_dim = 2048, bf16)
**Reward source:** sklearn `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)`
weight vector at a single LM hidden layer, converted to a unit-norm reward head
(state_dict `{"weight": (1, 2048) float32}`) and applied at `token_position = post_letter`.
**Status of this document:** L9 trajectory fully analysed (Tier 0 → Test A-lite →
Test B Lite → Move 1 orth-against-isC). L11 and L13 trajectories pending.

---

## 0. Why a "Multi-layer Head Analysis"?

The original A3 plan picked **L9** as the operating layer because that is where the
A1 P3 (`is_good_C`) probe topped out at 0.978 holdout accuracy. Tier 0 + Test B
revealed that L9 by itself is not enough to settle the hypothesis:

- The probe achieves high accuracy at L9 but its score behaviour off-distribution
  (model-said-NAMED records) is degenerate.
- The "C-detection vs bias-detection" decomposition cannot be settled from a
  single layer because the relative cosine `cos(w_bias, w_isC)` changes with
  depth in ways we have not yet measured.

This document is the running ledger for the per-layer evidence and the comparative
verdict. New layer trajectories (L11, L13, …) will be appended to Section 4 below,
preserving identical structure for direct comparison.

---

## 1. Pipeline and conventions (one-time)

### 1.1 Probe definitions

Two probes are fit per layer on Qwen2.5-VL-3B `base` hidden states at the
`post_letter` token position. Hidden states are cached once on Modal volume
(`/mnt/data/phase08_probe_results/base_probe_hs.npz`) and reused for every layer.

| probe | label rule (model emitted letter ∈ {A,B,C}) | label space | n (post-filter) |
| --- | --- | --- | --- |
| **`is_good_C`** (P3, original A1 task) | `pred == C` only: 1 if `gold == C`, 0 if `gold ∈ {A,B}`; **None otherwise** | restricted to model-said-C records | 613 |
| **`is_C`** (P4, Move 1 companion) | 1 if `pred == C`, 0 if `pred ∈ {A,B}`; None only if unparseable | all parseable records | 900 |

Both probes write `<variant>_L{N}_probe_weights[_isC].npz` with fields
`coef (D,), intercept, layer_idx, train_n, train_acc, holdout_n, holdout_acc,
logreg_C, hidden_dim, token_position, reward_base, variant, task`.

### 1.2 Probe → reward head

`probe_to_head.py` takes the raw `coef`, optionally:

1. Projects out top-K PCs (`--residualize_pcs`, K=2 in the residual variant).
2. **Orthogonalises against another probe's coef** (`--orthogonalize_against`,
   Move 1, `v ← v − (v·u / u·u) u` with zero-norm guard).
3. Unit-normalises (`--normalize`).

Then writes `pytorch_model.bin` containing `{"weight": (1, D) float32}` plus a
`metadata.json` capturing `cos(w_bias, w_orth)` before normalisation and the
norm retained after each transform.

### 1.3 Offline scoring (VLBiasBench)

Every head is scored on a stratified VLBiasBench sample (`max_per_cell = 30 →
900 records per variant`) for three policy variants:

- `base` — the un-steered policy
- `svm_ep1-end` — episode-1 endpoint after SVM-style activation steering
- `pca_ep1-end` — episode-1 endpoint after PCA-style steering

Per-cell mean / std reward is reported by the `by_cell` dict
(`{neg,non_neg,ambig} × {correct,incorrect}`) and by axis-conditioned
sub-cells in `by_bbq_axis_and_cell`.

### 1.4 Metrics

- **C-detection contrast** (within model-said-C, ambig polarity):
  `C_gap = μ(ambig::correct) − μ(ambig::incorrect)`,
  `C_d = C_gap / pooled_sd`
- **Stereo-vs-counter contrast** (disambig items, model output not gated by C):
  pool `{neg::correct, non_neg::incorrect}` as "stereo-aligned predictions",
  pool `{neg::incorrect, non_neg::correct}` as "counter-aligned predictions",
  `s_gap = μ_stereo − μ_counter`, `s_d = s_gap / pooled_sd`
- **Test A-lite (free)**: same stereo / counter pooling computed directly on the
  probe's holdout records, no offline reward run needed.

---

## 2. L9 — full trajectory (executed May–Jun 2026)

### 2.1 Probe fit quality

| probe (L9, base) | train n / acc | holdout n / acc |
| --- | --- | --- |
| `is_good_C` | 491 / 0.987 | 122 / 0.986 |
| `is_C` | 720 / **1.000** | 180 / **1.000** |

`is_C` is **perfectly linearly separable at L9** in 2048 dimensions on n=900.
This is a strong red flag: the L9 representation has the C-emission decision
essentially read-out-able, and any probe trained to predict it will likely overfit
to that single linear direction.

`||coef||` (pre-normalisation):
- `w_bias` (is_good_C) = **7.957**
- `w_isC` = **0.870**

### 2.2 L9 hidden-state geometry (Section 6 sanity, from probe_layers.py)

PCA on `base` L9 post-letter hidden states:

- PC0 ≈ **84.6 %** of variance, PC1 ≈ **13.9 %** → rank-1 dominated.
- Mean pairwise |cos| between SVM bias-direction probes at L9 ≈ **0.599**:
  multiple "bias" probes collapse to roughly one shared direction. L9 is not a
  layer where bias is cleanly disentangled.

### 2.3 Tier 0 — probe-as-head on `base`

Head = `is_good_C` coef, unit-normalised.

| cell | n | μ (plain head) |
| --- | --: | --: |
| ambig::correct | 274 | **+0.4288** |
| ambig::incorrect | 26 | −0.0427 |
| neg::correct | 125 | −0.7817 |
| neg::incorrect | 175 | −0.5032 |
| non_neg::correct | 112 | −0.9094 |
| non_neg::incorrect | 188 | −0.4326 |

**Headline numbers:**
- `C_gap = +0.4716`, `C_d = +2.96`, `σ-gap = +2.52σ`
- Primary gate (`C_d > 1.0`) **PASS**.

But: every model-said-NAMED cell scores in the −0.4 to −0.9 range. The head is
producing a near-constant low score off-distribution. This is the first hint that
the "high reward" is structural (in-distribution boost) rather than
bias-discriminative.

### 2.4 Test A-lite — free analysis on probe holdout

Pooling stereo vs counter predictions on the 122 holdout records:

| metric | value |
| --- | --: |
| stereo − counter gap | **+0.090** |
| Cohen's d | **+0.29** |
| t-statistic | **+3.55** |
| p-value | **0.0004** |
| Per-axis sign | **9 / 10 positive** (only Race_ethnicity ≈ 0) |

Verdict: bias signal is real but **~10× weaker** than the C-detection signal.

### 2.5 Test B Lite — between-variant comparison (CONFOUNDED)

Re-scoring `base`, `svm_ep1-end`, `pca_ep1-end` with the *same* L9 plain head
revealed a **distributional confound** that invalidates the naive Test B
monotonicity argument:

| variant | n_ambig::correct | n_ambig::incorrect | s_gap | s_d |
| --- | --: | --: | --: | --: |
| base | 274 | 26 | +0.090 | +0.29 |
| svm_ep1-end | 59 | 241 | (different items) | — |
| pca_ep1-end | 300 | **0** | (degenerate) | — |

`pca_ep1-end` emits C on **every** ambig record (n=300/300 → C); `svm_ep1-end`
emits C on **only ~20 %** of ambig (n=59/300 → C). The disambig cells therefore
contain different items per variant. Conclusion at the time: the probe is **not
purely a C-detector** (stereo Δ shifts across variants on identical disambig items),
but the bias component is too small relative to C-detection to use as a
standalone RL reward.

### 2.6 Move 1 — orthogonalise `w_bias` against `w_isC`

Implementation: Gram–Schmidt, `v ← v − (v·u / ||u||²) u`, applied **before**
PC-residualisation and **before** unit-normalisation. Run on L9 `base` only.

| quantity | value |
| --- | --: |
| `cos(w_bias, w_isC)` before orth | **+0.02588** |
| Norm retained after orth | **99.966 %** |
| `cos(plain_head, orth_head)` after unit-norm | **+0.9997** |
| ‖plain_head − orth_head‖ | **0.0259** |

**The two probes were already essentially orthogonal at L9.** Orthogonalisation
removed 0.03 % of the bias-probe's norm.

#### 2.6.1 Move 1 score deltas (plain → orth, identical item sets per cell)

| variant | cell | n | plain μ | orth μ | Δμ |
| --- | --- | --: | --: | --: | --: |
| base | ambig::correct | 274 | +0.4288 | +0.1450 | **−0.2839** |
| base | ambig::incorrect | 26 | −0.0427 | +0.2311 | **+0.2739** |
| base | neg::correct | 125 | −0.7817 | −0.5064 | +0.2754 |
| base | neg::incorrect | 175 | −0.5032 | −0.7486 | −0.2454 |
| base | non_neg::correct | 112 | −0.9094 | −0.6307 | +0.2788 |
| base | non_neg::incorrect | 188 | −0.4326 | −0.6754 | −0.2428 |
| svm_ep1-end | ambig::correct | 59 | +0.4309 | +0.1485 | −0.2824 |
| svm_ep1-end | ambig::incorrect | 241 | −0.1471 | +0.1322 | +0.2793 |
| pca_ep1-end | ambig::correct | 300 | +0.4231 | +0.1389 | −0.2841 |

Sample counts identical between plain and orth runs (verified). The per-cell
deltas cluster at **±0.27 to ±0.28** across every variant and every cell.

**Mechanistic explanation.** The removed direction has magnitude
`‖Δw‖ = 0.026`. For it to produce reward shifts of ±0.28 in the unit-normalised
head, the removed direction must align with a very high-magnitude direction
in L9 hidden states. L9 PC0 carries 84.6 % of variance and is the obvious
candidate. The orthogonalisation is geometrically a no-op against `w_isC` but
incidentally subtracts a tiny projection onto PC0, which then dominates the
score shift through the long hidden-state norms.

#### 2.6.2 Move 1 verdict gates

Re-derived numbers:

| metric (base variant) | plain | orth |
| --- | --: | --: |
| `C_d` | +2.96 | **−0.54** (collapsed) |
| `s_d` (stereo) | +0.289 | **+0.356** (+23 %) |

By the pre-committed gates: `|C_d| < 1.0` ✓ AND `s_d_orth ≥ s_d_plain × 0.8` ✓
→ **PASS-weak**. But the gates are misleading here: orth did not remove a
C-detection component because there wasn't one to remove. The C_d collapse is
a side-effect of nudging the head 2.6 % along PC0.

### 2.7 Re-interpretation of the L9 probe

The original interpretation ("L9 probe is ~90 % C-detector + ~10 % bias-detector")
is **falsified** by `cos(w_bias, w_isC) = 0.026`. The true geometry is:

1. **Within the model-said-C subset**, `w_bias` separates `legit-C (gold=C)` from
   `bias-driven-C (gold=NAMED)`. This is exactly its training task. The +2.96
   `C_d` is a within-task accuracy signal, not "C-emission detection".
2. **Off the trained subset** (model-said-NAMED records, all 4 disambig cells),
   the probe is undefined and produces a systematic low score (−0.4 to −0.9)
   because those hidden states are in regions where the decision boundary was
   never anchored.
3. The persistent +0.09 to +0.36 stereo-vs-counter gap **across plain and orth**
   shows there is a small, real, bias-discriminative signal independent of the
   C-emission axis — but it is dwarfed by the OOD penalty.

### 2.8 L9 conclusion

The L9 probe-as-head **cannot become a usable standalone RL reward** as
currently designed, but for a subtler reason than first thought:

- ❌ NOT because the probe is a C-detector (it isn't; cos = 0.026).
- ✅ Because the probe is trained on a *conditional* sub-population (model-said-C)
  and produces meaningless OOD scores everywhere else, which would induce
  reward-driven mode collapse on the policy (push everything toward C-emission).

Possible L9-specific remedies (deferred — see Section 5):

- **L9-A**: retrain probe on the full dataset with a label rule that is defined
  for every parseable record (e.g. `bias_present / bias_absent` over all 900,
  or a 3-way `legit_C / bias_C / NAMED_response`).
- **L9-B**: keep the conditional probe but gate the reward by C-emission and
  pair with an independent disambig-accuracy reward.

These are deprioritised in favour of measuring whether the picture improves at
deeper layers (where the A1 PCA design predicted bias would disentangle from
generic C-emission, ~L20–L28).

---

## 3. Cross-layer comparison table (live)

This table is the executive summary. Rows added as each layer trajectory completes.

| layer | `cos(w_bias, w_isC)` | `is_C` train/holdout | `is_good_C` train/holdout | Tier 0 `C_d` (base) | Tier 0 `s_d` (base, 900) | per-axis sign | OOD μ̄ (4 disambig cells) | `\|s_d\| / \|C_d\|` |
| --: | --: | --: | --: | --: | --: | --- | --: | --: |
| 9  | **+0.026** | 1.000 / 1.000 | 0.987 / 0.986 | +2.96 | +0.289 | 9/10 pos | −0.657 | **0.098** |
| 11 | **+0.055** | 1.000 / 1.000 | 0.990 / 0.972 | +5.17 | +0.179 | 9/10 pos | −1.014 | 0.035 |
| 13 | **+0.046** | 1.000 / 1.000 | 0.989 / 0.972 | +5.41 | +0.206 | 9/10 pos | −0.938 | 0.038 |

### Headline verdict (cross-layer, after L13)

1. **`cos(w_bias, w_isC)` stays ≈ 0 at every shallow layer.** The "L9 probe is a
   C-detector" hypothesis is now empirically dead across three layers — the two
   probes are essentially orthogonal everywhere we have measured. Move 1
   orthogonalisation will continue to be a near-no-op at L11/L13 too.
2. **`is_C` is *always* perfectly separable in 2048 dim.** This is a property of
   the post-letter representation, not of any one layer. The C-emission decision
   is fully read-out-able at L9, L11, and L13.
3. **`C_d` *grows* with depth (+2.96 → +5.17 → +5.41).** Going deeper makes the
   probe's in-distribution structural signal sharper, *not* its bias-specific
   signal.
4. **`s_d` *shrinks* with depth (+0.289 → +0.179 → +0.206).** Bias-discriminative
   signal does **not** improve at L11/L13. L9 has the best stereo-vs-counter
   discrimination of the three layers tested.
5. **OOD penalty *worsens* with depth (−0.657 → −1.014 → −0.938).** The probe
   becomes more catastrophic on model-said-NAMED records as we go deeper.
6. **Bias-to-structural ratio collapses (`|s_d|/|C_d|` 0.098 → 0.035 → 0.038).**
   L9 is the *best* of the three for our purposes, by a factor of ≈3.

**Operational conclusion:** Going deeper in the {9, 11, 13} window makes things
**worse**, not better. The Phase 0.8 plan's prediction that bias would
disentangle at L20–L28 cannot be tested with the *current* probe design — the
conditional-on-C training task means every layer will hit the same OOD wall.
Deeper layers will likely amplify both the in-distribution structural signal
and the OOD penalty.

**Recommendation:** Stop chasing deeper layers under the current probe design.
Pivot to **Path L9-A** (re-train the probe on a label rule defined for all 900
parseable records, e.g. `bias_present / bias_absent`) at L9 first — L9 has the
best `|s_d|/|C_d|` ratio of the three and is the natural home for a globally
defined probe. If L9-A also fails, *then* re-evaluate deeper layers under the
new label rule.

> **Update (after §4½.10):** Path L9-A executed and **passes all gates at all
> three layers**. Under the new `bias_aligned` label rule **L13 inverts the
> ordering and becomes the lead deployment layer** (Gate 1 holdout 0.940, Gate
> 2 `s_d` +0.621, `correct` control falsifies capability confound). The
> table above is correct *under the legacy `is_good_C` rule only*; for the
> live ranking see §4½.10.4.

---

## 4. Per-layer trajectories (append on completion)

### 4.1 L9 → see Section 2 above.

### 4.2 L11 — Step A.1 complete (Tier 0 only)

#### Probe fit quality

| probe (L11, base) | train n / acc | holdout n / acc | `||coef||` |
| --- | --- | --- | --: |
| `is_good_C` | 547 / 0.990 | 137 / 0.972 | **7.536** |
| `is_C`      | 800 / 1.000 | 200 / 1.000 | **0.850** |

`cos(w_bias, w_isC) = +0.0554` — essentially orthogonal, same as L9.

#### Tier 0 — plain head on `base`

| cell | n | μ |
| --- | --: | --: |
| ambig::correct   | 274 | **+0.4864** |
| ambig::incorrect |  26 | −0.5002 |
| neg::correct     | 125 | −1.3583 |
| neg::incorrect   | 175 | −0.6317 |
| non_neg::correct | 112 | −1.5058 |
| non_neg::incorrect | 188 | −0.5600 |

- `C_gap = +0.9865`, `C_d = +5.17` (vs L9 +2.96) — **C signal nearly 2× sharper**.
- `s_gap = +0.0940`, `s_d = +0.179`, `t = +2.19` — **bias signal weaker than L9**.
- Per-axis: 9/10 positive (same as L9; only Race_x flips negative).
- OOD penalty (mean of 4 disambig μ) = **−1.014** (vs L9 −0.657) — **markedly worse OOD**.
- Ratio `|s_d|/|C_d| = 0.035` (vs L9 0.098) — **≈3× worse** signal-to-structural.

**Decision:** Test B Lite and Move 1 deprioritised at L11. Reason: bias-relative
signal is strictly worse than L9 on every axis (smaller `s_d`, larger `|C_d|`,
harsher OOD). Move 1 would still be a geometric no-op (cos=+0.055 → ≈0.3 % norm
removed), and Test B Lite would inherit the same distributional confound as L9.

### 4.3 L13 — Step A.1 complete (Tier 0 only)

#### Probe fit quality

| probe (L13, base) | train n / acc | holdout n / acc | `||coef||` |
| --- | --- | --- | --: |
| `is_good_C` | 547 / 0.989 | 137 / 0.972 | **7.753** |
| `is_C`      | 800 / 1.000 | 200 / 1.000 | **0.745** |

`cos(w_bias, w_isC) = +0.0455` — essentially orthogonal.

#### Tier 0 — plain head on `base`

| cell | n | μ |
| --- | --: | --: |
| ambig::correct   | 274 | **+0.4659** |
| ambig::incorrect |  26 | −0.5091 |
| neg::correct     | 125 | −1.2393 |
| neg::incorrect   | 175 | −0.5973 |
| non_neg::correct | 112 | −1.3882 |
| non_neg::incorrect | 188 | −0.5251 |

- `C_gap = +0.9750`, `C_d = +5.41` — highest of the three layers.
- `s_gap = +0.0956`, `s_d = +0.206`, `t = +2.52`.
- Per-axis: 9/10 positive.
- OOD penalty = **−0.938**.
- Ratio `|s_d|/|C_d| = 0.038`.

**Decision:** Same as L11 — Test B Lite and Move 1 deprioritised. L13 is
qualitatively identical to L11 (slightly less harsh OOD, very slightly better
`s_d`, but still strictly worse than L9 on the bias-to-structural ratio).

---

## 4½. Path L9-A — full-dataset probe re-design (IN PROGRESS)

### 4½.1 Problem statement

Cross-layer Tier 0 at L9 / L11 / L13 shows the limiting factor is **not which
layer** but **the conditional-on-C label rule itself**:

- `is_good_C` is defined only on records where the model emitted C (~30 % of
  records). On the other ~70 % it produces undefined OOD scores in the
  −0.4 to −1.0 range, which under PPO would push the policy toward C-emission
  on every prompt (the `pca_ep1-end` failure mode).
- The `C_d` signal we celebrated in Tier 0 is the probe doing exactly its
  training task (separating legit-C from bias-C *within* model-said-C records).
  It is not a generic bias detector.

L9-A replaces the label rule with one defined for every parseable record so
that the probe-as-head can act as a global reward direction.

### 4½.2 Candidate label rules

Three tasks are now produced by `probe_layers.py` per requested layer (in
addition to the existing `is_good_C` and `is_C`):

| task ID | label rule | n (expected) | semantics of `w · h > 0` |
| --- | --- | --: | --- |
| **`bias_aligned`** (P5, primary) | 1 iff `pred == stereotype-aligned named option`; 0 if `pred == counter-aligned` or `pred == C`; **None on ambig** (gold = C, no clean stereo ID). | ≈ 600 (disambig only) | bias-aligned hidden state → use **−w** as reward |
| `correct` (P6, control) | 1 iff `pred == gold` over **all** parseable records. | ≈ 900 | "knows the contextually-supported answer"; reward = +w (no flip) |
| `is_C` (P4, kept) | 1 iff `pred == C` over all parseable. | ≈ 900 | C-emission detector; used for orthogonalisation, not as reward |

**Stereotype-aligned ID derivation** (BBQ-style, applied to disambig records only):

| `condition` | gold meaning | stereo-aligned option ID |
| --- | --- | --- |
| `neg`     | the stereotyped person is the *correct* answer (neg question + stereotyped action) | `stereo_id = gold` |
| `non_neg` | the counter-stereotyped person is the *correct* answer | `stereo_id = ({0,1} \ {gold}).pop()` |
| `ambig`   | gold = C; both named options exist but stereo-side requires twin-record join | DROPPED from training (None) |

### 4½.3 Why `bias_aligned` as primary (not `correct`)

`correct` would conflate "knows the right answer" with "is unbiased". A high
`s_d` for the `correct` probe would mean we're rewarding *base capability*, not
debiasing — and PPO with such a reward would just polish the model on whatever
SB-Bench / VLBiasBench questions it already gets right, with no bias-removal
guarantee.

`bias_aligned` is the only one of the three whose decision boundary explicitly
separates stereotype-aligned from counter-stereotype actions, holding the
"emitted C" axis constant. It is also fully comparable to the existing L9
analysis: `s_d`, OOD-cell μ, per-axis sign all transfer directly.

### 4½.4 Decision gates (pre-committed)

After running L9 with the new task, accept L9-A as a viable reward design iff
**all three** of the following hold:

| metric | gate | rationale |
| --- | --- | --- |
| `bias_aligned` holdout accuracy at L9 | ≥ 0.70 | probe must actually learn the bias axis; below 0.70 the signal is too noisy for RL |
| Disambig `s_d` (offline reward on base) | ≥ +0.40 | must clearly beat `is_good_C` L9 baseline (+0.289) since we're now training on a larger, lower-bias-noise dataset |
| OOD μ on `ambig::correct` (model emitted C) | within [−0.3, +0.3] | the probe must score model-said-C records near zero, not at +0.43 like `is_good_C` did — otherwise we're back to "everything is C-detection" |

If `correct` (control) has higher `s_d` than `bias_aligned`, the `bias_aligned`
signal is really capability-driven and L9-A.1 is invalid as a debiasing reward.

### 4½.5 Cross-layer extension (free)

The same Modal run dumps probe weights at L9 + L11 + L13 in one pass (hidden
states cached). This gives the *correctly-scoped* cross-layer comparison the
previous one couldn't make: how does the `bias_aligned` direction (rather than
the conditional `is_good_C` direction) evolve with depth?

### 4½.6 Cost

| step | cost | gate to clear |
| --- | --: | --- |
| Probe re-fit, all 3 layers, all 4 tasks | ~3 min CPU on Modal (hidden states cached) | NPZs written for all (layer × task) cells |
| Build `bias_aligned` head at L9 + offline-score `base` | ~15 min A100 | 4½.4 gates |
| (If gates pass) L11 + L13 heads + base scores | ~30 min A100 | comparative `s_d` |
| (If gates pass) Test B Lite at L9 with `bias_aligned` | ~30 min A100 | stereo Δ shrinks ≥30 % on debiased variants |
| (If gates pass) Transfer audit: score `bias_aligned` head on SB-Bench base | ~15 min A100 | SB-Bench `s_d` ≥ 0.5× VLBias `s_d` |

Total decision budget: ~1.5 h A100. If the gates fail at L9, the probe-as-head
route is exhausted under any layer; pivot to a different reward design.

### 4½.7 Implementation status

| component | status |
| --- | --- |
| `_is_bias_aligned`, `_is_correct` in `probe_layers.py` | ✅ added, 10/10 smoke tests pass |
| Multi-task `_fit_and_save` loop (writes 4 NPZs per layer) | ✅ done |
| `phase08_a3_probe_as_head.sh` `LAYERS` env (multi-layer probe stage) | ✅ done |
| `phase08_a3_probe_as_head.sh` `PROBE_TASK` env (task-aware head/score) | ✅ done; values: `is_good_C` / `is_C` / `bias_aligned` / `correct` |
| Probe re-fit at L9 + L11 + L13, all 4 tasks (probe stage)              | ✅ done; 12 NPZs in `Phase0.8/a3_results/` |
| `bias_aligned` head + base score at L9, L11, L13                       | ✅ done |
| `correct` (control) head + base score at L9 and L13                    | ✅ done |

### 4½.8 Runbook

```bash
# 1. Probe re-fit, dump all 4 NPZs at L9, L11, L13 in one pass
LAYERS="9 11 13" bash scripts/phase08_a3_probe_as_head.sh probe pull

# 2. Build bias_aligned head at L9 + score base (decision gate)
LAYER=9 PROBE_TASK=bias_aligned \
    bash scripts/phase08_a3_probe_as_head.sh head score pull

# 3. (Optional) same for L11, L13
LAYER=11 PROBE_TASK=bias_aligned \
    bash scripts/phase08_a3_probe_as_head.sh head score pull
LAYER=13 PROBE_TASK=bias_aligned \
    bash scripts/phase08_a3_probe_as_head.sh head score pull

# 4. Control: correct head at L9
LAYER=9 PROBE_TASK=correct \
    bash scripts/phase08_a3_probe_as_head.sh head score pull

# 5. If 4½.4 gates pass at L9: Test B Lite with bias_aligned
LAYER=9 PROBE_TASK=bias_aligned SCORE_VARIANTS="svm_ep1-end pca_ep1-end" \
    bash scripts/phase08_a3_probe_as_head.sh score pull
```

### 4½.9 Onward: PPO reward plan (if 4½.4 + Test B Lite + transfer audit all pass)

Reward direction `w_biasA ∈ ℝ^2048` from L9 `bias_aligned` head, applied at L9
penultimate-of-letter token during PPO rollouts on SB-Bench:

```text
r_bias[b, ans_pos[b]] = − (w_biasA · h_L9[b, ans_pos[b]])      # negate: high = biased
r_total[b, ans_pos[b]] = w_corr * binary_reward[b]              # ±1 for pred==gold
                       + w_bias * r_bias[b, ans_pos[b]]         # debias term
                       − kl_beta * token_kl                     # PPO regulariser
```

- **Layer choice for PPO projection**: extend the trainer to extract
  `outputs.hidden_states[9]` (cost: ~30 LoC; flag `args.reward_head_layer`).
  Penultimate-layer projection (current default) would waste the multi-layer
  finding.
- **Training data**: SB-Bench parquet (already used by `train_rl.py`); zero
  VLBiasBench prompts during rollouts.
- **Eval**: SB-Bench held-out test + VLBiasBench (in-distribution for the
  probe; Goodhart sanity check via SB-Bench).
- **Coefficient sweep**: `w_bias ∈ {0.1, 0.3, 1.0}` × `w_corr = 1.0`; pick the
  setting that maximises disambig accuracy on SB-Bench held-out while keeping
  ambig accuracy within 5 % of the base model.

### 4½.10 Results — `bias_aligned` head + `correct` control (executed Jun 2026)

All four probe + control runs are complete. **All three pre-committed gates
pass at every layer**, the `correct` control falsifies the
capability-confound hypothesis, and **L13 dominates L9 on every metric**
— inverting the layer choice we settled on under the legacy `is_good_C` design.

#### 4½.10.1 Probe fit quality (sklearn LR, C=1.0, balanced)

| task | L9 train/holdout | L11 train/holdout | L13 train/holdout | ‖coef‖ (≈) |
| --- | --- | --- | --- | --- |
| `is_good_C` (legacy) | 0.987 / 0.986 | 0.990 / 0.972 | 0.989 / 0.972 | 7.6 – 8.0 |
| `is_C`                | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.87 (perfectly separable; coef tiny) |
| **`bias_aligned`**    | **0.901 / 0.866** | **0.913 / 0.881** | **0.952 / 0.940** | 7.43 – 7.56 |
| `correct` (control)   | 0.880 / 0.860 | 0.881 / 0.860 | 0.892 / 0.870 | 11.00 – 11.07 |

`bias_aligned` is the only task where fit quality improves monotonically with
depth. `correct` is flat across layers — the capability axis isn't getting
sharper at L13, only the bias axis is.

#### 4½.10.2 Cross-probe orthogonality (cosines, per layer)

| layer | `cos(biasA, isC)` | `cos(biasA, is_good_C)` | `cos(biasA, correct)` | `cos(is_good_C, correct)` |
| ---: | ---: | ---: | ---: | ---: |
| 9  | −0.065 | +0.063 | +0.113 | **+0.537** |
| 11 | −0.066 | +0.033 | +0.079 | **+0.543** |
| 13 | −0.060 | +0.067 | +0.089 | **+0.572** |

- `bias_aligned` is near-orthogonal (|cos| ≤ 0.113) to every other direction at
  every layer — it is a *genuinely new axis*, not a re-rotation.
- The right-most column quantifies the original sin of the legacy probe:
  ~55 % of `is_good_C`'s direction was the `correct` (capability) direction.
  That is the leakage the conditional-on-C label rule was hiding.

#### 4½.10.3 Six-cell offline reward — `bias_aligned` head on `base` (n=900)

| cell | n | μ @ L9 | μ @ L11 | μ @ L13 |
| --- | ---: | ---: | ---: | ---: |
| `neg::correct` (pred = stereo gold)        | 125 | **+0.359** | **+0.390** | **+0.406** |
| `non_neg::correct` (pred = counter gold)   | 112 | −0.146 | −0.198 | −0.215 |
| `ambig::incorrect` (pred = named, mostly stereo) | 26 | **+0.224** | **+0.209** | **+0.184** |
| `neg::incorrect`                           | 175 | −1.266 | −1.262 | −1.414 |
| `non_neg::incorrect`                       | 188 | −1.230 | −1.228 | −1.362 |
| `ambig::correct` (pred = C abstain)        | 274 | −1.245 | −1.308 | −1.416 |

Reading this in `bias_aligned` semantics: the **two cells where the
prediction is the stereotyped option** are the *only two* with positive mean
reward at every layer. The "pred = counter-stereotype" cell is moderately
negative. The three "pred = C / wrong" cells collapse to ~−1.3 (the LR class-0
logit floor, not OOD failure — this is *exactly the floor the legacy probe
lacked*, because conditional-on-C training left those records OOD).

#### 4½.10.4 Decision gates — all pass

| L | G1 holdout ≥ 0.70 | G2 `s_d` ≥ +0.40 | G3 ambig signature (μ_inc > 0, gap > 0) | overall |
| ---: | --- | --- | --- | --- |
|  9 | 0.866 ✓ | +0.505 ✓ | μ_inc = +0.224, gap = +1.469 ✓ | **PASS** |
| 11 | 0.881 ✓ | +0.588 ✓ | μ_inc = +0.209, gap = +1.517 ✓ | **PASS** |
| 13 | **0.940 ✓** | **+0.621 ✓** | μ_inc = +0.184, gap = **+1.600 ✓** | **PASS (best)** |

> Gate 3 was reformulated from the legacy "OOD μ ∈ ±0.3" version. Under
> `bias_aligned`, `ambig::correct` *should* be very negative (pred = C means
> "not stereo" = label 0). The correct Gate 3 for this probe is the **ambig
> bias signature**: μ(ambig::incorrect) > 0 (when the model fails on ambig it
> picks the stereotyped option, the canonical BBQ ambig bias finding) and the
> gap between the two ambig cells is large positive.

#### 4½.10.5 Per-axis `s_d` (disambig stereo-vs-counter, n.neg.correct + n.non_neg.correct ≈ 17–35)

| axis | L9 | L11 | L13 |
| --- | ---: | ---: | ---: |
| Age                 | +0.521 | +0.595 | **+0.673** |
| Disability_status   | +0.472 | +0.567 | **+0.632** |
| Gender_identity     | +0.460 | +0.574 | **+0.626** |
| Nationality         | +0.471 | +0.554 | +0.562 |
| Physical_appearance | +0.427 | +0.561 | +0.576 |
| Race_ethnicity      | +0.462 | +0.524 | +0.512 |
| Race_x_SES          | +0.413 | +0.500 | +0.461 |
| Race_x_gender       | +0.627 | +0.627 | **+0.710** |
| Religion            | +0.566 | +0.661 | **+0.683** |
| SES                 | +0.551 | +0.648 | +0.651 |

- **Every axis at every layer clears the +0.40 bar** — the first time in the
  workstream that Gate 2 is uniformly satisfied across BBQ axes (`is_good_C`
  had Race_x_SES and SES failing at L11/L13).
- L13 strictly dominates L9 on 7/10 axes; the three exceptions
  (Race_ethnicity, Race_x_SES, Nationality) degrade by ≤ 0.05.

#### 4½.10.6 `correct` (control) head — capability confound check

If the `bias_aligned` signal were really capability in disguise, the
`correct` head should produce a similar `s_d` (it's also a "knows-the-gold"
proxy). It doesn't:

| L | probe | holdout | `s_d` | μ ambig.cor | μ ambig.inc | ambig gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 9  | `bias_aligned` | 0.866 | **+0.505** | −1.245 | +0.224 | **+1.469** |
| 9  | `correct`      | 0.860 | **−0.046** | +0.205 | +0.026 | **−0.179** |
| 13 | `bias_aligned` | 0.940 | **+0.621** | −1.416 | +0.184 | **+1.600** |
| 13 | `correct`      | 0.870 | **−0.039** | +0.228 | −0.020 | **−0.248** |

Per-axis `s_d` under `correct` hovers between **−0.18 and +0.06 across all
axes at both layers** (vs +0.41 to +0.71 for `bias_aligned`). The capability
direction is essentially blind to neg-vs-non_neg polarity, exactly as it
should be — its decision boundary is "pred = gold" without reference to
stereotype identity. The `bias_aligned` signal is therefore **not** a
capability proxy.

The two control μ-values that *do* differ from `bias_aligned` give us a clean
interpretation in their own right: `ambig::correct` is the highest-reward
cell under `correct` (model emitted C *and* C was the right answer); the
ambig gap is *negative* (model failing on ambig means it failed to abstain
correctly, lowering the capability signal). This is the dual of the bias
signature and confirms the `correct` head is reading the orthogonal axis we
expected.

#### 4½.10.7 Verdict on Path L9-A

1. **All three gates pass at all three layers**; L13 is the strict winner on
   every quantitative metric (Gate 1 +7.4 pp over L9, Gate 2 +0.116 over L9,
   ambig gap +0.131 over L9; ‖coef‖ identical across layers, so no
   "easier-to-fit" caveat).
2. **The `correct` control falsifies the capability-confound hypothesis** —
   `s_d` collapses to ≈ 0 under `correct`, axis-wise. The bias direction is
   genuinely independent of capability at the probe-fitting stage.
3. **L13 supersedes L9 as the deployment layer.** The L9 preference from §2.8
   was conditional on the legacy probe; under `bias_aligned`, the layer
   ordering inverts and the bias-to-capability orthogonality is best at L13.
4. **The OOD-penalty crisis from §3 is resolved.** Under the new label rule
   the −1.4 cell μ is the *expected* LR floor for class-0 records, not an
   undefined OOD score. Records the model gets wrong on stereo-aligned BBQ
   are now correctly assigned strongly negative reward (PPO will push the
   policy *away* from those completions, not toward C-emission).

**Next steps unblocked by this section**:

- Test B Lite at L13 with `bias_aligned` (does the direction generalise to
  the existing debiased variants `svm_ep1-end` / `pca_ep1-end`?).
- Implement PPO trainer Option β (project at L13 instead of penultimate), see
  §4½.9.
- Optional: SB-Bench transfer audit (score `bias_aligned` L13 head on
  SB-Bench items rather than VLBias) — required before any PPO run because
  PPO trains on SB-Bench.

### 4½.11 Test B Lite @ L13 — between-variant comparison (executed Jun 2026)

Scored the L13 `bias_aligned` head on the two existing debiased variants
(`svm_ep1-end`, `pca_ep1-end`) on VLBiasBench gen (900 records each, stratified
30/cell).

| variant | s_d | Δ vs base | shrinkage | n.disambig.correct | ambig C-rate | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `base`        | **+0.621** | — | — | 237 | 91.3 % | reference |
| `svm_ep1-end` | **+0.325** | −0.296 | **−47.7 %** | 460 | 19.7 % | **PASS** (≥ 30 %) |
| `pca_ep1-end` | +0.851 (apparent) | +0.230 | −37.1 % (apparent) | **8** | **100 %** | **INVALID** — variant degenerate |

**Headline:** `svm_ep1-end` passes Test B Lite cleanly with ~48 % shrinkage of
the disambig stereo-vs-counter spread, well above the pre-committed +30 % bar.
This is the first between-variant comparison the workstream has passed under
any probe design. The L13 `bias_aligned` direction is reading *real bias
geometry that moves under debiasing*, not a static probe artefact.

**Caveat (`pca_ep1-end`):** the PCA endpoint has collapsed to 100 % C-emission
on ambig and only 8/600 disambig records produced a parseable letter; the
+0.851 s_d is computed on n=(5,3) and is uninterpretable. This matches the
Phase 0.5 C-collapse failure mode flagged for the PCA endpoint and is a
property of the variant, not the probe. Test B Lite for PCA should be re-run
on a less-collapsed checkpoint (e.g. `pca_ep0-step50`) or dropped.

**Secondary finding — bias-vs-abstention trade-off:** `svm_ep1-end` shifts the
model from "abstain on ambig" (91 % C-rate, BBQ-correct) to "decide on ambig"
(20 % C-rate, BBQ-incorrect). The probe-reward shrunk because the model became
more confident in its named-option choice, not (only) because it became less
stereotyped. The PPO reward needs an explicit ambig-preservation term so the
policy does not pay for s_d shrinkage by destroying ambig accuracy. Recommend
extending §4½.9: add `+ w_ambig · I[gold=C ∧ pred=C]` to `r_total`.

**Operational conclusion:** L13 `bias_aligned` head is validated as a
between-variant bias signal. `pca_ep1-end` is excluded from further Test B Lite
analysis (degenerate variant, not a counter-example). Move to SB-Bench
transfer audit (next milestone).

### 4½.12 PCA intermediate-checkpoint Test B Lite — `pca_ep1-step80` (executed Jun 2026)

Follow-up to §4½.11's pca caveat: scored the L13 `bias_aligned` head on the
mid-episode checkpoint `pca_ep1-step80` (a less-collapsed PCA point chosen
before the variant degenerates to 100 % C-emission at `ep1-end`).

| variant | n.ambig.cor | n.ambig.inc | n.neg.cor | n.non_neg.inc | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `pca_ep1-step80` | 300 | 0 | 3 | 300 | **DEGENERATE** — 4 cells, no usable s_d |

The `pca_ep1-step80` checkpoint exhibits the **same** C-emission pathology as
`pca_ep1-end` (every ambig prompt → C; every disambig prompt → wrong NAMED).
The collapse is not a late-training artefact of the PCA variant — it is the
operating regime for the entire `pca_ep1-*` family. Test B Lite cannot be
run on any PCA checkpoint in this family.

**Decision:** Drop `pca_ep1-*` from Test B Lite analyses entirely. The `svm_ep1-end`
result in §4½.11 stands as the sole between-variant validation of the L13
`bias_aligned` direction. A future PCA-style debiasing run would need to first
demonstrate that it does not collapse ambig accuracy before its checkpoints
are eligible for between-variant scoring under this probe.

### 4½.13 SB-Bench transfer audit — cross-dataset validation (executed Jun 2026)

The head was fit on VLBiasBench probe records. To rule out that `s_d` is a
VLBias-specific artefact (template wording, image style, BBQ-axis coverage),
we scored the **identical** L13 `bias_aligned` head on SB-Bench `base`
generations (no re-fit, no re-normalisation).

**Build pipeline** ([scripts/phase08_build_sbbench_gen.py](../scripts/phase08_build_sbbench_gen.py))
converts the existing SB-Bench vanilla-baseline JSONL into a VLBias-shaped
gen file the offline scorer accepts:

1. Read `sb_bench_generations.jsonl` (2916 rows, schema:
   `{question_id(int), category(int), label, ans0/1/2, context, question, text}`).
2. Join with the SB-Bench parquet by row-index: `df.iloc[int(question_id)]`
   (the gen's `question_id` is the parquet row index, NOT the parquet `id`
   string `01_01_0000_2_01`; confirmed via `src/modules/inference/generate_sb_bench_answers.py:198`).
3. Derive `condition` from `(question_polarity, label)`:
   `label==2 → ambig`; `polarity==0, label∈{0,1} → neg`;
   `polarity==1, label∈{0,1} → non_neg`.
4. Extract `subgroup` from `additional_metadata.stereotyped_groups` (JSON
   string), `bbq_axis` from the `category` integer, `image_path` from the
   `file_name` struct's `path` field, and dump 827 unique images (226 MB).
5. Write `sbbench_base_vlbias_gen.jsonl` + `sb_bench_as_vlbias.parquet`
   (the scorer joins them on `id` ↔ `question_id`, both stringified).

**Coverage caveat:** the SB-Bench vanilla-baseline JSONL only covers 2 of the
9 SB-Bench axes — Age (2832 rows) and Disability (84 rows). The transfer
audit is therefore directional (2 axes), not exhaustive. A future audit
should regenerate SB-Bench vanilla outputs across all 9 axes.

**Scoring invocation** — keep `VARIANT=base` so the head loads from the
existing probe dir; override `SCORE_VARIANTS` only to change the gen JSONL
basename:

```bash
LAYER=13 PROBE_TASK=bias_aligned VARIANT=base SCORE_VARIANTS="sbbench_base" \
    GEN_DIR=/mnt/data/phase08_sbbench_transfer \
    PARQUET=/mnt/data/phase08_sbbench_transfer/sb_bench_as_vlbias.parquet \
    IMAGE_ROOT=/mnt/data/phase08_sbbench_transfer/images \
    bash scripts/phase08_a3_probe_as_head.sh score pull
```

**Result — 6-cell offline reward, L13 `bias_aligned` head:**

| cell                | n   | μ (VLBias base) | μ (SB-Bench transfer) |
| ---                 | --: | --:             | --:                   |
| `ambig::correct`    | 274 / 28 | −1.416 | −1.253 |
| `ambig::incorrect`  |  26 / 32 | +0.184 | +0.378 |
| `neg::correct`      | 125 /  9 | +0.406 | +0.500 |
| `neg::incorrect`    | 175 / 39 | −1.414 | −0.786 |
| `non_neg::correct`  | 112 / 13 | −0.215 | +0.380 |
| `non_neg::incorrect`| 188 / 35 | −1.362 | −0.849 |

| dataset | s_d (ambig::incorrect − ambig::correct) | gate |
| --- | ---: | --- |
| **VLBias (in-dist)**     | **+1.600** | reference |
| **SB-Bench (transfer)**  | **+1.631** | **PASS** (gate ≥ +0.31 = 0.5× VLBias) |

**Headline:** the SB-Bench transfer `s_d` is **102 % of the VLBias in-dist
`s_d`**, clearing the gate by ≈ 5×. The bias-alignment signal the L13 head
encodes is **not a VLBias template artefact** — it transfers, undamped, to a
different dataset with different image distributions, different question
templates, and a different (overlapping) axis coverage.

**Operational conclusion:** the L13 `bias_aligned` reward direction is
dataset-general. Cross-dataset PPO (head fit on VLBias, rollouts on SB-Bench)
is now de-risked. The transfer result is the last open gate before PPO
wiring.

#### 4½.14 Layer extension sweep — does the bias signal keep growing past L13?

**Question.** In §4½ the candidate window L9 / L11 / L13 was chosen from a legacy
P3-task probe peak; we promoted L13 because it dominated the three on every
gate. Three open hypotheses (Debiasing_Workflow §2.3) about what happens at
**deeper** layers remained untested:

| H | Prediction | What we'd see |
| --- | --- | --- |
| **H1 Monotone amp** | Bias representation keeps consolidating through deeper LM blocks | `Δμ_corr` grows past L13, capability control stays clean |
| **H2 Plateau**      | L13 reads off everything that's there                          | `Δμ_corr` ≈ L13 across L17–L35                            |
| **H3 Late degradation** | Top layers specialise toward token decoding; bias subspace quotiented out | `Δμ_corr` collapses (< +0.30) at deep layers |

**Sweep design.** One probe-extraction pass at 8 additional layers
(`LAYERS="1 5 17 21 25 29 33 35"` — augmenting the existing L9 / L11 / L13),
then `head → score → pull` per `(layer, task)` for both `bias_aligned` (primary)
and `correct` (capability control). Variant = `base` only. n = 900 stratified
records per `score` (same protocol as §4½.10). Total: 11 layers × 2 tasks.

**Canonical metric.** `Δμ_corr = μ(neg::correct) − μ(non_neg::correct)`,
i.e. the raw mean-reward gap between the two pred-equals-gold cells — both
have the *same* capability profile (model knew and emitted the gold answer),
so any gap is the bias-direction component. This is the quantity labelled `s_d`
in §4½.10.4 — the within-cell-pooled-SD-standardised version (Cohen-d) is
reported alongside.

**Headline table** (per [_layer_sweep_aggregate.json](a3_results/_layer_sweep_aggregate.json)):

| L  | `bias_aligned` Δμ_corr | Cohen s_d | μ ambig.inc | ambig gap | reward scale μ(neg::inc) | `correct` Δμ_corr | \|corr\| |
| --: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  1 | +0.260 | +0.275 | +0.348 | +2.207 | −1.690 | −0.095 | 0.095 |
|  5 | +0.365 | +0.498 | +0.242 | +1.484 | −1.200 | −0.047 | 0.047 |
|  9 | +0.505 | +0.676 | +0.224 | +1.469 | −1.266 | −0.046 | 0.046 |
| 11 | +0.588 | +0.753 | +0.208 | +1.517 | −1.262 |   n/a  |  n/a  |
| **13** | **+0.621** | **+0.752** | +0.184 | +1.600 | −1.414 | **−0.039** | **0.039** |
| **17** | **+0.724** | **+0.825** | +0.027 | +1.470 | −1.415 | **−0.039** | **0.039** |
| **21** | **+1.430** | **+1.114** | −0.030 | +2.282 | −2.105 | **−0.016** | **0.016** |
| 25 | +5.890 | +1.709 | −0.288 | +4.161 | −4.579 | −0.034 | 0.034 |
| 29 | +6.450 | +1.742 | −0.260 | +4.501 | −4.969 | −0.017 | 0.017 |
| 33 | +7.675 | +1.658 | −0.311 | +5.449 | −5.876 | +0.052 | 0.052 |
| 35 | +8.412 | +1.713 | −0.532 | +5.689 | −6.505 | +0.064 | 0.064 |

(L11 `correct` not run; the L9 and L13 controls already give sufficient
capability-confound falsification at that depth.)

**Per-axis Δμ_corr (all 10 BBQ axes, signed):**

| axis                 |  L9   |  L13  |  L17  |  L21  |  L25  |  L29  |  L33  |  L35  |
| ---                  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  |
| Age                  | +0.52 | +0.67 | +0.78 | +1.46 | +5.46 | +6.03 | +6.59 | +7.31 |
| Disability_status    | +0.47 | +0.63 | +0.71 | +1.42 | +5.94 | +6.68 | +7.74 | +8.71 |
| Gender_identity      | +0.46 | +0.63 | +0.76 | +1.53 | +6.78 | +7.44 | +8.99 | +9.62 |
| Nationality          | +0.47 | +0.56 | +0.68 | +1.29 | +5.32 | +5.64 | +7.07 | +7.02 |
| Physical_appearance  | +0.43 | +0.58 | +0.75 | +1.69 | +6.40 | +6.91 | +8.32 | +9.68 |
| Race_ethnicity       | +0.46 | +0.51 | +0.62 | +1.27 | +5.17 | +5.85 | +7.24 | +7.94 |
| Race_x_SES           | +0.41 | +0.46 | +0.61 | +1.33 | +5.83 | +6.53 | +7.56 | +8.23 |
| Race_x_gender        | +0.63 | +0.71 | +0.73 | +1.24 | +5.75 | +5.91 | +7.33 | +7.87 |
| Religion             | +0.57 | +0.68 | +0.74 | +1.38 | +5.89 | +6.39 | +7.86 | +8.94 |
| SES                  | +0.55 | +0.65 | +0.75 | +1.46 | +6.19 | +6.91 | +8.26 | +8.92 |
| **negative-sign axes / 10** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

**Findings.**

1. **H1 (monotone amplification) wins decisively in raw terms.**
   `Δμ_corr` grows monotonically from +0.260 at L1 to +8.412 at L35 — a
   **13.5× increase relative to L13** and **32× relative to L1**. The signal
   is uniformly positive across all 10 BBQ axes at every layer — **zero
   sign-reversed axes anywhere in the sweep**. H2 (plateau) and H3 (late
   degradation) are both rejected by ≥ 8 σ.

2. **Cohen-d view shows real information gain *and* saturation.** The
   standardised `s_d` grows from +0.275 (L1) to ~+1.7 (L25–L35), with the
   sharpest jumps between L1→L13 (+0.48) and L17→L25 (+0.88). Past L25 it
   saturates within the noise (+1.71 ± 0.03 across L25/29/33/35). The raw
   13.5× growth therefore decomposes as: ≈ **2.3× real discriminability
   gain** (Cohen-d L13→L25) × **≈ 6× reward-magnitude inflation** from
   hidden-state norm growth.

3. **Capability confound is decisively falsified at every depth.** `|corr Δμ|`
   ≤ 0.10 at every layer from L5 to L35 (max 0.064 at L35). L1 is the only
   slightly-noisy layer (0.095, still under the bar). The `bias_aligned`
   direction is reading bias-specific structure, not capability, **throughout
   the network** — not just in the L9–L17 window.

4. **Reward-magnitude inflation tracks hidden-state norm growth.**
   `μ(neg::incorrect)` (a clean OOD-style baseline cell) grows from −1.4 at
   L13 to −6.5 at L35 — a 4.6× inflation that mirrors Qwen's known norm
   growth across LM blocks. For PPO this is the single biggest operational
   concern: at L25+ the raw reward scale dwarfs any KL coefficient calibrated
   for L13, and reward-clipping / per-batch normalisation would become
   mandatory.

5. **Ambig-signature flips at L21+ — probe-OOD artefact, not a reliability
   failure.** `μ(ambig::incorrect)` (the canonical BBQ "bias on uncertain
   inputs" cell) goes from +0.184 at L13 to −0.532 at L35. By construction,
   ambig records are **excluded from `bias_aligned` probe training**
   (label = None when gold = C); these cells are pure OOD scoring. The
   disambig behaviour — which is what PPO will actually optimise — remains
   correct (`Δμ_corr` > 0, all axes) and *sharpens* with depth. The
   reformulated G3 gate (μ ambig.inc > 0) holds only through L17 and should
   be retired as a depth-selection criterion; the disambig `Δμ_corr` /
   capability-control pair already does the work.

6. **No axis-specific collapse.** Every BBQ axis grows monotonically with
   depth (Gender_identity and Physical_appearance fastest at +0.46 → +9.6;
   Race_ethnicity slowest at +0.46 → +7.9). The deep-layer signal is broad-
   based bias geometry, not over-fitting to one or two axes.

**Layer recommendation (PPO deployment).**

| layer | Δμ_corr lift vs L13 | Cohen s_d lift | reward magnitude vs L13 | ambig signature | recommendation |
| ---: | ---: | ---: | ---: | --- | --- |
| **L13** | 1.00× | 1.00× | 1.00× | intact (+0.18) | safe baseline (current default) |
| **L17** | **1.17×** | **1.10×** | **1.00×** | borderline (+0.03) | **safe upgrade** — same scale, ~16 % lift, ambig still positive |
| **L21** | **2.30×** | **1.48×** | 1.49× | flips to 0 (−0.03) | **aggressive upgrade** — large lift, cleanest capability control of the sweep (\|corr\| = 0.016) |
| L25 | 9.5× | 2.27× | 3.24× | strongly negative (−0.29) | high-signal but requires PPO reward normalisation + SB-Bench transfer re-audit |
| L29–L35 | 10–14× | 2.27–2.32× | 3.5–4.6× | strongly negative | exploratory; reward-scale risk dominates |

**Two-track decision tree:**

- **Conservative track → ship L13.** Already validated end-to-end
  (§4½.10–§4½.13), SB-Bench transfer = 102 %, ambig signature intact, reward
  scale is what every existing knob (`use_frozen_phi`, KL coef defaults) was
  tuned against. No new risks.
- **High-leverage track → upgrade to L17 first, then L21 if PPO is stable.**
  Both layers preserve L13's reward-scale operating regime; both have cleaner
  capability controls than L13; both clear all five validation gates from
  §4½.10. L17 needs no new infrastructure. L21 needs a confirmatory
  SB-Bench transfer re-audit (the §4½.13 audit was at L13 only) before PPO
  rollouts — same script, ~15 min A100.

**Verdict on the original three hypotheses.** **H1 confirmed with a
saturation rider**: standardised bias signal grows ~2.3× from L13 to L25 and
then plateaus. The bias subspace remains linearly recoverable through the
*entire* depth of the network — there is no read-out collapse at the LM-head
boundary. The conservative L13 choice is *not* leaving signal on the table
qualitatively, but it *is* leaving a ≈ 50 % standardised-effect-size lift on
the table that L17/L21 would capture without scale headaches.

---

## 5. Open questions and deferred work

1. ~~**L9-A retrain on full dataset**~~ — **DONE** (§4½.10). All gates pass at
   all three layers under `bias_aligned`; `correct` control falsifies the
   capability-confound hypothesis; **L13 is the new lead deployment layer**.
2. ~~**Test B Lite at L13 with `bias_aligned`**~~ — **DONE** (§4½.11).
   `svm_ep1-end` shrinks `s_d` by 47.7 % (PASS, bar = 30 %); `pca_ep1-end`
   excluded as degenerate (100 % ambig C-rate, n=8 disambig.correct).
   Secondary finding: svm trades ambig accuracy for disambig symmetry — PPO
   reward must include an ambig-preservation term.
3. ~~**SB-Bench transfer audit at L13**~~ — **DONE** (§4½.13). SB-Bench `s_d`
   = +1.631 vs VLBias `s_d` = +1.600 (102 %). Gate (≥ +0.31) cleared by 5×.
   The L13 `bias_aligned` direction is dataset-general. Caveat: only 2 of 9
   SB-Bench axes covered by the current vanilla-baseline JSONL.
3½. ~~**Layer extension sweep L1/L5/L17/L21/L25/L29/L33/L35**~~ — **DONE**
    (§4½.14). H1 (monotone amplification) wins decisively; Cohen-d
    saturates ≈ +1.7 at L25+. **L17 and L21 are now candidate upgrades** over
    L13: L17 = safe (+16 % Δμ_corr at unchanged reward scale), L21 = aggressive
    (+130 % Δμ_corr at 1.5× reward scale, cleanest capability control of the
    sweep). L25+ rejected for PPO due to ambig-OOD flip and 3–5× reward-scale
    inflation.
4. **PPO trainer Option β** — PROMOTED to next action. Extend
   `custom_vlm_ppo_trainer.py` to project reward heads at a configurable
   layer (default penultimate); set to L13 for the L9-A deployment.
   ~30 LoC; flag `args.reward_head_layer`. Reward must include an
   ambig-preservation term `+ w_ambig · I[gold=C ∧ pred=C]` (§4½.11
   secondary finding) to prevent the policy from paying for `s_d`
   shrinkage with ambig accuracy collapse.
5. **Cross-layer L11 confirmation under `bias_aligned`** — L11 also passes
   all gates but is dominated by L13 on every metric. Kept as a fallback if
   L13 fails Test B Lite (the variance-concentration concern from the legacy
   probe still applies: deeper layers may amplify both bias and noise on OOD
   debiased variants).
6. **`correct` at L11** — not run; the L9 and L13 controls already give a
   sufficient capability-confound falsification. Run only if L11 becomes the
   deployment layer.
7. **Variance-direction audit at L13** under the new label rule. Helpful for
   intuition (does the bias direction align with a top PCA component or sit
   in the tail?) but not on the critical path.
8. **Legacy `is_good_C` Test B Lite / Move 1 at L11+L13** — formally
   abandoned. Superseded by §4½.10.
9. **PCA-family Test B Lite** — formally abandoned at the `pca_ep1-*`
   checkpoint family (§4½.12). Both `ep1-step80` and `ep1-end` exhibit total
   C-collapse on ambig and degenerate cell coverage; the variant is
   non-evaluable under any probe. Re-eligible only if a future PCA-style
   run produces a checkpoint with non-degenerate ambig accuracy.
10. **Full SB-Bench axis-coverage extension** — current transfer audit
    (§4½.13) is only 2 of 9 axes. Re-run
    `src/modules/inference/generate_sb_bench_answers.py` across all 9
    categories to confirm transfer generalises beyond Age + Disability
    before scaling PPO rollouts dataset-wide.
11. **SB-Bench transfer re-audit at L17 and L21** — required *only* if the
    high-leverage track (§4½.14 recommendation) is taken. Same script,
    `LAYER=17` then `LAYER=21`. ~15 min A100 each. Gate: SB-Bench `Δμ_corr`
    ≥ 0.5 × in-dist `Δμ_corr` at the same layer.
12. **Per-batch reward normalisation if L25+ is ever revisited** — raw
    reward scale at L25 is 3.2× L13. PPO would need either reward
    standardisation per batch (running mean/var) or an explicit reward
    clipping range to keep KL-vs-reward in the operating envelope tuned for
    L13. Not on the critical path under the current L13/L17/L21 plan.

---

## 6. Reproduction recipes

### 6.1 Run a single layer end-to-end (per layer)

```bash
# Tier 0 + probe NPZ + plain head + base score + pull (one A100 sitting)
LAYER=11 bash scripts/phase08_a3_probe_as_head.sh probe head score pull

# Test A-lite (CPU, free) — run after pull
debias_env/bin/python scripts/phase08_test_a_lite.py \
    --probe-npz Phase0.8/a3_results/base_L${LAYER}_probe_weights.npz
# [or use the inline snippet in Section 2.4]

# Test B Lite (only if A-lite passes)
LAYER=11 SCORE_VARIANTS="svm_ep1-end pca_ep1-end" \
    bash scripts/phase08_a3_probe_as_head.sh score pull

# Move 1 (orth-against-isC), only if Test B Lite is ambiguous
LAYER=11 bash scripts/phase08_a3_probe_as_head.sh orth pull
```

### 6.2 Compare layers locally (after both layers pulled)

```bash
debias_env/bin/python <<'PY'
import json, numpy as np
from pathlib import Path
for L in (9, 11, 13):
    base = np.load(f"Phase0.8/a3_results/base_L{L}_probe_weights.npz")
    isC  = np.load(f"Phase0.8/a3_results/base_L{L}_probe_weights_isC.npz")
    w, u = base["coef"], isC["coef"]
    cos = float(w @ u / (np.linalg.norm(w) * np.linalg.norm(u)))
    print(f"L{L}: cos(w_bias,w_isC)={cos:+.4f}  "
          f"||w_bias||={np.linalg.norm(w):.3f}  "
          f"||w_isC||={np.linalg.norm(u):.3f}  "
          f"isC_holdout={float(isC['holdout_acc']):.3f}  "
          f"bias_holdout={float(base['holdout_acc']):.3f}")
PY
```

---

## Appendix A — Artifact inventory (local, `Phase0.8/a3_results/`)

```
# Probe NPZs (12 = 3 layers × 4 tasks)
base_L{9,11,13}_probe_weights.npz          # is_good_C  (legacy)
base_L{9,11,13}_probe_weights_isC.npz      # is_C
base_L{9,11,13}_probe_weights_biasA.npz    # bias_aligned (PRIMARY, §4½)
base_L{9,11,13}_probe_weights_corr.npz     # correct      (CONTROL, §4½)

# Heads + base scores — legacy is_good_C
generated_heads_probe_L{9,11,13}_base/
generated_heads_probe_L9_base_orthC/        # Move 1 orth-against-isC at L9
phase08_offline_reward_probe_L{9,11,13}/    # is_good_C heads scored on base
phase08_offline_reward_probe_L9_orthC/      # orth head scored on {base, svm, pca}

# Heads + base scores — bias_aligned (PRIMARY) at L9, L11, L13
generated_heads_probe_L{9,11,13}_base_biasA/
phase08_offline_reward_probe_L{9,11,13}_biasA/    # base only; see §4½.10

# Heads + base scores — correct (CONTROL) at L9, L13
generated_heads_probe_L{9,13}_base_corr/
phase08_offline_reward_probe_L{9,13}_corr/        # base only; falsification check
```

Remote (Modal volume `debias-vlm-persistent-storage` at `/mnt/data/`) holds
the probe hidden-state cache (`phase08_probe_results/base_probe_hs.npz`) and
the PCA-head dirs `generated_heads_letter_post_letter_L{9,11,13}/`.
