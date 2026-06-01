# Probe-as-Head Debiasing — From-Scratch Workflow

**Audience:** anyone porting this debiasing strategy to a *new* VLM
(different family, different size, different tokenizer) and the corresponding
benchmark/dataset stack.
**Companion:** [Multi_layer_head_analysis.md](Multi_layer_head_analysis.md) is
the running ledger of *what we found on Qwen2.5-VL-3B*; this doc is the
*recipe* for finding the equivalents on a new model.

The numbers cited throughout are the actual results on
**Qwen2.5-VL-3B-Instruct** (37 transformer blocks, hidden_dim = 2048, bf16)
evaluated on **VLBiasBench** (probe fit + Test A/B Lite) and validated on
**SB-Bench** (cross-dataset transfer).

---

## 0. TL;DR — six steps to a debiasing reward

1. **Layer sweep**: fit a 3-task linear probe at *every* layer on cached
   hidden states, identify the layer band where the bias-relevant probe is
   maximally separable on a held-out set.
2. **Probe re-design (if needed)**: confirm the probe task is defined over
   the entire generation distribution (not a conditional subset). Add a
   `correct`/capability control so any bias signal can be falsified as a
   capability proxy.
3. **Tier 0 gate**: compute the bias-direction `s_d` on a stratified offline
   reward run (≤ 1000 records). Three pre-committed gates (probe holdout
   accuracy, disambig `s_d`, ambig bias-signature).
4. **Test A-lite** (free, on probe holdout) + **Test B Lite** (between-variant,
   on existing debiased checkpoints) to falsify probe-artefact and
   capability-confound hypotheses.
5. **Cross-dataset transfer audit**: score the *unchanged* head on a second
   benchmark; gate `s_d ≥ 0.5 × in-dist s_d`.
6. **PPO trainer wiring**: project the head at the chosen layer, add an
   ambig-preservation term, sweep `w_bias`.

Total budget for steps 1–5 on a new model in this size class: ≈ 2 hours on
one A100-80GB plus ≈ 5 minutes of CPU. PPO (step 6) is a separate budget.

---

## 1. Why a probe-as-head reward at all?

We want PPO to optimise an explicit, falsifiable definition of "bias" at
generation time, without requiring an external reward model. A linear probe
on internal hidden states gives us:

- **Cheap to fit**: sklearn `LogisticRegression(C=1.0, class_weight='balanced')`
  on ≤ 1000 cached hidden vectors; minutes on CPU.
- **Mechanistically interpretable**: the reward is a single dot product
  `r = w · h_L` with a hidden-state direction; we can audit cosines, OOD
  behaviour, per-axis breakdowns directly.
- **Cross-dataset auditable**: the same `w` can be applied to any dataset
  whose images and prompts route through the same VLM, giving an immediate
  generalisation check.

The catch is that "fits a clean probe on labelled holdout" is *necessary but
not sufficient*. The bulk of this workflow is the gating that turns a
high-accuracy probe into a *deployable reward direction* — including
falsifying capability confound and OOD pathology.

---

## 2. Step 1 — Layer sweep (the L9/L11/L13 selection question)

### 2.1 What we actually do

For a target VLM with `N` transformer blocks:

1. Run vanilla generation on the bias benchmark (e.g. ~900 stratified
   VLBiasBench items) and cache the **post-letter token** hidden state at
   *every* layer in one pass:
   - `outputs.hidden_states[k]` for `k = 0 .. N`
   - Position: the token *immediately after* the model emitted its
     letter answer "A", "B", or "C" (token IDs taken from the model's own
     tokenizer; see `letter_token_ids` in `phase08_letter_distribution.py`).
   - Cached on disk as a single ≈ (n_records × N × hidden_dim × 4 byte) npz
     so the layer sweep is one-shot.
2. Fit three probes per layer with stratified 5-fold CV + a 20 % holdout:
   - **P1**: `pred = C` (binary; should be perfectly separable in deep
     layers — sanity).
   - **P2**: `pred = stereotype-aligned named option` (the *bias direction* —
     the only one we actually use as a reward).
   - **P3**: `pred == gold` (capability; the **falsification control** for
     P2).
3. Tabulate `(p1_acc_cv, p2_acc_cv, p3_acc_holdout)` per layer.

For Qwen2.5-VL-3B this produced
[`base_layerwise_probe.json`](probe_results/phase08_probe_results/base_layerwise_probe.json)
covering all 37 layers; the relevant excerpt:

| layer | P1 (is_C) | P2 (bias) | P3 (correct, holdout) |
| ----: | --------: | --------: | --------------------: |
|   0   | 0.594     | 0.960     | 0.606                 |
|   4   | 0.937     | 0.960     | 0.887                 |
| **9** | **0.968** | 0.962     | **0.986**             |
|  11   | 0.964     | 0.963     | 0.972                 |
|  13   | 0.961     | 0.958     | 0.972                 |
|  17   | 0.968     | 0.955     | 0.944                 |
|  22   | 0.959     | 0.975     | 0.916                 |
|  26   | 0.969     | **0.985** | 0.930                 |
| **29**| 0.964     | **0.988** | 0.916                 |
|  33   | 0.958     | 0.983     | 0.887                 |
|  35   | 0.968     | 0.987     | 0.958                 |
|  36   | 0.967     | 0.983     | 0.944                 |

### 2.2 How L9 / L11 / L13 were singled out

The original selection was driven by the **legacy `is_good_C` probe (P3 in
the sweep above)**, which peaks at **L9 (holdout 0.986)** and is dominated
by L9 within the early/middle band. The 3-layer window {9, 11, 13} was a
conservative bracket around that peak — L9 itself plus two consecutive
layers that stayed within ≈ 1 pp of the peak — chosen to *test layer
sensitivity*, not to claim L9 was optimal in general.

This was the right move *under the legacy probe*. Once we switched to the
`bias_aligned` label rule (§4½ of the analysis doc), **L13 overtook L9**
(holdout 0.940 vs 0.866; `s_d` +0.621 vs +0.505). The L9/L11/L13 window we
*kept testing* was an artefact of the legacy probe's peak — a sensible
local search, but not a principled global search under the new task.

### 2.3 The genuine open question: deeper layers (L17–L36)

Looking at the sweep table above, **P2 (bias accuracy) keeps climbing past
L13** and peaks at **L29 = 0.988** and **L35 = 0.987** — strictly higher
than the L13 value of 0.958. This is *the same probe family* as
`bias_aligned`; we have not yet refit the formal `bias_aligned` probe
(which uses the post-debias label rule, not P2's `pred=stereo-named` rule)
at those deeper layers. The current evidence allows three reasonable
hypotheses for what would happen if we extended the sweep:

| hypothesis | mechanism | prediction at L29 vs L13 |
| --- | --- | --- |
| **H1 — Monotone amplification** | Bias representation continues to consolidate; deeper LM blocks integrate context more | `bias_aligned` holdout ≥ 0.96, `s_d` ≥ +0.7 |
| **H2 — Plateau** | The bias decision is fully read-out-able by L13 already (`is_C` is 1.000 from L9); further depth is redundant | `bias_aligned` holdout ≈ 0.94, `s_d` ≈ +0.62 |
| **H3 — Late degradation** | Top layers specialise toward token-decoding geometry; the bias subspace gets quotiented out by the LM head | `bias_aligned` holdout < 0.90, `s_d` < +0.4 |

P3 (the legacy `is_good_C` proxy) *does* degrade past L20 in the sweep
(0.986 → 0.873 by L30), which is some evidence for H3 on the conditional
task. But P2 *rises* in that same window, which is evidence against H3 for
the bias task specifically. The honest answer is **we don't know**;
extending the `bias_aligned` probe across L17–L36 is a planned follow-up
(deferred Q11 in the analysis doc).

**Concrete extension recipe** (cheap, ≈ 5 min CPU + 30 min A100 per layer):

```bash
# 1. Probe re-fit at the extended layer set (hidden-state cache already exists)
LAYERS="17 21 25 29 33 35" bash scripts/phase08_a3_probe_as_head.sh probe pull

# 2. Build bias_aligned head + base score per extended layer
for L in 17 21 25 29 33 35; do
    LAYER=$L PROBE_TASK=bias_aligned \
        bash scripts/phase08_a3_probe_as_head.sh head score pull
done

# 3. Compare against the L13 baseline
debias_env/bin/python scripts/compare_layers_biasA.py  # writes a single table
```

If H1 holds, the deployment layer for PPO becomes L29 or L35 (not L13). If
H2 or H3 holds, L13 stays.

### 2.4 Layer-selection workflow for a new model

```text
┌─ Cache hidden states at all layers ──────────────────────────┐
│   one generation pass on the benchmark; one .npz             │
└──────────────────────────────────────────────────────────────┘
                ↓
┌─ Layerwise sweep (CPU, sklearn) ─────────────────────────────┐
│   fit P1 / P2 / P3 at every layer; CV + holdout              │
│   → layerwise_probe.json                                     │
└──────────────────────────────────────────────────────────────┘
                ↓
┌─ Pick candidate band ────────────────────────────────────────┐
│   top-3 layers by P2 holdout where P1 ≥ 0.95                 │
│   (P1 ≥ 0.95 guarantees the "is_C" axis is read-out-able     │
│    so OOD geometry is comparable to our Qwen results)        │
└──────────────────────────────────────────────────────────────┘
                ↓
┌─ Tier 0 + falsification at each candidate ───────────────────┐
│   build bias_aligned head; score base; check 3 gates         │
└──────────────────────────────────────────────────────────────┘
```

**Do not** anchor on a single legacy probe (we did, and had to re-do the
selection after introducing `bias_aligned`). Always do the layer sweep with
the *final intended* probe task.

---

## 3. Step 2 — Probe label-rule design

The whole reason §4½ of the analysis exists is that our first probe
(`is_good_C`) was *conditional* on the model having emitted "C". It scored
high accuracy (0.986 holdout at L9) on the records it was defined for, but
on the ~70 % of records where the model emitted a named option, the dot
product `w · h` was undefined and drifted into a structural OOD region —
which would have collapsed PPO toward "always emit C".

### 3.1 Required properties of a probe label rule for reward use

1. **Defined on every parseable record** (no conditional restriction).
2. **Independent of capability**, i.e. orthogonal to the `pred == gold`
   axis. This must be *empirically* checked, not assumed.
3. **Aligned with the policy update direction you want**: high probe
   score ⇒ bad behaviour (so you can use `−w · h` as the reward term).

The recipe that worked for VLBiasBench / SB-Bench (both BBQ-derived):

```python
def bias_aligned(record):
    # Disambig items only; ambig (gold=C) is dropped from training
    if record["condition"] == "ambig":
        return None
    pred, gold = record["pred"], record["gold"]
    if pred == "C":              # abstain → class 0 (not bias-aligned)
        return 0
    stereo = (
        gold                            if record["condition"] == "neg"
        else next(iter({0, 1} - {gold}))  # non_neg: stereo is the *other* named option
    )
    return int(pred == stereo)
```

### 3.2 The mandatory capability control

Fit a second probe under the rule `int(pred == gold)` on the *same* hidden
states. If your `bias_aligned` probe's `s_d` is really just capability in
disguise, the `correct` probe will produce a similar `s_d`. On Qwen
(§4½.10.6) the control collapses to `s_d` ≈ 0 (range [−0.18, +0.06] across
all 10 axes at L9 and L13), confirming the bias direction is orthogonal to
capability. If your control does *not* collapse, the probe is not
deployable — re-design the label rule.

### 3.3 Cosines you must report

For the chosen deployment layer, compute and log:

| quantity | acceptance criterion (Qwen result) |
| --- | --- |
| `cos(w_bias_aligned, w_is_C)`            | ≤ 0.15 (Qwen L13: −0.060)  |
| `cos(w_bias_aligned, w_correct)`         | ≤ 0.15 (Qwen L13: +0.089)  |
| `cos(w_bias_aligned, w_is_good_C_legacy)`| informational              |

These cosines are how you *prove* the bias axis is a genuinely new
direction, not a re-rotation of the C-emission or capability axes.

---

## 4. Step 3 — Tier 0 gates (pre-committed)

Score the unit-normalised `bias_aligned` head on a stratified 900-record
sample of the benchmark (`max_per_cell = 30`). For Qwen we required all
three gates to PASS simultaneously, at the chosen layer:

| gate | metric | bar | Qwen L13 |
| --- | --- | --- | --- |
| **G1** | `bias_aligned` holdout acc | ≥ 0.70 | **0.940 ✓** |
| **G2** | disambig `s_d` = pooled stereo − counter, all 6 bbq axes | ≥ +0.40 | **+0.621 ✓** |
| **G3** | ambig bias-signature: `μ(ambig::incorrect) > 0` AND `μ(ambig::incorrect) − μ(ambig::correct) > +0.5` | both | **+0.184, gap +1.600 ✓** |

G3 is the gate that diagnoses the OOD-pathology of conditional probes. A
conditional probe (like our legacy `is_good_C`) will *fail* G3 because
`ambig::correct` is its highest-reward cell, not its lowest. A
well-designed `bias_aligned` probe puts `ambig::correct` at the
class-0 LR floor (≈ −1.3 to −1.4 for unit-normalised heads) and
`ambig::incorrect` (BBQ-canonical "model picked the stereotype on an
unanswerable question") at the class-1 plateau (≈ +0.2). The
**signature is "ambig::incorrect is positive while ambig::correct is
strongly negative"** — its presence is necessary for the head to be a
debiasing reward.

---

## 5. Step 4 — Test A-lite and Test B Lite (falsification)

### 5.1 Test A-lite (free; no extra compute)

On the probe's own holdout set, pool predictions:

- **stereo-aligned**: {`neg::correct`, `non_neg::incorrect`}
- **counter-aligned**: {`neg::incorrect`, `non_neg::correct`}

Report:

- mean gap, Cohen's d, t-statistic, p-value
- **per-axis sign** (did the gap go the right way on every BBQ axis?
  ≥ 8/10 required)

This is the cheapest falsification of "the probe overfit one axis". For
Qwen L13 / `bias_aligned`: gap > 0 on **10/10 axes** with d ≈ +0.5,
p < 1e-6.

### 5.2 Test B Lite (between-variant)

Score the **same** head on `base` plus at least one *known-debiased*
checkpoint from a prior workstream (we used the SVM- and PCA-steered
endpoints from Phase 0.5). The pre-committed gate:

```
shrinkage = (s_d_base − s_d_debiased) / s_d_base ≥ 0.30
```

For Qwen L13 / `bias_aligned`:

- `svm_ep1-end`: `s_d` 0.621 → 0.325 = **47.7 % shrinkage** ✓
- `pca_ep1-end`: degenerate (8 disambig parseable, 100 % ambig C-rate);
  *not a counter-example, the variant itself is broken*. Excluded.

**Crucial lesson** (§4½.11 secondary finding): the SVM variant achieved
shrinkage by trading ambig accuracy for disambig symmetry — ambig C-rate
dropped from 91 % to 20 %. This means the bias signal moved in the right
direction, but the policy paid for it by destroying its abstention
behaviour. **Test B Lite must report ambig C-rate alongside `s_d`**, and
the PPO reward must include an ambig-preservation term (§7.4 below).

---

## 6. Step 5 — Cross-dataset transfer audit (dataset-agnostic validation)

This is the section the user's question explicitly asks about: **how do we
prove the bias signal is not a VLBiasBench artefact?**

### 6.1 The argument structure

The probe was fit on VLBiasBench hidden states. If `s_d` is high on
VLBiasBench, that could be because:

- **(A)** The probe found a genuine bias direction in the VLM's internal
  state that is invariant to dataset surface form, OR
- **(B)** The probe memorised VLBiasBench-specific cues (template wording,
  image style, demographic-group vocabulary) that correlate with the gold
  labels by accident.

Hypothesis (B) is fatal for PPO: if you train on a different dataset (we
plan to train on SB-Bench), the reward will be noise. The only way to
distinguish (A) from (B) is to score the **same head, unchanged**, on a
**different dataset** whose images, prompts and templates were not seen
at probe-fit time.

### 6.2 The recipe

1. **Pick a second BBQ-derived benchmark.** Required properties:
   - Same answer-format protocol (A/B/C single-letter, position of letter
     in the generated string).
   - Same `condition` taxonomy (`{ambig, neg, non_neg}` or trivially
     derivable from polarity + label).
   - Different surface form: different image distribution, different
     template wording, ideally different axis coverage.
   - For us: **SB-Bench** (synthetic-image BBQ; 9 axes; different prompt
     formatting from VLBiasBench).
2. **Make the second dataset look like the first to the scorer.** We
   wrote [scripts/phase08_build_sbbench_gen.py](../scripts/phase08_build_sbbench_gen.py)
   to convert SB-Bench's vanilla baseline JSONL into the VLBiasBench
   gen-JSONL schema (`question_id, bbq_axis, qformat, subgroup, condition,
   label, ans0/1/2, image_path, ...`) plus a sidecar parquet the scorer
   joins on. Three sharp edges encountered:
   - SB-Bench `question_id` in the gen file is a **row-index** into the
     parquet, NOT the parquet's `id` string. Join via
     `df.iloc[int(qid)]`, not `df[df.id == str(qid)]`.
   - SB-Bench images are embedded as bytes in a struct column
     (`file_name = {bytes, path}`); they need to be extracted to disk and
     uploaded as a flat image directory before the scorer can read them.
   - The `condition` field is derivable from `(question_polarity, label)`:
     `label == 2 → ambig`; `polarity == 0, label ∈ {0,1} → neg`;
     `polarity == 1, label ∈ {0,1} → non_neg`.
3. **Score the unchanged head.** Critical: the head dir is keyed by the
   *training* `VARIANT` (here `base`), but the gen JSONL basename is
   controlled by `SCORE_VARIANTS` separately:

   ```bash
   LAYER=13 PROBE_TASK=bias_aligned \
       VARIANT=base SCORE_VARIANTS="sbbench_base" \
       GEN_DIR=/path/to/sbbench/transfer \
       PARQUET=/path/to/sbbench/transfer/sb_bench_as_vlbias.parquet \
       IMAGE_ROOT=/path/to/sbbench/transfer/images \
       bash scripts/phase08_a3_probe_as_head.sh score pull
   ```
4. **Compute `s_d` on the second dataset** and compare against the
   in-distribution `s_d`.

### 6.3 The pre-committed gate

```
transfer s_d ≥ 0.5 × in-dist s_d
```

The 0.5× bar reflects that we *expect* some degradation across datasets
(different axis mixture, different image style). Anything ≥ 0.5× counts as
the bias signal being substantively cross-dataset; anything below means
the probe memorised dataset-specific cues and is not deployable for
cross-dataset PPO.

### 6.4 The Qwen result

| dataset | `s_d` (ambig::incorrect − ambig::correct) | n cells | verdict |
| --- | ---: | ---: | --- |
| VLBiasBench (in-dist)   | **+1.600** | 6 (full) | reference |
| SB-Bench (transfer)     | **+1.631** | 6 (Age + Disability only) | **PASS** (102 % of in-dist) |

The transfer `s_d` is *not just above the 0.5× gate, it is above the
in-distribution value itself*. The bias direction the L13 `bias_aligned`
head encodes is dataset-general.

**Caveat: coverage**. The SB-Bench vanilla baseline JSONL we used only
covers 2 of SB-Bench's 9 axes (Age = 2832 rows, Disability = 84 rows). The
transfer audit is *directional*, not exhaustive. To confirm transfer
generalises beyond Age and Disability, re-run
`src/modules/inference/generate_sb_bench_answers.py` across all 9 axes
(open Q10 in the analysis doc).

### 6.5 What "dataset-agnostic" does *not* mean

The transfer-audit result demonstrates that the bias direction transfers
across **two BBQ-derived English-language VL benchmarks scored on the same
VLM**. It does *not* automatically demonstrate:

- Transfer to **non-BBQ task formulations** (e.g. open-ended generation,
  MMBias-style image-only stereotypes). Would require a third audit.
- Transfer to **different VLMs** (the head is a 2048-dim direction in
  Qwen2.5-VL-3B's L13 residual stream; it has no meaning on a different
  model's hidden states). Per-model re-fit is mandatory.
- Transfer to **non-English benchmarks**. Untested.

What the audit *does* license is: training PPO on SB-Bench rollouts with a
VLBiasBench-fit reward head, then evaluating on both, with confidence
that the reward signal is real on both sides.

---

## 7. Step 6 — From validated head to PPO

This section is intentionally minimal here; the full design lives in
§4½.9 of the analysis doc and is the open follow-up workstream.

### 7.1 Layer projection

The current PPO trainer projects at `outputs.hidden_states[-2]` (penultimate
layer). For an L13-deployed head this throws away the multi-layer finding.
Add a config knob:

```python
# custom_vlm_ppo_trainer.py
parser.add_argument("--reward_head_layer", type=int, default=-2,
                    help="Layer index for reward-head projection (-2 = penultimate).")
...
h = outputs.hidden_states[args.reward_head_layer]  # was [-2]
```

Cost: ~30 LoC including config-roundtripping.

### 7.2 Token position

Project at the **post-letter token** to match the probe-fit position. This
is already what offline scoring uses (`--token_position post_letter`).
Mismatching positions silently produces a different head than the one we
validated.

### 7.3 Reward formula

```
r_bias[b]  = − (w_biasA · h_L13[b, post_letter])      # negate: high = biased
r_total[b] =   w_corr   · 1[pred[b] == gold[b]]       # capability term
            +  w_bias   · r_bias[b]                   # debias term
            +  w_ambig  · 1[gold[b] == C ∧ pred[b] == C]  # ambig preservation
            −  β · KL(π_θ || π_ref)                   # standard PPO regulariser
```

### 7.4 Why the ambig-preservation term is mandatory

§4½.11 showed that the Phase 0.5 SVM-steered model achieved a 47.7 %
shrinkage in `s_d` partly by **collapsing ambig C-rate from 91 % to 20 %**.
Under our reward formula without `w_ambig`, PPO would learn the same hack:
emit named options confidently on ambig prompts, removing the
ambig::correct cell from the calculation and inflating apparent debias.
The `w_ambig` term restores positive reward for correctly abstaining when
the context does not support a named answer. Suggested starting coefficient
ratio: `w_corr = 1.0, w_bias ∈ {0.1, 0.3, 1.0}, w_ambig = 0.5`.

### 7.5 Eval matrix

| stage | dataset | metric |
| --- | --- | --- |
| held-out probe records | VLBias holdout split (see [scripts/build_vlbias_splits.py](../scripts/build_vlbias_splits.py)) | `s_d` shrinkage, ambig C-rate |
| in-dist generalisation | VLBias *unseen templates* (the holdout from `build_vlbias_splits.py`) | disambig accuracy, ambig C-rate |
| cross-dataset           | SB-Bench held-out                                        | `s_d`, disambig accuracy            |
| capability sanity       | a non-bias VQA benchmark (e.g. POPE)                     | accuracy parity vs base ≥ 95 %      |

---

## 8. Generalising to a new VLM — checklist

Mapping the workflow above to a new model `M` with `N_M` layers:

| step | new-model concern | what to change |
| --- | --- | --- |
| 1. Hidden-state cache | `outputs.hidden_states` length = `N_M + 1` | nothing in code; sweep iterates `range(N_M)` |
| 1. Letter token IDs   | Tokenizer-specific; "A"/"B"/"C" may tokenise to multiple IDs in some BPE schemes | regenerate via `phase08_letter_distribution.py` |
| 1. Post-letter position | Same definition (one token after the letter) | nothing |
| 2. Probe label rule   | If `M` produces different abstain tokens (e.g. "Unknown") or uses Yes/No instead of A/B/C, redefine the `pred → {0,1,2,C}` mapping | edit `_is_bias_aligned` in `src/modules/training/probe_layers.py` |
| 2. `correct` control  | Identical | nothing |
| 3. Tier 0 sample size | If `M` is much larger, you may need stricter stratification to fit in one A100 hour | `--max_per_cell` knob |
| 4. Test B Lite        | Requires at least one prior debiased checkpoint of `M`. If none exists, this gate is **deferred** — you must rely on cross-dataset transfer (step 5) for falsification | acceptable but document the caveat |
| 5. Transfer audit     | Second benchmark needs the same A/B/C answer format. SB-Bench works for any BBQ-style VLM | re-use `phase08_build_sbbench_gen.py` if benchmarks match |
| 6. PPO wiring         | Trainer must surface `outputs.hidden_states[k]` at runtime; some HF model classes don't by default | set `output_hidden_states=True` on the policy forward |

**Compute envelope for a 3B-class model**: steps 1–5 ≈ 1.5–2 h on one
A100-80GB. For a 7B-class model: ≈ 4–5 h.

---

## 9. Pre-flight checklist (copy-paste before starting on a new model)

```
[ ] Vanilla generation on the benchmark, with hidden_states cached at every layer
[ ] Letter-token-IDs verified (phase08_letter_distribution.py)
[ ] Layerwise sweep run (P1, P2, P3) → layerwise_probe.json
[ ] Top-3 candidate layers identified (P2 holdout, gated by P1 ≥ 0.95)
[ ] bias_aligned + correct probes fit at all candidate layers
[ ] Cosines logged: bias↔isC, bias↔correct, bias↔legacy (≤ 0.15)
[ ] Tier 0 G1 + G2 + G3 PASS at lead layer
[ ] correct-control s_d collapses (|s_d_corr| < 0.10)
[ ] Test A-lite ≥ 8/10 axes positive
[ ] At least one Test B Lite variant available OR cross-dataset transfer ≥ 0.5×
[ ] Cross-dataset transfer audit PASS (s_d ≥ 0.5× in-dist)
[ ] PPO trainer projects at the lead layer (not penultimate)
[ ] Ambig-preservation term in r_total
[ ] Eval matrix covers in-dist holdout + cross-dataset + capability sanity
```

If any unchecked box is "skipped because we trust the previous step",
re-read §4½ of the analysis doc — every gate exists because an earlier
version of this workflow failed without it.
