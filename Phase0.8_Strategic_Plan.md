# Phase 0.8 — Strategic Plan (complementary to `Phase0.8_Critical_Review.md`)

> **Purpose.** The critical review documents *what happened* (results, ablations,
> probe audits). This plan documents *what we do next, in what order, and under
> which decision rules* — so that future work is auditable against pre-committed
> gates rather than post-hoc rationalization.
>
> **Status snapshot (entering this plan, ORIGINAL).** Phase 0.8 PPO at L13 yields
> +1.85 pp on SB-Bench (p = 7.4e-12), |bias_score| −71% on VLBiasBench, ambig
> preserved. Reward is verified *synergistic*: corrOnly = −5.25 pp, biasOnly =
> −1.27 pp, full = +1.85 pp (interaction ≈ +8.4 pp). The L13 probe is a clean
> linear readout (holdout acc 0.972, plateau L9–L19). The 11-layer sweep
> (§4½.14) shows the bias direction is monotone-amplified across depth and the
> capability control is falsified at every layer.
>
> The headline therefore rests on **one model, one seed, one reward layer**.
> The plan below is designed to either (a) harden that headline into a causal
> claim with a noise floor, or (b) falsify it cheaply.

---

## 0bis. Status snapshot — REVISION 2026-06-20 (post-KLFIX, post-wash-out)

The original plan's two highest-value gates (T0#1 seed-replication, T0#3 wash-out)
have both closed. The headline now rests on a **4-seed, KL-stabilised** result with
a falsifier (wash-out) that came back **flat**. Cross-walk:

| # | Original status | Now | Evidence |
|---|---|---|---|
| **T0#1** seed × 3 L13 | open | ✅ **CLOSED — passed** | KLFIX 4-seed canonical Δ = **+1.08 pp ± 0.35 pp**; per-seed 0.626 / 0.632 / 0.633 / 0.628 (all > vanilla 0.619). Beats original "mean Δ ≥ 1.0 pp, std ≤ 0.7 pp" criterion. See [seed_diag/klfix/FINAL_REPORT.md](Phase0.8/seed_diag/klfix/FINAL_REPORT.md). |
| **T0#2** counterfactual | open | ⚠️ **DEMOTED** | n=228 underpowered for 3pp effects. Replaced by Action 3 (n=6166 build). See [REFRAMING_DECISION_v2.md](Phase0.8/REFRAMING_DECISION_v2.md) §5. |
| **T0#3** wash-out | open | ✅ **CLOSED — Pattern 2 across ALL variants** | Prod phase08_2k: max \|z\|=0.032σ. corrOnly: 0.075σ. biasOnly: 0.021σ. **KLFIX-s1: 0.019σ (flattest)**. No variant crosses the §3 ≤−0.30σ at L13 threshold. See [washout_klfix_verdict.md](Phase0.8/washout_klfix_verdict.md). |
| **T1#4** L17 reward seed×3 | gated | 🟡 **SUPERSEDED** | Subsumed by Action 1 ensemble (covers L17 + 4 deeper layers in a single objective). |
| **T1#5** multi-layer LoRA | gated on wash-out 2/3 | 🟡 **DEFERRED** | Pattern 2 authorises it, but offline ensemble evidence (Cohen-d 3.70 → 7.21) makes the **reward** lever cheaper-and-stronger. Tackle LoRA placement after Action 1 verdict. |
| **T1#6** multi-layer reward | gated on wash-out 3 only | 🟢 **PROMOTED to Tier 0** (= Action 1) | The flat wash-out + offline ensemble combination is stronger evidence than the original Pattern-3 gate envisioned. See §3 below. |

### Key mechanistic update — the L13 reward is a *selector*, not an *eraser*

Wash-out across four PPO variants gives max \|z\| ≤ 0.075σ at *every* layer — well
below the original ≤ −0.30σ Pattern-1 cut-off. KLFIX-s1 is the flattest
(0.019σ at L17). Interpretation:

- **The L13 probe reward IS producing accuracy gains** (+1.08 pp KLFIX,
  +1.85/+3.10 pp prod) **without measurably attacking the L13 bias direction.**
- The probe is acting as a **selection signal** that preferentially reinforces
  trajectories that *happen to* be less biased on the L13 axis. Those
  trajectories' bias improvement does **not** store itself back in that axis.
- This explains why corrOnly / biasOnly individually fail (§8.2 synergy) while
  full succeeds: only the *combination* of "correctness-weighted gradient" ×
  "bias-axis-selecting head" produces a non-noise update.
- **Bottleneck is reward-signal coverage, not training stability** (KLFIX gave
  us stability already). Adding more *kinds* of reward signal (different probe
  layers, ensembled) is the right next lever; adding more *data* of the same
  reward kind is not (see §7bis).

This is the empirical justification for re-ranking T1#6 → Action 1 below.

---

## 0. Guiding principles

1. **Cheap falsifiers before expensive amplifications.** Every multi-layer or
   multi-objective extension is gated on a sub-day diagnostic that can kill it.
2. **Pre-register decision rules.** Every experiment below has a written
   pass/fail gate *before* it runs.
3. **Causal > correlational.** A counterfactual eval (paired demographic swaps)
   outranks any new training run in expected reviewer value.
4. **Preserve the ambig channel.** No change is accepted that reduces
   `1[gold==C ∧ pred==C]` rate by more than 2 pp absolute on VLBias ambig.

---

## 1. Tiered priorities — REVISED 2026-06-20

> Original v1 tier table preserved in [archive/Phase0.8_Strategic_Plan_v1.md](Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md) §1; revised
> table below reflects T0#1 / T0#3 closure and the promotion of T1#6 → Action 1.

### Tier 0 — Live items (do next)

| # | Item                                              | Cost           | Decision rule                                                                                                                  |
|---|---------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------|
| **A1** | **Multi-layer ensemble reward (zmean-top5)** under KLFIX backbone | ½-day code + ~5 h GPU | Adopt as new baseline iff: (i) mean canonical Δacc vs vanilla ≥ +1.5 pp; (ii) cross-seed std ≤ 0.5 pp; (iii) beats KLFIX on ≥ 6/9 axes; (iv) tail-KL std ≤ 0.005 (matches KLFIX). See §3bis. |
| **A3** | **Counterfactual n=6166** on KLFIX-mean + variants | ½-day build + ~3 h GPU | Causal claim accepted iff `flip_rate(KLFIX) ≤ 0.7 × flip_rate(vanilla)` AND `accuracy_swap − accuracy_orig` within ±2 pp. See §4 (refreshed). |

### Tier 0 — Closed items (do not re-run)

| # | Original gate | Status |
|---|---|---|
| T0#1 seed × 3 L13 | passed | KLFIX 4-seed +1.08 pp ± 0.35 pp |
| T0#3 wash-out diagnostic | Pattern 2 across 4 variants | See §3 (refreshed) |
| A2 (was Action 2) KLFIX-s1 wash-out | Pattern 2 (max \|z\|=0.019σ) | Confirms prod result on KLFIX backbone |

### Tier 1 — Conditional on Action 1 verdict

| # | Item                                  | Trigger                                                                                  |
|---|---------------------------------------|------------------------------------------------------------------------------------------|
| T1#5 | Multi-layer LoRA (L13/17/21/25)    | Run **after** Action 1 only if ensemble passes its gate. Tests whether placement of the *store* (LoRA) also matters once the *signal* (reward) is fixed. |
| T1#4 | L17-only reward seed×3              | **Superseded** by Action 1. Skip. |
| Scaling A1 (2-epoch) | open                       | Promoted to Tier 0 only if Action 1 fails its gate (tripwire #1 in [SCALING_PRIORITY_ANALYSIS.md](Phase0.8/SCALING_PRIORITY_ANALYSIS.md) §3). |
| Scaling A2 (8k corpus) | open                     | Same tripwire as A1. |
| Scaling A3 (2-stage)   | open                     | **Skip** — high downside risk (destabilises proven recipe), low EV. |

### Tier 2 — Generalization (post-Tier-0)

Unchanged from v1: vision-tower probe (T2#7), OOD eval (T2#8), reward-head
distillation (T2#9). T2#8 brought forward to immediately after Action 1 if it
passes (provides OOD evidence for the new baseline).

> Full v1 tier table preserved in [archive/Phase0.8_Strategic_Plan_v1.md](Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md) §1.

---

## 2. Decision tree — REVISED 2026-06-20

The original v1 decision tree (preserved in [archive/Phase0.8_Strategic_Plan_v1.md](Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md) §2) was authored *before* the
wash-out diagnostic returned. The diagnostic returned **Pattern 2 across all
4 PPO variants** (max |z| ≤ 0.075σ), so the live branch is determined.

```
                              ┌────────────────────────────────────┐
                              │ A1: Multi-layer ensemble reward     │
                              │     (KLFIX backbone, zmean-top5)    │
                              └────────────────┬───────────────────┘
                                               │
              ┌────────────────────────────────┼────────────────────────────────┐
              ▼                                ▼                                ▼
   PASSES gate (§3bis)               INTERMEDIATE                       FAILS gate
   Δacc ≥ +1.5pp, std ≤ 0.5pp        +0.5 ≤ Δacc < +1.5pp                Δacc < +0.5pp
              │                                │                                │
   ┌──────────┴──────────┐         ┌──────────┴──────────┐         ┌────────────┴───────────┐
   │ Adopt as new        │         │ Keep KLFIX-L13      │         │ Hold KLFIX-L13.        │
   │ baseline. Run       │         │ as baseline. Run    │         │ Promote Scaling A1     │
   │ T1#5 (multi-LoRA)   │         │ T2#8 (OOD) + A3     │         │ (2-epoch) and A2 (8k)  │
   │ + T2#8 (OOD) +      │         │ (CF n=6166).        │         │ per SCALING_PRIORITY   │
   │ A3 (CF n=6166).     │         │ Document negative   │         │ tripwire #1.           │
   └─────────────────────┘         │ ensemble result.    │         └────────────────────────┘
                                   └─────────────────────┘
```

Action 3 (CF n=6166) runs **in parallel** with Action 1 — it's diagnostic, not
gated, and required for the causal-debias claim regardless of A1 outcome.

> Original v1 decision tree (wash-out-conditioned three-way branch) preserved in [archive/Phase0.8_Strategic_Plan_v1.md](Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md) §2. Empirically all 4 PPO variants returned sub-threshold (max |z| ≤ 0.075σ), so the v1 "Pattern 1/2/3" branching never bound. The mechanistic implication (§0bis) is the L13 reward is a **selector**, not an **eraser** — which is why we promoted T1#6 (multi-layer reward) over T1#5 (multi-layer LoRA): the right fix is more reward *signal*, not more *store* capacity.

---

## 3. Wash-out diagnostic — RESULTS (was: spec)

> Original v1 spec preserved in [archive/Phase0.8_Strategic_Plan_v1.md](Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md) §3. Diagnostic now CLOSED.

**Method (as run).** Per-layer probe scoring on 2000 VLBias records, frozen
base-fit `bias_aligned` probe at each L ∈ {1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35}.
Compared four PPO variants (prod phase08_2k, corrOnly, biasOnly, KLFIX-s1) against
vanilla. Δμ_corr per layer and (variant − base) / σ_van.

**Results.**

| variant                | pattern | max \|z\| | argmax L |
|------------------------|---------|----------|----------|
| phase08_2k (prod)      | Pattern 2 | 0.032σ | 11 |
| phase08_2k_corrOnly    | Pattern 2 | 0.075σ | 25 |
| phase08_2k_biasOnly    | Pattern 2 | 0.021σ | 25 |
| **phase08_2k_klfix_s1**| **Pattern 2** | **0.019σ** | **17** |

**Per-layer (klfix-s1 − base) / σ_van** (illustrative — full table in [washout_diagnostic_klfix.json](Phase0.8/washout_diagnostic_klfix.json)):
```
L 1  +0.008    L13 +0.016    L25 -0.009
L 5  +0.015    L17 +0.019    L29 -0.013
L 9  +0.012    L21 +0.019    L33 -0.006
L11  +0.016                  L35 -0.000
```

**Interpretation.** All four variants are sub-threshold (well below ±0.10σ
everywhere). The L13 PPO is **not measurably attacking the L13 bias
representation**, yet still delivers +1.08–3.10 pp accuracy gains. The reward
is functioning as a **selector** over already-existing trajectory variation,
not as a representational **eraser**. KLFIX-s1 is the flattest variant — the
KL-stability fix successfully removed the noise floor but did **not** add new
representational signal. This validates the Action 1 thesis: the bottleneck
is reward-signal coverage, not training stability.

**Artifacts.**
- [Phase0.8/washout_klfix_verdict.md](Phase0.8/washout_klfix_verdict.md) — short verdict + decision implication
- [Phase0.8/washout_diagnostic_klfix.json](Phase0.8/washout_diagnostic_klfix.json) — full per-layer z-scores
- [Phase0.8/washout_scores/](Phase0.8/washout_scores) — 11 per-layer per-variant offline-reward JSONs
- [scripts/phase08_klfix_washout.sh](scripts/phase08_klfix_washout.sh) — launcher (STEP=1 gen, STEP=2 score)
- [scripts/phase08_washout_classify.py](scripts/phase08_washout_classify.py) — classifier

> Original v1 wash-out spec (Pattern 1/2/3 method, Question, Implementation) preserved in [archive/Phase0.8_Strategic_Plan_v1.md](Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md) §3.

---

## 3bis. Action 1 — Multi-layer ensemble reward (new live spec)

**Question.** Does z-score-normalised mean of probe rewards across multiple
optimal-band layers (L ∈ {9, 11, 13, 17, 21}, or the offline-tested top-5
L ∈ {17, 21, 25, 29, 33}) deliver a stronger and more stable accuracy gain
than single-layer L13 reward, on top of the KLFIX backbone?

**Offline evidence supporting the experiment.**
- Cohen-d on the bias-axis projection rises from **3.70 (single-layer L13)** to
  **7.21 (zmean-top5)**, a +95% gain.
- Sign-consistency: **10/10 BBQ axes positive** at the ensemble vs 7-8/10 single
  layer.
- Reward scale stays in the ±0.2σ band (no rank bloat) under z-score
  normalisation.
- Per-layer Pearson correlation 0.87–0.99 across the top window → ensemble
  averages out per-layer noise rather than introducing new noise.

**Implementation gaps to close.** Three files + one launcher:
1. [src/modules/training/drm_loader.py](src/modules/training/drm_loader.py):
   load **N** probe heads keyed by layer; store as a `dict[int, nn.Linear]`.
2. [src/modules/rl_components/phase08_ppo_trainer.py](src/modules/rl_components/phase08_ppo_trainer.py)
   (or equivalent reward call-site): single forward pass capturing hidden
   states at all N layers; per-layer z-score normalisation; sum/mean pooling
   to a scalar reward.
3. [src/modules/training/args.py](src/modules/training/args.py): new flag
   `--reward-mode bias_aligned_ensemble`, or extend `--reward-head-layer` to
   accept a comma-separated list with a `--reward-pool {zmean,mean,max}` flag.
4. New launcher under `scripts/phase08_ensemble_*.sh` mirroring
   `scripts/phase08_klfix_*.sh` structure.

**Recipe (inherits from KLFIX).** Same hyperparameters as the
seed-replication runs (lr=5e-6, lora_r=16, α=32, batch=8, max_gen=8,
max_train=2000, target_kl=0.02, kl_β=0.1, w_corr=1.0, w_bias=1.0, w_ambig=0.5,
KLFIX bundle: value-head zero-init + value-clip-range=0.2 + kl-adapt-rate=0.3).
Differs only in reward source: ensemble probe heads instead of single L13.

**Seed plan.** 2-seed smoke run first (seeds 1, 2). If smoke is green,
4-seed full sweep (seeds 1, 2, 3, 4) in parallel.

**Pre-committed gate.**
- Mean canonical-acc Δ vs vanilla ≥ **+1.5 pp** (beats KLFIX +1.08 pp by ≥ 0.4 pp).
- Cross-seed std ≤ **0.5 pp** (matches or improves KLFIX's 0.35 pp).
- Per-axis: beats KLFIX on ≥ **6/9** BBQ axes.
- KL stability: tail-KL std ≤ **0.005** (matches KLFIX).
- Ambig channel preserved: VLBias ambig `1[gold==C ∧ pred==C]` rate
  within −2 pp of vanilla.

**Expected outcome (priors).** If offline +95% Cohen-d transfers at any
non-trivial rate, KLFIX +1.08 pp should rise to +2.0 to +3.0 pp on canonical
n=2916, with cross-seed std staying ≤ 0.5 pp (ensemble averages out per-layer
noise).

**Cost.** ½-day code + 1 h smoke GPU + ~4 h full sweep GPU (4 seeds parallel)
+ canonical gens.

**Tripwires (auto-stop conditions).**
- Reward variance per step > 5× KLFIX baseline → suspect z-score normalisation
  bug; halt and inspect.
- Tail-KL drifts > 0.03 → KLFIX bundle no longer sufficient at ensemble scale;
  document and reduce KL target.
- Per-step reward magnitude > ±2σ on smoke → ensemble pool mode wrong (try
  `mean` instead of `zmean`).

**Output artifacts.**
- `Phase0.8/ensemble/seed_{1..4}/...` (training logs, final checkpoints)
- `Phase0.8/ensemble/canonical_eval_summary.json`
- `Phase0.8/ensemble/FINAL_REPORT.md` mirroring `seed_diag/klfix/FINAL_REPORT.md`

---

## 4. Counterfactual flip-rate eval — Action 3 (refreshed spec)

> **Status update.** The original n=228 CF set was run on prod phase08_2k but
> declared underpowered for ~3 pp effects in [REFRAMING_DECISION_v2.md](Phase0.8/REFRAMING_DECISION_v2.md) §5.
> Action 3 (this section) replaces it with the n=6166 full build on the
> KLFIX-mean checkpoint + variants.

**Question.** Does the KLFIX model change its answer when we swap the
demographic referent in a question (e.g. *grandfather* ↔ *grandson*,
*husband* ↔ *wife*) while holding image and frame fixed? A *bias-free* model
should flip its answer at the rate dictated by the gold answer alone; a
biased model flips much more.

**Construction (n=6166 — was n=228).**
1. Filter SB-Bench to the unambig (non-C) subset (gold ∈ {A, B}). Full subset,
   not the n=228 stratified pilot.
2. For each question, parse the two named entities `e1` and `e2` referenced by
   the answer choices A and B (SB-Bench encodes these as templated swap pairs
   already; cross-reference the unpacked SB-Bench schema).
3. Generate the *swapped* prompt by exchanging `e1` and `e2` in the question
   text (and the answer-choice ordering, so gold is preserved at the *referent*
   level rather than the *letter* level).
4. Pair `(orig_prompt, swap_prompt)` keyed by `qid`. Expected n ≈ 6166 pairs.

**Metrics (per model).**
- `flip_rate` = fraction of pairs where `model(orig) != letter_of_swap(model(swap))`
   under the referent re-mapping (i.e. did the answer change *meaning*?).
- `accuracy_orig` and `accuracy_swap`, individually.
- McNemar on (correct_orig vs correct_swap) — symmetry test.
- **Per-axis flip-rate** (mirroring the per-axis canonical accuracy story).

**Variants to run.** vanilla, KLFIX-mean (best of 4 seeds), Action-1 ensemble
best seed (if A1 passes), corrOnly (prod), biasOnly (prod). Five jobs × ~30 min
A100 each.

**Decision rule (causal claim).**
KLFIX (or ensemble) accepted as causally debiased iff:
- `flip_rate(target) ≤ 0.7 × flip_rate(vanilla)` AND
- `accuracy_swap(target) − accuracy_orig(target)` within ±2 pp (no asymmetric collapse) AND
- corrOnly does NOT meet the first criterion (controls for "any RL helps").

**Tripwire (links to SCALING_PRIORITY_ANALYSIS §3 #3).** If flip-rate Δ < 5 pp
in absolute terms despite the gate passing, headline reads as "small causal
effect" and promotes scaling A1 (2-epoch) as a remediation.

**Output artifact.** `Phase0.8/counterfactual_eval/cf_pairs_n6166.parquet` and
`cf_metrics_n6166.json`.

---

## 5. Seed × 3 spec (CLOSED — superseded by KLFIX 4-seed)

> **Status update.** This section is closed. KLFIX delivered a **4-seed**
> result that exceeds every pass criterion below. The L17-vs-L13 head-to-head
> is superseded by Action 1 (the ensemble covers L17 + 4 deeper layers).

**Result (KLFIX 4-seed, canonical n=2916).**
- Mean Δacc = **+1.08 pp ± 0.35 pp** (vs vanilla 0.619).
- Per-seed: 0.626 / 0.632 / 0.633 / 0.628 — all > vanilla, gate met 4/4.
- Per-axis: beats vanilla on 8/9 BBQ axes; beats vwarmup on 9/9.
- KL stability: tail-KL std 0.0028 (≤ 0.005 target).
- Full report: [seed_diag/klfix/FINAL_REPORT.md](Phase0.8/seed_diag/klfix/FINAL_REPORT.md).

Original recipe (lr=5e-6, lora_r=16, α=32, batch=8, max_gen=8, max_train=2000,
target_kl=0.02, kl_β=0.1, w_corr=1.0, w_bias=1.0, w_ambig=0.5) was extended
with the KLFIX bundle: value-head zero-init + value-clip-range=0.2 +
kl-adapt-rate=0.3. That bundle is now the canonical PPO recipe and is inherited
by Action 1.

**Output artifact.** `Phase0.8/seed_diag/klfix/` (4 seeds × training logs +
canonical eval) + `Phase0.8/seed_diag/klfix/FINAL_REPORT.md`.

---

## 6. Execution timeline — REVISED 2026-06-20

> Original v1 5-day timeline preserved in [archive/Phase0.8_Strategic_Plan_v1.md](Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md) §6. Revised 3-day plan
> below replaces it.

```
Day 1 (now, in parallel):
  - Implement multi-layer ensemble reward in drm_loader + ppo_trainer.
    Files: src/modules/training/drm_loader.py,
           src/modules/rl_components/phase08_ppo_trainer.py,
           src/modules/training/args.py.
    Cost: ~3-4 h of code. [Action 1, step 1-3]
  - Begin CF n=6166 pair build (parallel to code work).
    Cost: ~½ day of build + 50-sample hand-audit. [Action 3, build phase]

Day 2:
  - 2-seed smoke run of ensemble reward (seeds 1, 2). ~1 h GPU.
  - If smoke green: launch 4-seed full sweep + canonical eval.
    ~4 h GPU (parallel) + ~30 min canonical gens × 4.
  - In parallel: launch CF n=6166 jobs on vanilla, KLFIX-mean, corrOnly,
    biasOnly. 4 × ~30 min GPU.

Day 3:
  - Harvest Action 1 + Action 3 outputs.
  - Run pre-committed gates (Action 1 §3bis, Action 3 §4).
  - Decision branch (§2 revised tree):
      A1 passes → adopt ensemble as new baseline; queue T1#5 (multi-LoRA)
                  + T2#8 (OOD).
      A1 intermediate → keep KLFIX-L13; queue T2#8 + A3.
      A1 fails → hold KLFIX-L13; promote Scaling A1 (2-epoch) per
                 SCALING_PRIORITY tripwire #1.
  - Append §11 (ensemble) and §12 (n=6166 CF) to critical review.
```

> Original v1 5-day timeline preserved in [archive/Phase0.8_Strategic_Plan_v1.md](Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md) §6.

---

## 7. What we are explicitly **not** doing next — REVISED 2026-06-20

These are *deferred* with reasons, so we do not lose them:

- **PPO scale-up beyond 2k (Scaling A1/A2/A3, Critical Review Direction A)** —
  ranked **#4 of 4** in [SCALING_PRIORITY_ANALYSIS.md](Phase0.8/SCALING_PRIORITY_ANALYSIS.md).
  Trajectory decelerates (+1.03 → +0.79 → +0.17 pp across quartiles), probe
  sits in L9–L19 optimal plateau, synergy is saturated at 2k. Tripwires
  (§3 of that memo) that would promote it back: Action 1 failure, wash-out
  Pattern shift, or CF flip-rate Δ < 5 pp.
- **DPO / IPO replacement of PPO** — orthogonal to the question of "does the
  reward signal point at the right thing"; the answer to that is what the
  wash-out + Action 1 ensemble + Action 3 CF tests deliver.
- **Reward-model retraining at deeper layers** (e.g. L25 single-layer) —
  superseded by Action 1 (ensemble covers L17/21/25/29/33 in one objective).
- **VLBiasBench retraining** — VLBias stays as the *transfer* set. Training on
  it would dissolve the strongest piece of evidence we have.
- **Switching to the 7B model** — capacity is not the current bottleneck.
  Wash-out shows the bottleneck is reward-signal *coverage*, not policy
  capacity. 7B would inherit the same selector-not-eraser pathology.
- **2-stage training (Scaling A3, epoch 1 full → epoch 2 bias-only)** —
  high downside risk (destabilises the proven KLFIX recipe), low EV. Skip
  unconditionally.

---

## 8. Risk register — REVISED 2026-06-20

| Risk                                                                  | Mitigation                                                                |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------|
| Action 1 ensemble Δacc ≤ KLFIX baseline (+1.08 pp)                    | Hold KLFIX-L13 as baseline; promote Scaling A1 per tripwire #1; write up ensemble as negative result. |
| Action 1 introduces reward-magnitude / KL instability at ensemble scale | Stop on tripwires (§3bis); fall back to `mean` pool or reduce N from 5 to 3 (top L17/21/25). |
| CF n=6166 flip-rate Δ < 5 pp absolute despite gate passing            | Read headline as "small causal effect"; promote Scaling A1; reword paper claim. |
| Action 1 passes Δacc gate but ambig collapses (1[gold==C∧pred==C] drops > 2 pp) | Reject ensemble; investigate per-class reward asymmetry; reduce w_bias. |
| McNemar `qid` schema bug recurs                                       | Patch `scripts/phase0_mcnemar.py` to row-index pair always.               |
| Multi-layer LoRA (T1#5, post-A1) blows up memory                      | Fall back to LoRA on subset {L13, L21}; rank 8.                           |
| KLFIX backbone destabilises under ensemble reward scale               | Verify KLFIX bundle still applies; re-tune kl-adapt-rate (0.3 → 0.5) and value-clip-range. |

---

## 9. Pointers (single source of truth) — REVISED 2026-06-20

**PPO checkpoints (Modal volume `debias-vlm-persistent-storage`):**
- KLFIX seeds 1–4: `/mnt/data/output_ppo_phase08_2k_L13_s{1,2,3,4}_klfix/final_debiased_model` ← **NEW BASELINE**
- Prod (single-seed, pre-KLFIX): `/mnt/data/output_ppo_phase08_2k/final_debiased_model`
- corrOnly: `/mnt/data/output_ppo_phase08_2k_corrOnly/final_debiased_model`
- biasOnly: `/mnt/data/output_ppo_phase08_2k_biasOnly/final_debiased_model`

**Probe heads (`sb_bench-PROBE-component`):**
- L13 single-layer: `/mnt/data/generated_heads_probe_L13_base_biasA/sb_bench-PROBE-component/sb_bench-PROBE-component0.pth`
- 11-layer sweep: `/mnt/data/generated_heads_probe_L{1,5,9,11,13,17,21,25,29,33,35}_base_biasA/sb_bench-PROBE-component/`
- Per-layer probe weights (npz): `Phase0.8/a3_results/base_L{N}_probe_weights_biasA.npz`
- **TODO (Action 1):** ensemble head bundle at `/mnt/data/generated_heads_probe_L{9,11,13,17,21}_base_biasA/` or `L{17,21,25,29,33}_base_biasA/` — pick one in §3bis-step-1.

**Baselines / evals:**
- Vanilla SB-Bench gens: `/mnt/data/sb_bench_vanilla_baseline/sb_bench_generations.jsonl`
- Vanilla VLBias gens: `/mnt/data/phase07_vlbiasbench/base_vlbias_gen.jsonl`
- KLFIX-s1 VLBias gens: `/mnt/data/phase07_vlbiasbench/phase08_2k_klfix_s1_vlbias_gen.jsonl`
- Wash-out scores (4 variants × 11 layers): `/mnt/data/phase08_washout/{variant}__probe_L{N}__offline_reward.json` (also synced locally to `Phase0.8/washout_scores/`).

**Memos & analyses (this branch):**
- Critical review: `Phase0.8_Critical_Review.md`
- This strategic plan: `Phase0.8_Strategic_Plan.md` (you are here)
- Post-KLFIX strategic synthesis: [Phase0.8/NEXT_STEPS.md](Phase0.8/NEXT_STEPS.md)
- Scaling-priority ranking (Direction A vs Actions 1-3): [Phase0.8/SCALING_PRIORITY_ANALYSIS.md](Phase0.8/SCALING_PRIORITY_ANALYSIS.md)
- KLFIX 4-seed report: [Phase0.8/seed_diag/klfix/FINAL_REPORT.md](Phase0.8/seed_diag/klfix/FINAL_REPORT.md)
- KLFIX wash-out verdict: [Phase0.8/washout_klfix_verdict.md](Phase0.8/washout_klfix_verdict.md)
- Wash-out diagnostic JSON: [Phase0.8/washout_diagnostic_klfix.json](Phase0.8/washout_diagnostic_klfix.json)
- Multi-layer probe analysis: `Phase0.8/Multi_layer_head_analysis.md` §4½.14
- Reframing rationale (CF demoted): [Phase0.8/REFRAMING_DECISION_v2.md](Phase0.8/REFRAMING_DECISION_v2.md)

**Scripts:**
- KLFIX wash-out runner: [scripts/phase08_klfix_washout.sh](scripts/phase08_klfix_washout.sh)
- Wash-out classifier: [scripts/phase08_washout_classify.py](scripts/phase08_washout_classify.py)
- CF pair builder (n=228 → upgrade to n=6166 for A3): [scripts/phase08_build_counterfactual_pairs.py](scripts/phase08_build_counterfactual_pairs.py)

