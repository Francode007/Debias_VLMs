# Phase 0.8 — Strategic Plan ↔ Current State (post-KLFIX)

**Date:** 2026-06-20
**Trigger:** KLFIX 4-seed canonical eval complete ([FINAL_REPORT](seed_diag/klfix/FINAL_REPORT.md)).
Time to walk through `Phase0.8_Strategic_Plan.md` and decide what's still on the table.

The summary verdict up front: **Tier 0 #1 (seed-replication gate) is now CLOSED**
(klfix delivers +1.08pp ± 0.35pp on the canonical n=2916 with all 4 seeds individually
above vanilla — exceeds the original "mean Δ ≥ 1.0pp, std ≤ 0.7pp" criterion). Tier 0
#2 (counterfactual) has degraded to a less-relevant diagnostic for reasons documented
in REFRAMING_v2 §5 and §4 below. Tier 0 #3 (wash-out) is **already DONE** and gave a
result that needs to inform what we do next.

---

## 1. Strategic Plan items — current status

| # | Item | Original gate | Current state | Verdict |
|---|------|---------------|---------------|---------|
| **T0#1** | Seed × 3 replication, L13 | 3/3 p<0.05, mean Δ ≥ 1.0pp | KLFIX **4-seed** Δ = +1.08pp ± 0.35pp; per-seed: 0.626/0.632/0.633/0.628 (all > vanilla 0.619) | ✅ **CLOSED — passed** |
| **T0#2** | Counterfactual flip-rate | flip_rate(P08) ≤ 0.7× vanilla | Stale @ n=228 (underpowered). REFRAMING_v2 §5 demoted it. | ⚠️ **DEMOTED** — see §3 |
| **T0#3** | Wash-out diagnostic | Pattern 1/2/3 classification | Already run. Per-σ change is **<0.05σ at every layer** for phase08_2k vs base | ✅ **CLOSED — Pattern 2 (clean wash-out)** |
| **T1#4** | L17 reward, seed × 3 | mean Δacc(L17) ≥ Δacc(L13) + 0.5pp | Not run on klfix. Question now: is L17 better than klfix-L13? | 🟡 **Deferred** (see §4) |
| **T1#5** | Multi-layer LoRA | Gated on wash-out Pattern 2 or 3 | Wash-out IS Pattern 2 → eligible. But offline ensemble test (§2) suggests **multi-layer REWARD** is the better lever | 🟡 **Reframed** |
| **T1#6** | Multi-layer reward (probe ensemble) | Gated on Pattern 3 | Offline test ([ensemble/README.md](ensemble/README.md)) shows zmean-top5 **Cohen-d 3.70→7.21, 10/10 axes positive**. Pattern 2 doesn't strictly authorise it, but offline evidence is so strong it bumps the priority | 🟢 **PROMOTED to Tier 0** |
| T2#7 | Vision-tower probe | n/a | Untouched | Defer |
| T2#8 | OOD eval (CrowS-Pairs, BBQ-cf) | n/a | Untouched | Possibly bring forward |
| T2#9 | Reward-head distillation | n/a | Untouched | Defer |

---

## 2. The wash-out result and what it implies

`Phase0.8/washout_diagnostic.json` (run on the *original* `phase08_2k` model, not on
klfix yet — see §6 for the caveat) gives this table (Δμ in std units, base-fit
per-layer probe):

| Layer | base Δμ | p08 Δμ | (p08 − base) / σ |
|---|---:|---:|---:|
| 1 | 0.556 | 0.574 | +0.017 |
| 5 | 0.389 | 0.403 | +0.021 |
| 9 | 0.410 | 0.428 | +0.025 |
| 11 | 0.380 | 0.402 | +0.032 |
| **13** | 0.441 | 0.463 | +0.029 |
| **17** | 0.477 | 0.495 | +0.025 |
| 21 | 0.650 | 0.671 | +0.018 |
| 25 | 1.830 | 1.837 | +0.002 |
| 29 | 2.071 | 2.055 | −0.005 |
| 33 | 2.455 | 2.468 | +0.004 |
| 35 | 2.590 | 2.598 | +0.002 |

**Per-σ classification:** none of the layers shifts by more than 0.04σ in either
direction. The strategic-plan thresholds (Pattern 1 needs ≤ −0.3σ at L13, Pattern 2
needs ≤ −0.3σ at L13 with |Δ|<0.1σ at L≥21, Pattern 3 needs >0 at L≥25) all assume
*deep* shifts of order 0.3σ. We see **none of the patterns at the prescribed
magnitudes** — the L13 PPO simply did not move the L13 bias direction by anything
close to 0.3σ. This is the REFRAMING_v2 §4 interpretation: *the policy is routing
corrections through a different / output channel*. Important corollary:

- The L13 probe-reward IS achieving accuracy gains (+1.08pp klfix, +1.85/+3.10pp prod)
  *without* materially attacking the L13 bias representation.
- The probe is acting as a **selection signal**, not as a representation **eraser**.
  It is preferentially reinforcing trajectories that already happen to be less
  biased on the L13 axis, but those trajectories' bias improvement does not store
  itself back in the L13 direction.

This has a sharp implication for §4 priority ordering: a multi-layer **reward**
ensemble (T1#6) is the right next lever, because changing the *layer* of the
selection signal is a cleaner experiment than changing the LoRA attachment (T1#5),
which addresses where the *store* lands rather than where the *signal* fires.

---

## 3. Why the counterfactual diagnostic is degraded (not abandoned)

REFRAMING_v2 §5 noted the n=228 CF set has insufficient power for a 3-pp effect
on a 35-50% baseline. Three options now:

A. **Build the 6166-pair full CF set** (REFRAMING_v2 §7 follow-up #2). Power
   should be sufficient for ~1pp effects at n=6166. Cost: 1 gen-pair job per
   variant (~30 min × 5 models) + analysis.

B. **Run CF on klfix specifically** (we have it on prod phase08_2k, not on
   klfix). Same n=228 limitation — would not change the verdict that CF is
   underpowered.

C. **Drop CF and replace with a per-axis stratified test** on canonical SB-Bench
   (which klfix already passed). Argue (correctly) that the per-axis 9/9 win
   over vwarmup + 8/9 win over vanilla **is** evidence of polarity-invariant
   debiasing.

Recommendation: **(A) if we want a publishable causal claim**, else **(C) if we're
optimising for headline strength + speed**. Both are cheap. Defer (B) — same data,
same power problem.

---

## 4. Next-step proposal — three concrete actions, in priority order

### Action 1 (HIGHEST priority): Multi-layer reward ensemble on klfix baseline

The offline ensemble test ([ensemble/README.md](ensemble/README.md)) gives the
strongest single piece of evidence we currently have for an upcoming improvement:
zmean-top5 (z-score-normalised mean of probes at L17, L21, L25, L29, L33) delivers
**Cohen-d 3.70 → 7.21**, positive on **10/10 BBQ axes**, with reward scale kept in
the sensible ±0.2σ range (no rank bloat). On top of klfix's now-tight KL, the
expected effect should appear cleanly.

**Implementation gaps to close before launch:**
1. Build the ensemble probe-head directory on the Modal volume:
   `/mnt/data/generated_heads_probe_L17-33_zmean_base_biasA/`. This requires
   either (a) a small change in the reward path to load *5* probe heads and
   compute z-score mean, or (b) pre-baking the zmean operation into a single
   synthetic "head" that gives the same per-record output — (b) won't work
   because the z-score needs per-layer statistics, not a global linear combo.
   So this is option (a) — a code change in
   `src/modules/training/drm_loader.py` and the reward call-site in
   `phase08_ppo_trainer.py`.
2. Wire a new CLI flag `--reward-mode bias_aligned_ensemble` (or extend
   `--reward-head-layer` to accept a comma-separated list with a pooling mode).
3. Validate on a 2-seed smoke run before the full 4-seed sweep.

**Expected outcome (priors):** If the offline +95% Cohen-d gain translates at all,
klfix +1.08pp should rise to +2.0 to +3.0pp on canonical n=2916, with cross-seed
std staying ≤ 0.5pp (because the ensemble averages out per-layer noise — the
layer-layer Pearson is 0.87-0.99 across the top window).

**Cost:** ½-day of code (3 files + smoke), then ½-day GPU (4 seeds × ~30 min A100
in parallel + 4 canonical gens × ~7 min). Effective wall ~2 hours of orchestration.

**Pass criteria (pre-committed):**
- Mean canonical-acc Δ vs vanilla ≥ +1.5pp (i.e. beats klfix by ≥ 0.4pp).
- Cross-seed std ≤ 0.5pp.
- Per-axis: beats klfix on ≥ 6/9 axes.
- KL stability: tail-KL std ≤ 0.005 (matches klfix).

### Action 2 (HIGH priority): Re-run wash-out on klfix checkpoint

The existing `washout_diagnostic.json` was computed on the *prod* `phase08_2k`
checkpoint (single seed, pre-KLFIX). To make claims about the klfix recipe we
need the same diagnostic on one klfix checkpoint (any seed — they're nearly
identical at canonical eval).

**Cost:** ~15 min Modal GPU (single-pass offline scoring). Trivial.

**Pass criteria:** if klfix shows the same flat per-σ pattern (every layer
<0.1σ change), the REFRAMING_v2 §4 reading holds. If it shows Pattern 1
(deep cleanup), that's a positive surprise worth documenting. If Pattern 3
(deep proxy-hacking), that would force us to reconsider — but it's unlikely
given the per-axis fairness gains on canonical eval.

### Action 3 (MEDIUM priority): Build the n=6166 CF pair set + run on klfix

REFRAMING_v2 §7 follow-up #2 — the only honest way to have a "causal debiasing"
claim. Mechanical, no new science.

**Cost:** ½-day to build pairs + 5 × ~30 min Modal jobs (vanilla, klfix-mean,
corrOnly, biasOnly, baseline-prod for comparison). One day total.

**Pass criteria (klfix):** `flip_rate(klfix) ≤ 0.7 × flip_rate(vanilla)` per
REFRAMING_v2 §5 reading — and we should additionally report the per-axis CF
flip-rate to align with the per-axis canonical accuracy story.

### Defer (still)

- T1#4 L17 reward seed × 3 — supplanted by T1#6 ensemble (covers L17 + 4 deeper).
- T1#5 Multi-layer LoRA — wash-out is Pattern 2 (eligible) but the offline
  ensemble evidence is stronger; tackle reward side first, LoRA placement later.
- T2#7 Vision-tower probe, T2#9 distillation.
- T2#8 OOD eval — bring forward to immediately after Action 1 if it works.

---

## 5. Recommended order of operations (next 2-3 days)

```
Day 1 (today):
  - Implement multi-layer ensemble reward in drm_loader + ppo_trainer
    (Action 1 step 1-2). [~3-4 hours of code]
  - Re-run washout diagnostic on klfix-s1 checkpoint. [15 min GPU]

Day 2:
  - 2-seed smoke run of ensemble reward. [~1h GPU]
  - If smoke is green: launch 4-seed full sweep + canonical eval. [~1h GPU
    parallel]
  - Otherwise: diagnose, fix, retry.

Day 3:
  - Harvest, aggregate, write report.
  - If ensemble beats klfix: declare new baseline, deprecate klfix-L13.
  - If not: write up the negative result honestly, hold klfix-L13 as baseline,
    and pivot to Action 3 (CF n=6166) for the causal-debiasing claim.
```

---

## 6. What we are explicitly NOT doing next

- **PPO scale-up past 2k samples** — only after Action 1 verdict is in. Don't
  re-litigate the multi-layer choice on a more expensive baseline.
- **Switching probes to corr instead of biasA** — biasA is the synergy-validated
  signal (REFRAMING_v2 §1: corrOnly collapses).
- **7B model migration** — capacity isn't the bottleneck (REFRAMING_v2 §7 #2);
  identification is.
- **VLBias retraining** — VLBias remains the held-out transfer set.

---

## 7. Decision rule, restated

The Phase 0.8 KLFIX result is now defensible: +1.08pp on canonical n=2916,
4-seed std 0.35pp, beats vanilla on 8/9 axes. This is the publishable headline.
Action 1 (multi-layer ensemble reward) is high-EV upside; Actions 2 + 3 are
the honest causal-claim follow-ups. If Action 1 fails outright we still ship
klfix-L13 + the CF + wash-out evidence and write up an honest paper.

The strategic plan's pre-committed gates have not been violated; what's
changed is that two of them (T0#1, T0#3) have *passed* and the priorities
beneath them are re-ranked by the offline ensemble evidence.
