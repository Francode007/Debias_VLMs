# Phase 0.8 — Scaling-up ranking vs Action 1/3

**Date:** 2026-06-20
**Context:** Phase0.8_Critical_Review §8.5 / §8.6 proposed Direction A
("scale up the working recipe") as the recommended next move. That memo
was written when `phase08_2k_full` was the only stable point. Since then:
- KLFIX delivered a clean 4-seed +1.08pp ± 0.35pp on canonical (FINAL_REPORT).
- Wash-out diagnostic shows the L13 PPO recipe doesn't touch the L13 bias
  representation (Pattern 2; flat <0.05σ per layer).
- Offline ensemble test shows zmean-top5 Cohen-d 3.70→7.21 (10/10 axes).

So Direction A's *menu* has not changed but its *priority* relative to the
multi-layer ensemble (Action 1) and the n=6166 CF (Action 3) needs to be
re-decided in light of those three new pieces of evidence.

---

## 1. The Direction A menu (4 sub-experiments)

| # | Sub-experiment | Cost (Modal A100) | Information per $ |
|---|----------------|-------------------|-------------------|
| A1 | 2-epoch full reward on the same 2k corpus | ~1 h | Low-medium — KLFIX is already at near-asymptote (§7.1 trajectory was monotone, +0.17pp 74%→100%) |
| A2 | 8k corpus, 1 epoch | ~3 h | Medium — tests data-axis scaling, but the L13 recipe is plateauing per wash-out |
| A3 | 2-stage (epoch 1: full reward, epoch 2: bias-only at smaller LR) | ~2 h | Low — exploits the synergy story (§8.2) but at the cost of the proven single-recipe stability |
| A4 | Seed×3 noise floor on the headline | ~3 h | **DONE — KLFIX is the noise-floor measurement** (4 seeds, std 0.35pp) |

A4 is **already-superseded by KLFIX** (4 seeds, ±0.35pp std, beats the Critical Review's
"need std ≤ 0.6pp" rule by 1.7×). It's done.

A1, A2, A3 are all variants of "give the same recipe more data / more epochs." None of
them changes the *signal*. Wash-out tells us the L13 PPO is already extracting roughly
all the L13-direction information available — additional epochs/data will move accuracy
within the asymptote-noise range (≤ +0.5pp upside on canonical, by extrapolation of the
§7.1 trajectory's decelerating curve from +1.03pp → +0.79pp → +0.17pp across quartiles).

The §8.2 finding ("biasOnly + corrOnly are both worse than vanilla; the gain is from
the synergy") is mechanistically why scaling up data alone won't escape this asymptote:
the corrOnly+biasOnly trajectory is already a saturated synergy at 2k. More data of the
same kind reinforces the same plateau.

---

## 2. Ranking — Action 1 vs Action 3 vs Direction A

Scoring on three dimensions: **expected uplift (Δacc)**, **cost (GPU-h)**, and
**information value (IV)** — how much it shifts our priors about *what the recipe is
doing*.

| Action | Expected Δacc upside | Cost | Info value | Priority |
|--------|---------------------|------|------------|----------|
| **Action 1 — zmean-top5 ensemble reward** | **+1-2pp** (offline Cohen-d goes 3.70→7.21, full transfer would be ~2pp) | ~5 h (4-5h code + 1h GPU) | **Very high** — first time we move the reward away from a single layer; changes the per-axis story | **🟢 #1** |
| **Action 2 — wash-out on KLFIX-s1** | n/a (diagnostic) | ~30 min GPU | **High** — first confirmation whether KLFIX changes the mechanistic story (Pattern shift) | **🟢 #2** |
| **Action 3 — CF n=6166 on KLFIX** | n/a (causal evidence) | ~3 h | **High** — only path to a causal-debias claim; required for any external write-up per Critical Review §8.5 Direction C | **🟢 #3** |
| **Direction A1 — 2-epoch** | +0 to +0.5pp | ~1 h | Low — trajectory is already decelerating | 🟡 Defer until Action 1 verdict |
| **Direction A2 — 8k corpus** | +0 to +0.5pp | ~3 h | Low — same plateau, more data | 🟡 Defer until Action 1 verdict |
| **Direction A3 — 2-stage** | -0.5 to +1pp | ~2 h | Medium — destabilises the proven recipe (high downside risk) | 🟠 Skip — high risk, low EV |

### Why scaling is now Tier 2, not Tier 0

The Critical Review wrote Direction A while §3.3 #2 (no checkpoint analysis) and §3.3 #3
(no layer sweep) were still open. §7.1 closed #2 (curve monotone, single epoch enough)
and §7.2 closed #3 (L13 sits in the L9-L19 optimal plateau). After those closures, the
Critical Review's own Decision Rule §7.4 should have caused a re-ranking that didn't
happen explicitly: with monotone curves and an optimal-band probe, *the bottleneck is no
longer compute or data — it is the choice of reward representation*. And the offline
ensemble test (built in this branch, post-Critical-Review) is direct evidence that the
**reward representation has more room than scaling does**.

Quantitatively: KLFIX's 4 seeds on the 2k corpus give canonical acc 0.6298 ± 0.0035.
Scaling extrapolation (linear-in-log-data, optimistic) suggests 4k → ~0.633, 8k → ~0.638.
That's +0.3 to +0.8pp gross, swamped by per-seed CI. The ensemble offline signal
predicts +1-2pp at constant compute. So the cost-adjusted EV is roughly 5-10× higher
for Action 1 than for any of A1/A2/A3.

---

## 3. When does scaling become Tier 0 again?

Three concrete tripwires that promote scaling back to top priority:

1. **Action 1 (ensemble) under-performs** (Δacc(ensemble) ≤ Δacc(KLFIX) on canonical).
   In that case the ensemble was the wrong hypothesis and the next-cheapest lever is data.
   At that point launch A1 (2-epoch) immediately as the cheapest probe of the data axis.

2. **Action 2 (KLFIX wash-out) shifts to Pattern 1 or 3** (deep representational change).
   That would mean KLFIX is encoding corrections somewhere, which makes adding data a
   meaningful next experiment because we're no longer fighting a saturated synergy.

3. **Action 3 (CF n=6166) shows flip-rate Δ < 5pp.** In that case the headline +1.08pp
   accuracy is mostly accuracy gain, not causal debiasing, and the response is to scale
   the bias-aligned reward weight (a hyperparameter, but in the same family as scaling).

---

## 4. Concrete ordering (revision of NEXT_STEPS.md §5)

Replace the original 3-day plan with:

```
Day 1 (now, in parallel):
  - Action 2: launch klfix-s1 VLBias gens + per-layer scoring [bash script ready].
    Wall: ~45 min, Modal time: ~45 min, my time after launch: 0.
  - Action 1: implement multi-layer ensemble reward (drm_loader + ppo_trainer + args).
    Wall: ~3-4 h coding.

Day 2:
  - Action 2 result: re-run washout_classify on the merged scores; write 1-page
    pattern-classification update. Wall: ~30 min.
  - Action 1: 2-seed smoke. ~1 h GPU. Decide gate.

Day 3:
  - Action 1: 4-seed full sweep + canonical eval. ~1 h GPU parallel.
  - Action 3: queue cf_pairs.parquet n=6166 build script. Mechanical; runs offline locally.

Day 4 (gated):
  - Action 1: write report. If passes gates → declare new baseline.
  - Action 3: launch 5 × CF gen jobs on volume (klfix-mean, vanilla, full P08, corrOnly,
    biasOnly). ~3 h GPU.

Day 5+ (gated on Action 1 outcome):
  - If Action 1 ≥ +1.5pp: skip Direction A entirely, go straight to Direction B
    (vision-stream probe at L_v) and OOD eval (CrowS-Pairs, BBQ-cf).
  - If Action 1 ∈ [+1.0, +1.5pp]: run A1 (2-epoch) as a cheap upside probe.
  - If Action 1 < +1.0pp: pivot to A2 (8k corpus) + re-think reward design.
```

---

## 5. Bottom line

Scaling (Direction A) is **rank 4 of 4 priorities**, behind Actions 1/2/3. The Critical
Review's recommendation was right for its date but stale by today: it was written before
KLFIX, before the offline ensemble test, and before the wash-out result. With those three
pieces of evidence in hand, the bottleneck is reward signal width (Action 1) and
mechanistic confirmation (Actions 2/3), not training-data volume or epoch count.

Promote scaling back to Tier 0 only if Action 1 fails the canonical gate, OR if Action 2
flips the wash-out pattern, OR if Action 3 shows the headline isn't causally debiasing.
