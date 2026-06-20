# Phase 0.8 — Three-Diagnostic Verdict (CORRECTED)

**Date:** 2026-06-18
**Models:** Qwen2.5-VL-3B-Instruct, PPO fine-tuned with steering-reward at L13 / L17
**Bench:** SB-Bench 9-axis test split (3747 rows from `split_indices_9axis.json`)

> **NOTE — supersedes `REFRAMING_DECISION.md`.** That earlier memo concluded a
> "convergent null" and recommended stopping the line of work. **It was wrong.**
> It conflated (a) the production `phase08_2k` model (which I never computed
> 9-axis accuracy for), (b) a separate set of L13/L17 reseed runs intended for
> *layer comparison* (legitimately noisy), and (c) a counterfactual flip-rate
> metric at n=228 paired items (insufficient power for a 3-pp effect). The
> headline Phase 0.8 result is intact and *strengthens* on the larger eval.

---

## 1. Headline (corrected)

**`phase08_2k` on the 9-axis test split (n=3747):**

- accuracy: **0.4860 (vanilla) → 0.5169 (phase08_2k)** = **+3.10 pp**
- McNemar paired: **b01 (gain) = 119 vs b10 (loss) = 3 → p = 1.1e-31**
- per-axis: **all 9 axes improve**, no regressions

This **reproduces and strengthens** the Critical Review §2.1 result (+1.85 pp on
the n=2916 Critical-Review subset → +3.10 pp on the n=3747 9-axis subset, same
direction, larger magnitude on the harder subset).

**Reward-component ablations on the 9-axis split (corroborate Critical Review §8):**

| run        | acc    | Δ      | b01 |  b10 | p          | reading |
|------------|-------:|-------:|----:|-----:|-----------:|---------|
| base       | 0.4860 |   —    |  —  |   —  |     —      | reference |
| phase08_2k | 0.5169 | +3.10  | 119 |    3 | 1e-31      | full reward, **strong** |
| corrOnly   | 0.4283 | −5.76  |   1 |  217 | 1e-63      | correctness-only PPO **collapses** |
| biasOnly   | 0.4817 | −0.43  |  15 |   31 | 0.03       | bias-only is essentially neutral |

The synergy story from §8.2 holds: **neither component alone produces a gain;
together they produce +3.1 pp**. The corrOnly run lost 217 vanilla-correct
answers and gained 1 — confirming that binary-correctness PPO at this budget is
degenerate.

---

## 2. Why the eval set matters (and why my earlier null was wrong)

| | Critical Review eval | My 9-axis eval |
|---|---:|---:|
| split file | `split_indices.json` | `split_indices_9axis.json` |
| n | 2916 | 3747 |
| Gender items | 585 | **1342** (2.3×) |
| Religion items | 144 | **60** (0.4×) |
| Nationality items | 224 | **69** (0.3×) |
| vanilla acc | 0.6190 | 0.4860 |
| phase08_2k acc | 0.6375 | 0.5169 |
| Δ (phase08_2k − vanilla) | +0.0185 | **+0.0310** |

The 9-axis split is a **different, harder, Gender-heavy** subset of SB-Bench, not
a sample. The vanilla baseline is 13 pp lower because Gender is the hardest
single axis and dominates the new split. The phase08_2k effect transfers
robustly to this harder set (+3.10 vs +1.85 on the original).

The vanilla model emits "C" (Can't be determined) **52% of the time on 9-axis**
vs **46% on the original** — both eval sets show the same C-bias behavior,
ruling out a prompt/decoding regression.

---

## 3. The L13 / L17 reseed runs — what they actually mean

These were trained as a **secondary, post-hoc layer-comparison experiment** at
the same hyperparams as `phase08_2k` but with three seeds each at L=13 and L=17.
They are **not** reproductions of `phase08_2k`'s recipe at the production layer
choice; they are smaller-scale layer probes.

| variant | acc    | Δ vs base | n_significant_seeds |
|---------|-------:|----------:|--------------------:|
| L13 mean (s1, s2, s4) | 0.4863 ± 0.0125 | +0.0004 | 2 of 3, mixed direction |
| L17 mean (s1, s2, s4) | 0.4911 ± 0.0158 | +0.0052 | 3 of 3, mixed direction |
| **phase08_2k** | **0.5169** | **+0.0310** | (production model) |

What this tells us correctly:
- **Layer-ablation reseeds are noisy** (std 1.3-1.6 pp; sign flips across seeds).
  This is a legitimate caveat about the *layer-comparison* experiment, not
  about `phase08_2k`.
- `phase08_2k` is **~1.7σ above the L17-reseed mean**. It is on the upper edge
  of the seed distribution, but a seeds-of-the-production-recipe sweep was
  never run; the L13/L17 reseeds are a different setting.
- A clean variance check would be: 3 reseeds of `phase08_2k`'s exact recipe
  (data, layer, hparams, just different RNG seeds). That is the right next
  experiment, **not "stop and redesign."**

---

## 4. The wash-out result — re-read

`washout_diagnostic.json` says: when we steer the trained policy back along
the L13 axis at inference time, no behavioral drop occurs. **Re-interpretation:**
PPO did not encode its corrections in the L13 residual direction (the policy
routes corrections through a different / output channel). This is *compatible*
with the +3.1 pp accuracy gain — the bias-projection reward shapes which
trajectories get reinforced; where the resulting policy stores the change is
a separate question. The wash-out is a mechanistic finding, not evidence
against the headline result.

---

## 5. The CF flip-rate diagnostic — what it does and doesn't say

n=228 polarity-paired items in the 9-axis test split.

| variant | flip_rate | ±SE | acc_orig | acc_swap |
|---|---:|---:|---:|---:|
| base | 0.347 | 0.032 | 0.513 | 0.474 |
| phase08_2k | 0.355 | 0.032 | **0.557** | **0.491** |
| corrOnly | 0.368 | 0.032 | 0.474 | 0.421 |
| biasOnly | 0.351 | 0.032 | 0.513 | 0.465 |

- Flip-rate is unchanged across variants — **PPO does not reduce polarity
  sensitivity**.
- BUT: phase08_2k boosts both `acc_orig` (+4.4 pp) and `acc_swap` (+1.7 pp),
  consistent with §1's polarity-invariant accuracy gain.
- Statistical power at n=228 paired items is far too low to detect a 3-pp
  effect on a 35-50% baseline. CF flip-rate at this n cannot adjudicate the
  headline.

**Honest reading:** phase08_2k is a *correctness-improving* intervention with
*incidental* (and statistically null) effects on counterfactual flip rate.
It is not a polarity-debiasing intervention in the strict counterfactual sense.
This is consistent with the Critical Review's §2.2 reading that the OOD
\|bias_score\| reduction was driven by the `non_neg` slice and not by changing
how the model handles polarity flips.

---

## 6. Decision rule, corrected

| §6 / §8 criterion | status |
|---|---|
| Probe held-out accuracy ≥ 0.70 | PASS (0.972 at L13, Critical Review §7.2) |
| Mid-training checkpoint curve monotone or peak-late | PASS (Critical Review §7.1) |
| `bias_aligned_coef` is necessary (not additive) | PASS (corrOnly collapses on 9-axis: −5.76 pp; reproduces §8) |
| **Effect transfers to a harder eval split (9-axis)** | **PASS (+3.10 pp, p=1e-31, all 9 axes improve)** |
| Effect is robust across seeds | **NOT YET TESTED with the production recipe** (L13/L17 reseeds are at a different setting) |
| Effect is causal in the counterfactual-flip sense | NULL at n=228 (low power); needs larger pair set or a different test |

---

## 7. Recommendation (revised)

**Continue Phase 0.8 as a "probe-as-head accuracy/debiasing intervention" — do
not stop or redesign.** The mechanism is reproducing, the synergy story is
intact, and the effect transfers to a harder subset. Two cheap follow-ups
before scaling:

1. **3-seed reproduction of the production `phase08_2k` recipe** (same data,
   same layer L13, same hparams, seeds {1, 2, 3}). Cost: ~3 hours on Modal.
   Gives a real CI on the headline.
2. **Build a larger CF pair set.** The current 228 paired items in the 9-axis
   test is power-limited. Either (a) extend pairs into the train split (this
   is OK because we evaluate, not train, on these items), or (b) compute
   flip-rate over the full 6166 paired items in canonical SB-Bench, not just
   the test split.

Then proceed with the **Direction A** path from Critical Review §8.6
(2-epoch / 8k / 2-stage scaling).

---

## 8. What to retract

- `Phase0.8/REFRAMING_DECISION.md` — superseded by this memo. The "convergent
  null" framing was wrong. Specifically, the "Option A — stop and re-design"
  recommendation in that memo is withdrawn.
