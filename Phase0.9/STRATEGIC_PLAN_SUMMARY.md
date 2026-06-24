# Phase 0.9.5 Live Experiment Log

> **Update this file after EVERY experiment lands.** One row per experiment. Each
> row has explicit s1 / 4-seed columns + a per-experiment verdict + cumulative spend.
> This is the operational summary; the prose justification is in
> [../Phase0.9_Critical_Review.md](../Phase0.9_Critical_Review.md) and the full
> sequencing rules are in [../Phase0.9_Strategic_Plan.md](../Phase0.9_Strategic_Plan.md).
>
> **Format invariants**:
> - Update the **Status** column to one of: `not-started`, `s1-running`, `s1-done`,
>   `4-seed-running`, `4-seed-done`, `dropped`, `winning`.
> - Update the **s1 verdict** column to `green` / `red` per §2.1 rule
>   (SB-Bench Δ ≥ +1.0 pp AND VLBias |bs| reduction ≥ +0.001).
> - Update the **Cumulative spend** total at the bottom every time a row's status changes.
> - When a row reaches `winning` status, run R4 with that config.
>
> **Last updated**: 2026-06-24 · **Author**: open Phase 0.9.5

---

## Reference baselines (frozen)

| Variant | n_seeds | SB-Bench Δ vs vanilla | VLBias Δ overall_acc | VLBias \|bs\| reduction | VLBias ambig_acc | Source |
|---|---|---|---|---|---|---|
| Vanilla Qwen2.5-VL-3B-Instruct | n/a | 0.0 (61.49 %) | 0.0 (55.13 %) | 0.0 (\|bs\|=0.0070) | 91.90 % | [Phase0.9/canonical/canonical_base_sbbench_gen.jsonl](canonical/canonical_base_sbbench_gen.jsonl) + [Phase0.9/vlbias/base_vlbias_results.json](vlbias/base_vlbias_results.json) |
| KLFIX-L13 (4 seeds, full) | 4 | **+1.49 pp** ± 0.35 | −0.48 pp | −0.0005 (worsens) | 90.90 % | [Phase0.9/canonical/klfix_L13_4seeds_aggregate.json](canonical/klfix_L13_4seeds_aggregate.json) |
| KLFIX-L13 (3 seeds s1/s2/s4, like-for-like) | 3 | **+1.37 pp** ± 0.31 | −0.71 pp | −0.0005 (worsens) | 90.90 % | [Phase0.9/canonical/klfix_L13_3seeds_lf_aggregate.json](canonical/klfix_L13_3seeds_lf_aggregate.json) |
| **R3 ensemble (3 seeds, 5-layer L17–33 zmean)** | 3 | **+1.29 pp** ± 0.38 | −0.66 pp | **+0.0021** (reduces) | 89.51 % (−1.4 pp vs KLFIX) | [Phase0.9/canonical/ensemble_klfix_3seeds_aggregate.json](canonical/ensemble_klfix_3seeds_aggregate.json) + [Phase0.9/vlbias/phase09_ensemble_s{1,2,4}_klfix_vlbias_results.json](vlbias/) |

---

## Experiment table

| ID | Title | Window | Pool | LoRA placement | Status | s1 SB-Bench Δ | s1 VLBias \|bs\| Δ | s1 verdict | 4-seed ID Δ | 4-seed OOD ambig_acc | 4-seed verdict | Spend ($) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E1a** | Inference-time L13 erasure | n/a | n/a | n/a (no training) | not-started | — | — | — | — | — | — | 0 | one-shot baseline; if matches PPO → invalidates PPO investment |
| **E1b** | Inference-time multi-layer erasure | {L13,L17,L21,L25} | n/a | n/a | not-started | — | — | — | — | — | — | 0 | optional follow-up to E1a |
| **E2** | Probe-direction convergence test | L13 (and optionally L17–33) | n/a | n/a | not-started | n/a | n/a | n/a | n/a | n/a | cos(w_900,w_max) ≥ 0.95 = pass | 0 | gates E7 trigger |
| **E3** | Add L13 + zmedian | {L13,L17,L21,L25,L29,L33} | zmedian | distributed (default) | not-started | — | — | — | — | — | — | 0 | direct L13-compensation test |
| **E4** | Targeted LoRA L17–25 + winning E3 ensemble | E3-winner | E3-winner | L17–25 only | not-started | — | — | — | — | — | — | 0 | conditional on E3 4-seed green |
| **E5** | Disentangled-regime-only | {L13,L17,L21} | zmedian | distributed | not-started | — | — | — | — | — | — | 0 | falsification of "deep regime is needed" |
| **E6** | Agreement-pool (K=4/6) | {L13,L17,L21,L25,L29,L33} | agreement_mean K=4 | distributed | not-started | — | — | — | — | — | — | 0 | first-significant-layer principle |
| **E7** | Probe re-fit n > 900 | n/a | n/a | n/a | not-started (cond on E2) | n/a | n/a | n/a | n/a | n/a | new bundle | 0 | rebuild ensemble bundle from larger probe set |
| **R4** | CF flip-rate n=6166 with winning config | winning-window | winning-pool | winning-LoRA | deferred | n/a | n/a | n/a | n/a | n/a | flip_rate(target) ≤ 0.8× vanilla = pass | 0 | post-0.9.5 |

**Cumulative spend**: **$0** / $50 budget cap.

---

## Per-experiment notes (append as runs complete)

### E1a — Inference-time L13 erasure baseline (not yet started)

- Spec: [Phase0.9_Strategic_Plan.md](../Phase0.9_Strategic_Plan.md) §3.1
- Outputs to land here: `Phase0.9/erasure/E1a_sbbench_metrics.json` + `E1a_vlbias_metrics.json`
- Decision rule: see §2.2 decision tree.

### E2 — Probe-direction convergence diagnostic (not yet started)

- Spec: [Phase0.9_Strategic_Plan.md](../Phase0.9_Strategic_Plan.md) §3.2
- Output: `Phase0.9/probe_convergence/diagnostic.json` with per-layer convergence table.
- Decision rule:
  - `cos(w_900, w_max) ≥ 0.95` → existing bundle is fine; do NOT trigger E7
  - `cos(w_900, w_max) < 0.90` → trigger E7; pause all PPO ablations until new bundle is built
  - `0.90 ≤ cos < 0.95` → soft trigger; document and decide.

---

## Closure scorecard (against §4 Phase 0.9.5 gates)

| Gate | Threshold | Best result to date | Pass? |
|---|---|---|---|
| ID SB-Bench Δ vs vanilla | ≥ +1.3 pp (n=4) | R3: +1.29 pp (n=3); KLFIX-L13 4-seed: +1.49 pp | ⚠️ R3 short by 0.01 pp; KLFIX-L13 passes but is not the ensemble |
| OOD VLBias \|bs\| reduction | ≥ +0.0015 (n=3–4) | R3 ensemble: +0.0021 | ✅ |
| OOD VLBias ambig_acc | ≥ −1.0 pp vs vanilla | R3 ensemble: −2.39 pp | ❌ R3 too aggressive; this is what 0.9.5 is fixing |
| R4 CF flip_rate | ≤ 0.8× vanilla | not measured | ⏸ post-0.9.5 |

**Composite verdict (today)**: 1 of 4 gates passes. Phase 0.9.5 ablations target the failing two gates (ID Δ and ambig erosion).

---

## Living spend ledger

| Date | Experiment | Step | Spend ($) | Cumulative |
|---|---|---|---|---|
| 2026-06-24 | — | (Phase 0.9.5 opens; baselines frozen) | 0 | 0 |

---

## Update protocol

1. After every Modal run completes + metrics pull:
   - Update the experiment's row: `Status`, the numeric s1/4-seed cells, `Spend ($)`.
   - Update the **Cumulative spend** total at the bottom of the experiment table.
   - Append a per-experiment note in the next section.
2. After every decision point (s1 verdict, 4-seed verdict):
   - Update the **Composite verdict** in the closure scorecard if the result is new best.
3. After Phase 0.9.5 closes:
   - Mark the winning config's row as `winning`.
   - Promote the row to a separate "Winner" section at the top of this file.
   - Trigger R4 spec writing.
   - Update [PROJECT_STATE.md](../PROJECT_STATE.md) phase status table.