# Project State — Live Navigator

> **Read this FIRST in any new conversation.** The repo's `README.md` describes
> the very early C-DeFR-L scaffolding and is no longer the source of truth for
> what we're working on. This file is. It is updated at every phase close and
> whenever the topic-to-doc map changes.
>
> **Last updated**: 2026-06-24
> **Active branch**: `phase0.9_multilayer_ensemble_reward`
> **HEAD commit**: `cd68767` (Phase 0.9 R3 step 5: FINAL_REPORT + REPORT_EXTENSIVE integration)
> **Model under study**: Qwen2.5-VL-3B-Instruct (LM-decoder backbone, 36 transformer blocks, hidden_dim = 2048, bf16, flash-attn-2)
> **Reward backbone**: KLFIX-stabilised PPO with frozen `bias_aligned` linear probes

---

## Where we are right now

- ✅ **Phase 0.9 R1–R3 closed.** Verdict: multi-layer ensemble reward is **partially vindicated** — no in-distribution accuracy benefit on SB-Bench (+1.29 pp vs +1.37 pp like-for-like KLFIX-L13), but **better out-of-distribution debiasing** on VLBias transfer (R3 is the only method that reduces |bias_score| toward 0; +0.0021 vs KLFIX −0.0005). Trade-off: 1.4 pp ambig-collapse cost.
- 🔄 **Phase 0.9.5 opening.** Six pre-committed ablations spanning {ensemble window, pool function, probe sample size, LoRA placement} + one inference-time-erasure baseline + one probe-direction-convergence diagnostic. All run s1-first; expand to 4 seeds only on green s1.
- ⏸ **R4 (counterfactual flip-rate at n = 6166) deferred** until a robust ensemble config emerges from Phase 0.9.5.

---

## Phase status

| Phase | Status | Outcome doc | Active branch |
|---|---|---|---|
| 0.6 (D-blocker fixes) | closed | (legacy; superseded by 0.7+) | merged |
| 0.7 (VLBiasBench probe pipeline) | closed | `REPORT_EXTENSIVE.md` §5 | merged |
| 0.8 (KLFIX baseline + probe-as-reward) | closed | `REPORT_EXTENSIVE.md` §1–§8, §10.2; `Phase0.8/washout_klfix_verdict.md` | `phase0.8_strategic_plan_day1` |
| 0.9 R1–R3 (hardening + ensemble) | closed | `Phase0.9/FINAL_REPORT.md` | `phase0.9_multilayer_ensemble_reward` |
| **0.9.5 (ablations)** | **active** | `Phase0.9_Strategic_Plan.md` + `Phase0.9/STRATEGIC_PLAN_SUMMARY.md` | `phase0.9_multilayer_ensemble_reward` |
| R4 (CF causal evidence) | deferred | TBD | TBD |

---

## Topic → doc map (current truth source per topic)

| If you need… | Read this |
|---|---|
| Current phase strategy + sequencing rules + closure gates | [Phase0.9_Strategic_Plan.md](Phase0.9_Strategic_Plan.md) |
| Critical review behind the 0.9.5 ablation cycle | [Phase0.9_Critical_Review.md](Phase0.9_Critical_Review.md) |
| Live experiment log (updated as runs land) | [Phase0.9/STRATEGIC_PLAN_SUMMARY.md](Phase0.9/STRATEGIC_PLAN_SUMMARY.md) |
| Phase 0.9 R1–R3 final closure verdict | [Phase0.9/FINAL_REPORT.md](Phase0.9/FINAL_REPORT.md) |
| R1 bootstrap CI verdict (two-regime publication-grade) | [Phase0.9/r1_bootstrap_verdict.md](Phase0.9/r1_bootstrap_verdict.md) |
| R2 SB-Bench cross-dataset replication verdict | [Phase0.9/r2_sbbench_replication_verdict.md](Phase0.9/r2_sbbench_replication_verdict.md) |
| Workshop submission draft (paper-shaped) | [Workshop_Submission/REPORT_EXTENSIVE.md](Workshop_Submission/REPORT_EXTENSIVE.md) |
| 1-page workshop summary | [Workshop_Submission/REPORT_SUMMARY.md](Workshop_Submission/REPORT_SUMMARY.md) |
| Phase 0.8 KLFIX recipe (the +1.49 pp baseline) | `REPORT_EXTENSIVE.md` §6 + [scripts/phase08_klfix_s1_smoke.sh](scripts/phase08_klfix_s1_smoke.sh) |
| Multi-layer probe analysis (per-layer Δμ_corr, Cohen-d, ambig signature) | [Phase0.8/Multi_layer_head_analysis.md](Phase0.8/Multi_layer_head_analysis.md) |
| Two-regime geometric structure (R1+R2 result) | `REPORT_EXTENSIVE.md` §9 (revised 2026-06-23) |
| Ensemble reward measured result (R3) | `REPORT_EXTENSIVE.md` §10.1 + `Phase0.9/FINAL_REPORT.md` §3 |
| Wash-out diagnostic (KLFIX selector-not-eraser finding) | `REPORT_EXTENSIVE.md` §8 + `Phase0.8/washout_klfix_verdict.md` |
| Ensemble bundle + per-layer offline μ/σ | [Phase0.9/ensemble_bundle/](Phase0.9/ensemble_bundle/) (on disk) + `/generated_heads_probe_L17-33_zmean_base_biasA/` on Modal volume |
| Per-seed canonical SB-Bench gens (KLFIX-L13 baseline) | [Phase0.9/canonical/](Phase0.9/canonical/) |
| Per-seed VLBias transfer gens | [Phase0.9/vlbias/](Phase0.9/vlbias/) |
| Modal volume layout + reproduction commands | `Phase0.9/FINAL_REPORT.md` §7 |

---

## Files that are EXPLICITLY OUT OF DATE — do not use for state

| File | Why it's stale |
|---|---|
| [README.md](README.md) | Describes Phase 1/2/3 C-DeFR-L pipeline (Phase 0.5 era). Has a deprecation banner pointing here. |
| [README_v2_architecture.md](README_v2_architecture.md) | Phase 0.6 architecture proposal; superseded by Phase 0.8 KLFIX recipe. |
| [Phase0.6.md](Phase0.6.md) | Closed phase; verdicts rolled into REPORT_EXTENSIVE §2. |
| [Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md](Phase0.8/archive/Phase0.8_Strategic_Plan_v1.md) | Phase 0.8 v1 strategic plan, archived. v2 is at `Phase0.8_Strategic_Plan.md`. |
| Any `scratch/*.md` | Working-doc scratchpads; not authoritative. |

---

## Most recent commits (HEAD → backwards)

```
cd68767 Phase 0.9 R3 step 5: FINAL_REPORT + REPORT_EXTENSIVE integration
589ff46 Phase 0.9 audit: like-for-like 3-seed KLFIX SB-Bench aggregate
0bb1c83 Phase 0.9 R3 step 4b+4c: aggregate canonical + VLBias transfer results
b517f82 Phase 0.9 R3 step 4c: VLBias cross-dataset transfer gens launcher
1ef40bb Phase 0.9 R3 step 4b: canonical SB-Bench gens launcher + seed aggregator
afa7ef5 Phase 0.9 R3 step 4a: 4-seed ensemble PPO sweep launcher
44b84df Phase 0.9 R3.3: drop --value-warmup-* from ensemble smoke launcher
fcc61f8 Phase 0.9 R3 step 1+2: ensemble probe bundle + multi-layer reward path
600f28c Phase 0.9 R1+R2: bootstrap CIs + SB-Bench cross-dataset replication
```

Branch tips on origin:
- `phase0.9_multilayer_ensemble_reward` (HEAD) = current work
- `phase0.8_strategic_plan_day1` = Phase 0.8 close-out
- `phase0.8-tail` = Phase 0.8 in-flight scratch (160-file snapshot)

---

## Update protocol (for future you / agents)

When closing a major experiment or a phase:

1. Append a one-paragraph closure summary at the top of this file under "Where we are right now."
2. Move the closed sub-phase from "active" to "closed" in the **Phase status** table.
3. Update the **Topic → doc map** if any topic's truth source moved.
4. Update **Most recent commits** with `git log --oneline -10`.
5. Commit with message `Project state: <event>` (e.g., `Project state: Phase 0.9.5 closed`).
6. Push to origin so a fresh-context conversation can read it.

Any file added to the repo that contains *verdicts* or *strategy* should be added to the topic map at the same time.