# Phase 0.8 — Strategic Plan (complementary to `Phase0.8_Critical_Review.md`)

> **Purpose.** The critical review documents *what happened* (results, ablations,
> probe audits). This plan documents *what we do next, in what order, and under
> which decision rules* — so that future work is auditable against pre-committed
> gates rather than post-hoc rationalization.
>
> **Status snapshot (entering this plan).** Phase 0.8 PPO at L13 yields
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

## 1. Tiered priorities

### Tier 0 — Hardening the existing headline (must-do, this week)

| # | Item                          | Cost          | Decision rule                                                                                                                      |
|---|-------------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------|
| 1 | **Seed × 3 replication, L13** | ~3 h GPU      | Headline retained iff mean(SB-Bench Δ) > 0 with all 3 seeds individually p<0.05 vs vanilla on McNemar. Otherwise demote to "trend". |
| 2 | **Counterfactual flip-rate**  | ~½ day human  | Causal claim accepted iff full P08 reduces flip-rate by ≥30% relative to vanilla on paired demographic swaps, with corrOnly ≤ 0.   |
| 3 | **Wash-out diagnostic (L13→deeper)** | ~15 min, no training | Decides Tier 1#4 (multi-layer LoRA). See §3.                                                                              |

### Tier 1 — Cheap probes of the recipe's design space (gated)

| # | Item                          | Cost     | Gate / Decision rule                                                                                                               |
|---|-------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------|
| 4 | **L17 reward, seed × 3**      | ~3 h GPU | Compare (mean_acc_L17, mean_|bias|_L17) to L13 triple. Adopt L17 only if Δacc ≥ +0.5 pp AND Δ|bias| ≤ 0 with overlapping 95% CIs.   |
| 5 | **Multi-layer LoRA (L13/17/21/25 attached)** | ~6 h GPU | **Gated on wash-out Pattern 2 or 3** (Tier 0 #3). If Pattern 1, do NOT run; instead reduce LoRA rank.                        |
| 6 | **Multi-layer reward** (probe at L13 + L21 averaged) | ~3 h GPU | Gated on wash-out Pattern 3 only. Otherwise unjustified objective complication.                                              |

### Tier 2 — Generalization / mechanistic understanding (lower urgency)

| # | Item                                            | Cost          | Justification                                                       |
|---|-------------------------------------------------|---------------|---------------------------------------------------------------------|
| 7 | Vision-tower probe (`bias_aligned` on image patches) | ~½ day        | Tests whether bias circuit is text-only or cross-modal. Gated on Tier 0/1 done. |
| 8 | OOD eval suite (CrowS-Pairs MM, BBQ-cf)         | ~½ day eval   | Tests transfer to held-out distributions, not just VLBias.          |
| 9 | Reward-head distillation into smaller adapter   | ~1 day        | Engineering polish — only after Tier 0/1 verdicts.                  |

---

## 2. Decision tree

```
                              ┌──────────────────────────┐
                              │ Wash-out diagnostic (#3) │
                              └────────────┬─────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
      Pattern 1 (clean)           Pattern 2 (wash-out)          Pattern 3 (gaming)
      Δμ drops L13 AND ≥L17       Drops L13, flat ≥L17          Drops L13, GROWS ≥L25
              │                            │                            │
   ┌──────────┴──────────┐       ┌─────────┴─────────┐       ┌──────────┴─────────┐
   │ Skip multi-layer    │       │ Run multi-layer    │       │ Multi-layer LoRA + │
   │ LoRA (#5).          │       │ LoRA (#5) attached │       │ multi-layer reward │
   │ Reduce LoRA rank    │       │ to L13/17/21/25.   │       │ (#5 + #6).         │
   │ instead.            │       │ Single-layer reward│       │ Treat L13-only as  │
   └─────────────────────┘       │ kept at L13.       │       │ proxy-hacked.      │
                                 └────────────────────┘       └────────────────────┘
```

Independent of the diagnostic, Tier 0 (#1, #2) runs always.

---

## 3. Wash-out diagnostic spec

**Question.** When we PPO-train against the L13 probe, does the bias
representation also drop at deeper layers (L17–L35), stay unchanged
(wash-out), or *grow* (proxy-hacking)?

**Method.**
1. Take the same VLBias 900-stratified probe sample used in §4½.14.
2. Run inference with each of: `vanilla`, `phase08_full`, optionally
   `corrOnly`, `biasOnly`. Capture hidden states at every layer in
   {1, 5, 9, 11, 13, 17, 21, 25, 29, 33, 35}.
3. Score with the *base-fit* per-layer probe weights (frozen — same heads
   used in §4½.14). Compute Δμ_corr per layer for each model vs vanilla.
4. Classify per-layer pattern:
   - **Pattern 1 (clean):** Δμ_phase08 ≤ −0.3 σ at L13 *and* ≤ −0.15 σ at every layer ≥ L17.
   - **Pattern 2 (wash-out):** Δμ_phase08 ≤ −0.3 σ at L13 *but* |Δμ| < 0.10 σ at every layer ≥ L21.
   - **Pattern 3 (proxy-hack):** Δμ_phase08 ≤ −0.3 σ at L13 *but* Δμ > 0 at any layer ≥ L25.

**Output artifact.** `Phase0.8/washout_diagnostic.json` with per-layer Δμ for
each model + classification verdict.

**Implementation.** Use existing `run_vlbias_offline_score` with `--layer_idx N`
and per-layer probe heads packaged under `--heads_dir`. Must pre-publish the
per-layer probe heads to the volume (they currently exist as
`Phase0.8/a3_results/base_L{N}_probe_weights_biasA.npz` — these need to be
serialized into the `sb_bench-PROBE-component` `.pth` format expected by the
scorer).

---

## 4. Counterfactual flip-rate eval set spec

**Question.** Does the P08 model change its answer when we swap the
demographic referent in a question (e.g. *grandfather* ↔ *grandson*,
*husband* ↔ *wife*) while holding image and frame fixed? A *bias-free* model
should flip its answer at the rate dictated by the gold answer alone; a
biased model flips much more.

**Construction.**
1. Filter SB-Bench to the unambig (non-C) subset (gold ∈ {A, B}).
2. For each question, parse the two named entities `e1` and `e2` referenced by
   the answer choices A and B (SB-Bench encodes these as templated swap pairs
   already; cross-reference the unpacked SB-Bench schema).
3. Generate the *swapped* prompt by exchanging `e1` and `e2` in the question
   text (and the answer-choice ordering, so gold is preserved at the *referent*
   level rather than the *letter* level).
4. Pair `(orig_prompt, swap_prompt)` keyed by `qid`.

**Metrics (per model).**
- `flip_rate` = fraction of pairs where `model(orig) != letter_of_swap(model(swap))`
   under the referent re-mapping (i.e. did the answer change *meaning*?).
- `accuracy_orig` and `accuracy_swap`, individually.
- McNemar on (correct_orig vs correct_swap) — symmetry test.

**Decision rule (causal claim).**
Phase 0.8 model accepted as causally debiased iff:
- `flip_rate(P08) ≤ 0.7 × flip_rate(vanilla)` AND
- `accuracy_swap(P08) − accuracy_orig(P08)` is within ±2 pp (no asymmetric collapse).
- corrOnly does NOT meet the first criterion (controls for "any RL helps").

**Output artifact.** `Phase0.8/counterfactual_eval/cf_pairs.parquet` and
`cf_metrics.json`.

---

## 5. Seed × 3 spec (L13 and L17)

**Recipe (identical to the headline 2k run).**
- `lr=5e-6`, `lora_r=16`, `lora_alpha=32`, `batch_size=8`,
  `max_gen_tokens=8`, `max_train_samples=2000`,
  `target_kl=0.02`, `kl_β=0.1`, `w_corr=1.0`, `w_bias=1.0`, `w_ambig=0.5`.
- Reward layer: 13 for #1, 17 for #4. Seeds: {1, 2, 3} each.
- Each run: SB-Bench eval + VLBias transfer eval at end.

**Pass criteria (#1 L13 triple).**
- 3/3 individual SB-Bench McNemar p<0.05.
- mean Δacc ≥ +1.0 pp.
- mean |bias_score| reduction ≥ 50% of headline (i.e. ≥ 35.5%).

**Adoption criterion (#4 L17 vs #1 L13).**
- L17 adopted as new default iff: mean Δacc(L17) ≥ mean Δacc(L13) + 0.5 pp AND
  mean |bias_score|(L17) ≤ mean |bias_score|(L13). 95% CIs must not overlap on
  at least one of the two metrics.

**Output artifact.** `Phase0.8/seed_replication/{layer}_{seed}/...` plus a
summary table appended as §9 to the critical review.

---

## 6. Execution timeline

| Day | Action                                                                                                  | Owner | Cost           |
|-----|---------------------------------------------------------------------------------------------------------|-------|----------------|
| 1   | (a) Land this plan; (b) wire `--seed` flag in args/train_rl/run_modal; (c) draft counterfactual eval generator; (d) prep wash-out runner. | agent | local edits     |
| 1   | Launch wash-out diagnostic (offline scoring, ~15 min GPU).                                               | user  | 1 modal call    |
| 1–2 | Launch seed × 3 L13 + seed × 3 L17 (6 detached modal apps in parallel).                                  | user  | 6 × ~3 h GPU    |
| 2   | Generate counterfactual pair set; sanity-check 50 random pairs by hand.                                  | agent + user | ½ day      |
| 2–3 | Counterfactual eval runs (vanilla, full, corrOnly, biasOnly).                                            | user  | 4 × ~30 min GPU |
| 3   | Wash-out classification → decide on multi-layer LoRA (#5).                                               | agent | analysis        |
| 3   | Seed-triple results in → decide if headline is replicated.                                               | agent | analysis        |
| 4   | (Conditional) Multi-layer LoRA run if Pattern 2/3.                                                       | user  | ~6 h GPU        |
| 5   | Append §9 (replication) and §10 (counterfactual) to critical review.                                     | agent | local edits     |

---

## 7. What we are explicitly **not** doing next

These are *deferred* with reasons, so we do not lose them:

- **PPO scale-up beyond 2k** — until the noise floor is established by Tier 0
  #1, more samples is more compute on a possibly noisy estimate.
- **DPO / IPO replacement of PPO** — orthogonal to the question of "does the
  reward signal point at the right thing"; the answer to that is what the
  wash-out + counterfactual tests deliver.
- **Reward-model retraining at deeper layers** (e.g. L25) — premature without
  the wash-out verdict; would conflate two changes.
- **VLBiasBench retraining** — VLBias stays as the *transfer* set. Training on
  it would dissolve the strongest piece of evidence we have.
- **Switching to the 7B model** — capacity is not the current bottleneck;
  identification (causal vs correlational) is.

---

## 8. Risk register

| Risk                                                                  | Mitigation                                                  |
|-----------------------------------------------------------------------|-------------------------------------------------------------|
| Seed-triple mean Δ < 1.0 pp                                           | Demote headline to "directional"; pivot to counterfactual.  |
| Counterfactual flip-rate(P08) ≈ flip-rate(corrOnly)                   | Headline becomes an accuracy claim, not a debias claim.     |
| Wash-out shows Pattern 3 (gaming)                                     | Block all single-layer-L13 conclusions; require multi-layer.|
| McNemar `qid` schema bug recurs                                       | Patch `scripts/phase0_mcnemar.py` to row-index pair always. |
| Multi-layer LoRA blows up memory                                      | Fall back to LoRA on subset {L13, L21}; rank 8.             |

---

## 9. Pointers (single source of truth)

- Headline model: `/mnt/data/output_ppo_phase08_2k/final_debiased_model`
- corrOnly model: `/mnt/data/output_ppo_phase08_2k_corrOnly/final_debiased_model`
- biasOnly model: `/mnt/data/output_ppo_phase08_2k_biasOnly/final_debiased_model`
- L13 probe head: `/mnt/data/generated_heads_probe_L13_base_biasA/sb_bench-PROBE-component/sb_bench-PROBE-component0.pth`
- Per-layer probe weights (npz, need conversion): `Phase0.8/a3_results/base_L{N}_probe_weights_biasA.npz`
- Vanilla SB-Bench gens: `/mnt/data/sb_bench_vanilla_baseline/sb_bench_generations.jsonl`
- Vanilla VLBias gens: `/mnt/data/phase07_vlbiasbench/base_vlbias_gen.jsonl`
- Critical review: `Phase0.8_Critical_Review.md`
- Multi-layer probe analysis: `Phase0.8/Multi_layer_head_analysis.md` §4½.14

