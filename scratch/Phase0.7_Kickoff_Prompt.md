# Phase 0.7 Kickoff Prompt

> Paste the block below into a new GitHub Copilot Chat conversation to start Phase 0.7 with full context.

---

```
We are continuing work on the Debias_VLMs project. Phase 0.6 is complete.

CONTEXT TO LOAD FIRST (please read all three, in order, before doing anything):
1. scratch/Phase0.7_Context_Handoff.md   — full Phase 0.7 handoff: what 1.0 means, priority experiments, acceptance gates.
2. Phase0.6.md                           — D1/D2/D3 fixes, training/eval commands, deferred items.
3. scratch/Phase0.5_Context_Handoff.md   — original mode-collapse diagnosis (background only; do not re-derive).

SUMMARY OF WHERE WE STAND:
- Phase 0.6 closed all three D-blockers (D1 frozen-φ, D2 kept-heads filter, D3 token-position alignment).
- Phase 0.6 F (no-causal) training run hit 1.0000 SB-Bench test accuracy at step 70 and held it across 18 consecutive checkpoints. No collapse. v_loss / kl_beta / grad norms all stable.
- Letter-distribution analysis on ep1-end.jsonl produced a perfectly diagonal 2916×2916 confusion matrix with balanced letter predictions (A=32.9%, B=34.6%, C=32.5%) matching the gold distribution exactly. The "always-C / always-unknown" shortcut hypothesis is conclusively rejected.
- BUT: the SVM reward heads themselves score 1.0 on the (chosen, rejected) pair task on the SB-Bench test split. They are effectively a gold-label oracle on the same distribution as eval. So the 1.0 is the ceiling of in-distribution reward-model distillation — it does NOT yet demonstrate transferable debiasing.

PHASE 0.7 SCOPE — VERIFY THE 1.0 IS REAL, THEN TEST TRANSFER.

Priority-1 experiments (must complete before any new training run):

  P1 — question_polarity (or context_condition) split on the existing ep1-end.jsonl.
       Goal: confirm the 1.0 holds when gold is a named group, not just when gold is "unknown".
       Cost: ~1 min, purely local Python. Join /tmp/ep1-end.jsonl against sb_bench_data/data/test-*.parquet on `id`.
       Acceptance gate G1: both polarity splits ≥ 0.95 accuracy.

  P2 — letter-order perturbation eval (cyclic-1 shift) on ep1-step70 and ep1-end.
       Goal: confirm policy is reasoning about answer content, not memorising positional cues.
       Implementation: add --shuffle_answers flag to src/modules/inference/generate_sb_bench_answers.py; re-run run_phase0_eval_sweep with --only-tags "ep1-step70,ep1-end" and a new gen-subdir.
       Cost: ~30 min on Modal.
       Acceptance gate G2: perturbed acc ≥ 0.95.

  P3 — POPE regression eval on ep1-step70 and ep1-end, plus a one-time base-model POPE baseline.
       Goal: detect capability tax (loss of general VQA from over-fitting the SVM reward).
       Implementation: POPE data is already downloaded via run_setup; run_phase0_eval_sweep has a stub POPE generation branch but skips eval. Add a run_pope_eval entrypoint that loads a checkpoint, generates on /mnt/data/pope_data/pope_data.jsonl, and scores against POPE gt.
       Cost: ~20 min/ckpt on Modal.
       Acceptance gate G3: drop ≤ 5 pp vs base-model POPE accuracy.

Priority-2 (closer of Phase 0.7 — only after P1/P2/P3 are answered and reviewed):

  T1 — BBQ transfer eval (closest analog, but the SVM heads never saw it).
  T2 — One vision-bias benchmark (VLStereoSet / PAIRS / FACET, choose based on availability).
  T3 — (Optional) StereoSet / CrowS-Pairs text-only sanity check.
       Acceptance gate G4: T1 BBQ accuracy strictly above base-model BBQ accuracy → transferable debiasing.

OUT OF SCOPE FOR PHASE 0.7 (DEFER TO PHASE 0.8):
- Reintroducing the (rectified) causal penalty. Phase 0.8 plan emerges from the P3 capability-tax magnitude.
- kl_beta / optimizer state checkpoint-resume bug.
- Trainer --reward_position flag (off-by-one between trainer's curr_penultimate[:,:-1,:] and how D3 heads were trained — benign so far, leave alone unless P2 reveals positional sensitivity).
- PCA-DRM ablation.
- Multi-epoch training (1 epoch already saturates).

CRITICAL CAVEAT (must not be forgotten during the conversation):
SB-Bench acc = 1.0 is NOT the win condition for the project. It is the ceiling of in-distribution oracle distillation. The win condition is transfer evidence on a benchmark the SVM heads never saw. The point of Phase 0.7 is to discriminate between "real debias transfer" and "in-distribution oracle distillation".

Please start by:
  (a) reading the three context docs in order,
  (b) committing the uncommitted Phase 0.6 work as `feat(phase0.6): close D1/D2/D3 + drop causal penalty from loss` and branching `phase0.7_transfer_eval` off it,
  (c) running P1 immediately (purely local, no Modal needed) and reporting the per-polarity (or per-context_condition) accuracy breakdown,
  (d) waiting for my approval on the P1 result before designing the P2 generator perturbation,
  (e) implementing P2 → running on Modal → reporting,
  (f) implementing P3 → running on Modal → reporting base baseline + both checkpoints,
  (g) ONLY after G1+G2+G3 are all reported, propose the T1/T2 implementation plan.

Working tree is on branch `phase0_collapse_mitigation` with uncommitted Phase 0.6 changes (Phase0.6.md, .gitignore, evaluate_drm_heads.py, custom_vlm_ppo_trainer.py, args.py, drm_loader.py, run_modal.py). Commit before branching.
```

---

## Notes for the operator

- The prompt forces P1 first because it's *one Python script away* and immediately discriminates "real" vs "policy ignores context". Don't let the new agent skip ahead.
- P1 may surface that SB-Bench `question_polarity` is the wrong cut — if so, the right cut is `gold = the "unknown" option`, computable from `is_unknown(ansN)` against `label`. The handoff doc flags this; the new agent must verify before reporting G1.
- P2 needs a small generator change (`--shuffle_answers cyclic_1`). Keep the change minimal: a single function that rotates `(ans0, ans1, ans2)` by k and remaps `label` accordingly. Do **not** let the new agent rewrite the generator.
- P3 requires writing a thin `run_pope_eval` entrypoint. POPE data + generator already exist; only the eval-step wrapper is missing. ~50 LOC, not a refactor.
- T1 (BBQ) is the largest lift in Phase 0.7. Don't authorise it until P1/P2/P3 results are all on the table; if G3 fails badly (capability tax > 10pp), it may be more urgent to launch Phase 0.8 first with the rectified causal penalty.
- Recommended order during the Phase 0.7 conversation:
  1. Read docs → commit Phase 0.6 → branch
  2. P1 local script → report G1
  3. P2 generator patch → Modal sweep → report G2
  4. P3 POPE entrypoint → Modal eval (base + 2 ckpts) → report G3
  5. Joint write-up of G1+G2+G3 → operator decides whether to (a) proceed to T1, (b) jump to Phase 0.8, or (c) something else
  6. If T1 approved: implement BBQ pipeline → Modal eval → report G4 → close Phase 0.7
- Whatever the outcome, Phase 0.7 ends with a `Phase0.7.md` at the repo root mirroring the format of `Phase0.6.md`, documenting the four gate outcomes and the recommended Phase 0.8 plan.
