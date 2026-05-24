#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0.5 Ablation Matrix  (revised after Phase-0 trajectory diagnosis)
# ─────────────────────────────────────────────────────────────────────────────
# What Phase 0 telemetry showed (see metrics.jsonl analysis):
#   • Eval peaked at step 50 (0.9949), then monotonically collapsed to 0.54
#     by ep1-end.
#   • Between step 30→40 the per-token KL spiked 11x (0.07 → 0.76).
#   • The adaptive KL controller could only triple β per step, so by the time
#     it caught up at step 60 it was pinned at the ceiling (β=5.0). KL then
#     decayed to ~0.03 but β stayed near 5.0 → over-regularised policy could
#     not recover, mode-collapsed onto letter B (63% of predictions).
#   • Training-batch binary_accuracy at bs=8 with temp=0.3 is far too noisy
#     to use as early-stop signal (range 0.125–0.625 even at peak eval=0.99).
#
# Revised experiments:
#   A) AGGRESSIVE high KL floor + fast adapt + raised ceiling — keeps β high
#      through the warmup-end lurch so the spike never crosses target_kl.
#   B) Early stopping driven by HELD-OUT EVAL ACC (not training-batch acc).
#      Uses the new --use_eval_for_early_stop signal added to the trainer.
#   C) Reduced LR — orthogonal: tests whether smaller updates avoid the spike.
#   D) Larger effective batch (grad_accum=8 → effective BS=64) — reduces
#      reward-signal variance entering the controller.
#   E) Tight gradient clipping (max_grad_norm=0.1, was 1.0) — directly caps
#      the single-step weight delta that triggered the step-40 KL eruption.
#
# Each run also enables mid-training eval every 10 steps so we have a clean
# trajectory signal (greedy decoding, n=64 held-out) in metrics.jsonl alongside
# the noisy training-batch acc.
#
# Each run ≈30-60 min on A100-80GB. Dense checkpoints every 10 steps + eval
# sweep afterward.
#
# Usage:
#   cd /Users/f0s03xp/Debias_VLMs
#   source debias_env/bin/activate
#   bash scripts/phase05_ablation.sh         # runs all 5 experiments
#   bash scripts/phase05_ablation.sh A E     # runs A and E
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# Common flags shared across all experiments.
# Mid-training eval gives a clean signal every 10 steps (n=64 greedy);
# dense per-10-step ckpts mean we always have a recoverable best ckpt.
COMMON=(
  --epochs 1
  --max-train-samples 2000
  --reward-mode binary
  --batch-size 8
  --max-gen-tokens 8
  --value-clip-range 0.2
  --lr-schedule cosine
  --min-lr-ratio 0.2
  --warmup-ratio 0.05
  --ckpt-every-steps 10
  --midtrain-eval-every-steps 10
  --midtrain-eval-samples 64
)

run_experiment() {
  local name="$1"; shift
  local outdir="/mnt/data/output_ppo_phase05_${name}"
  echo ""
  echo "================================================================"
  echo "  EXPERIMENT: $name"
  echo "  Output:     $outdir"
  echo "================================================================"
  echo ""

  modal run src/run_modal.py::run_training \
    "${COMMON[@]}" \
    --output-dir "$outdir" \
    "$@"

  modal run src/run_modal.py::run_phase0_eval_sweep \
    --train-output-dir "$outdir" \
    --combined-json "${outdir}/phase05_eval_sweep.json" \
    --data-path /mnt/data/sb_bench_data/sb_bench_data.parquet \
    --dataset sb_bench
}

# ── Experiment A: AGGRESSIVE high-KL-floor controller ────────────────────────
# Targets the proximate cause: the controller was too slow + ceiling too low.
# kl_beta starts 10x higher (1.0 vs 0.1), floor 20x higher (1.0 vs 0.05),
# ceiling 10x higher (50.0 vs 5.0), adapt_rate 5x faster (0.5 vs 0.1),
# target_kl 4x tighter (0.005 vs 0.02).
run_A() {
  run_experiment "A_high_kl_floor" \
    --learning-rate 1e-4 \
    --value-learning-rate 5e-4 \
    --kl-beta 1.0 \
    --target-kl 0.005 \
    --kl-adapt-rate 0.5 \
    --kl-beta-min 1.0 \
    --kl-beta-max 50.0
}

# ── Experiment B: Early stop on HELD-OUT EVAL accuracy ───────────────────────
# Drops the unreliable training-batch acc signal; uses --use_eval_for_early_stop
# so the rolling window is computed over greedy eval points instead.
run_B() {
  run_experiment "B_early_stop_eval" \
    --learning-rate 1e-4 \
    --value-learning-rate 5e-4 \
    --kl-beta 0.1 \
    --target-kl 0.02 \
    --kl-adapt-rate 0.1 \
    --kl-beta-min 0.05 \
    --kl-beta-max 5.0 \
    --early-stop-patience 3 \
    --early-stop-threshold 0.05 \
    --early-stop-window 3 \
    --use-eval-for-early-stop
}

# ── Experiment C: Reduced LR ─────────────────────────────────────────────────
run_C() {
  run_experiment "C_reduced_lr" \
    --learning-rate 5e-5 \
    --value-learning-rate 2.5e-4 \
    --kl-beta 0.1 \
    --target-kl 0.02 \
    --kl-adapt-rate 0.1 \
    --kl-beta-min 0.05 \
    --kl-beta-max 5.0
}

# ── Experiment D: Larger effective batch ─────────────────────────────────────
run_D() {
  run_experiment "D_large_batch" \
    --learning-rate 1e-4 \
    --value-learning-rate 5e-4 \
    --kl-beta 0.1 \
    --target-kl 0.02 \
    --kl-adapt-rate 0.1 \
    --kl-beta-min 0.05 \
    --kl-beta-max 5.0 \
    --gradient-accumulation-steps 8
}

# ── Experiment E: Tight gradient clipping (NEW) ──────────────────────────────
# Caps the single-step weight delta. Keeps default KL controller.
run_E() {
  run_experiment "E_tight_grad_clip" \
    --learning-rate 1e-4 \
    --value-learning-rate 5e-4 \
    --kl-beta 0.1 \
    --target-kl 0.02 \
    --kl-adapt-rate 0.1 \
    --kl-beta-min 0.05 \
    --kl-beta-max 5.0 \
    --max-grad-norm 0.1
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
  experiments="A B C D E"
else
  experiments="$*"
fi

for exp in $experiments; do
  case "$exp" in
    A) run_A ;;
    B) run_B ;;
    C) run_C ;;
    D) run_D ;;
    E) run_E ;;
    *) echo "Unknown experiment: $exp (choose from A B C D E)"; exit 1 ;;
  esac
done

echo ""
echo "================================================================"
echo "  ALL REQUESTED EXPERIMENTS COMPLETE"
echo ""
echo "  To pull results locally:"
echo "    for x in A_high_kl_floor B_early_stop_eval C_reduced_lr \\"
echo "             D_large_batch E_tight_grad_clip; do"
echo "      modal volume get debias-vlm-persistent-storage \\"
echo "        /output_ppo_phase05_\${x}/phase05_eval_sweep.json ./phase05_\${x}.json"
echo "      modal volume get debias-vlm-persistent-storage \\"
echo "        /output_ppo_phase05_\${x}/metrics.jsonl ./phase05_\${x}_metrics.jsonl"
echo "    done"
echo "================================================================"
