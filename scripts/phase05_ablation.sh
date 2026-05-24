#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0.5 Ablation Matrix
# ─────────────────────────────────────────────────────────────────────────────
# Run a small grid of binary-PPO experiments to isolate what fixes the collapse.
#
# Experiments:
#   A) High KL floor (kl_beta_min=0.5, target_kl=0.01) → tests whether the
#      adaptive controller loosening β was the proximate collapse cause.
#   B) Early-stopping (patience=20 steps, threshold=5%) → same hyperparams as
#      Phase 0 but auto-stops once accuracy drops — confirms step-50 as the
#      practical stopping point.
#   C) Reduced LR (5e-5 instead of 1e-4) → tests whether the update magnitude
#      is too large for the policy to stay near the initial optimum.
#   D) Larger batch (gradient_accumulation_steps=8 → effective BS=64) → tests
#      whether reward-signal variance is destabilizing the policy.
#
# Each run takes ≈30-60 min on A100-80GB (2000 samples, 1 epoch).
# Dense checkpoints every 10 steps + eval sweep afterward.
#
# Usage:
#   cd /Users/f0s03xp/Debias_VLMs
#   source debias_env/bin/activate
#   bash scripts/phase05_ablation.sh         # runs all 4 experiments
#   bash scripts/phase05_ablation.sh A       # runs only experiment A
#   bash scripts/phase05_ablation.sh A B     # runs A and B
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# Common flags
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

  # Training
  modal run src/run_modal.py::run_training \
    "${COMMON[@]}" \
    --output-dir "$outdir" \
    "$@"

  # Eval sweep
  modal run src/run_modal.py::run_phase0_eval_sweep \
    --train-output-dir "$outdir" \
    --combined-json "${outdir}/phase05_eval_sweep.json" \
    --data-path /mnt/data/sb_bench_data/sb_bench_data.parquet \
    --dataset sb_bench
}

# ── Experiment A: High KL floor ──────────────────────────────────────────────
run_A() {
  run_experiment "A_high_kl_floor" \
    --learning-rate 1e-4 \
    --value-learning-rate 5e-4 \
    --kl-beta 0.5 \
    --target-kl 0.01 \
    --kl-adapt-rate 0.05 \
    --kl-beta-min 0.5 \
    --kl-beta-max 5.0
}

# ── Experiment B: Early stopping ─────────────────────────────────────────────
run_B() {
  run_experiment "B_early_stop" \
    --learning-rate 1e-4 \
    --value-learning-rate 5e-4 \
    --kl-beta 0.1 \
    --target-kl 0.02 \
    --kl-adapt-rate 0.1 \
    --kl-beta-min 0.05 \
    --kl-beta-max 5.0 \
    --early-stop-patience 20 \
    --early-stop-threshold 0.05 \
    --early-stop-window 10
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

# ── Dispatch ─────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
  experiments="A B C D"
else
  experiments="$*"
fi

for exp in $experiments; do
  case "$exp" in
    A) run_A ;;
    B) run_B ;;
    C) run_C ;;
    D) run_D ;;
    *) echo "Unknown experiment: $exp (choose from A B C D)"; exit 1 ;;
  esac
done

echo ""
echo "================================================================"
echo "  ALL REQUESTED EXPERIMENTS COMPLETE"
echo ""
echo "  To pull results locally:"
echo "    for x in A_high_kl_floor B_early_stop C_reduced_lr D_large_batch; do"
echo "      modal volume get debias-vlm-persistent-storage \\"
echo "        /output_ppo_phase05_\${x}/phase05_eval_sweep.json ./phase05_\${x}.json"
echo "    done"
echo "================================================================"
