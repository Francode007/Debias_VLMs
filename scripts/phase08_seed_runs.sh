#!/usr/bin/env bash
# Phase 0.8 Strategic Plan §5 — Seed × 3 launch cheatsheet (L13 + L17).
#
# This is the focused launcher for ONLY the 6 PPO replication jobs.
# Use phase08_day1_launch.sh for the full Day 1 program (wash-out + CF eval).
#
# Source debias_env first; run from repo root. All 6 jobs are detached, so
# closing your terminal is safe. Each job runs ~3 h on an A100-80GB.
#
# Pre-requisites:
#   • L13 probe head dir already exists on volume:
#       /mnt/data/generated_heads_probe_L13_base_biasA/sb_bench-PROBE-component
#   • L17 probe head dir must exist before launching the L17 triple. Build
#     it once via the wash-out script (which builds all 11 layers):
#         bash scripts/phase08_washout_make_heads.sh 17
#     OR re-use phase08_a3_probe_as_head.sh's run_probe_to_head step
#     pointed at base_L17_probe_weights_biasA.npz.
#
# This file is documentation-by-example. To run only the L13 triple:
#       bash -c 'set -e; SEEDS_L13="1 2 3" bash scripts/phase08_seed_runs.sh'
# But by default, sourcing this file will print but not execute.
exit 1

set -euo pipefail

source debias_env/bin/activate
COMMON_FLAGS=(
    --epochs 1
    --dataset sb_bench --model-family qwen
    --kl-beta 0.1 --target-kl 0.02
    --learning-rate 5e-6 --lora-r 16 --lora-alpha 32
    --batch-size 8 --max-gen-tokens 8
    --max-train-samples 2000
    --reward-mode bias_aligned
    --bias-aligned-coef 1.0 --correctness-coef 1.0 --ambig-preservation-coef 0.5
    --use-frozen-phi
    --midtrain-eval-every-steps 50 --midtrain-eval-samples 64
)

# ─── L13 reward (primary, 3 seeds) ──────────────────────────────────────────
for SEED in 1 2 3; do
  echo "▶ Launching L13 seed=${SEED}"
  modal run --detach src/run_modal.py::run_training \
      "${COMMON_FLAGS[@]}" \
      --output-dir "/mnt/data/output_ppo_phase08_2k_L13_s${SEED}" \
      --reward-head-layer 13 \
      --reward-heads-dir-override "/mnt/data/generated_heads_probe_L13_base_biasA/sb_bench-PROBE-component" \
      --seed "${SEED}"
done

# ─── L17 reward (deeper, 3 seeds) ───────────────────────────────────────────
for SEED in 1 2 3; do
  echo "▶ Launching L17 seed=${SEED}"
  modal run --detach src/run_modal.py::run_training \
      "${COMMON_FLAGS[@]}" \
      --output-dir "/mnt/data/output_ppo_phase08_2k_L17_s${SEED}" \
      --reward-head-layer 17 \
      --reward-heads-dir-override "/mnt/data/generated_heads_probe_L17_base_biasA/sb_bench-PROBE-component" \
      --seed "${SEED}"
done

echo "✅ All 6 detached jobs submitted. Track via:"
echo "    modal app list"
echo "    modal app logs <app-id>"
