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
#     it once via:
#         bash scripts/phase08_upload_a3_npz.sh 17     # upload npz to volume
#         bash scripts/phase08_washout_make_heads.sh 17 # npz → .pth head dir
#
# Usage:
#   bash scripts/phase08_seed_runs.sh           # both L13 and L17, 3 seeds each
#   LAYERS="13" bash scripts/phase08_seed_runs.sh   # only L13 triple
#   LAYERS="17" bash scripts/phase08_seed_runs.sh   # only L17 triple
#   SEEDS="1 2"  bash scripts/phase08_seed_runs.sh  # only seeds 1 and 2

set -euo pipefail

# Auto-source venv if not already active.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source debias_env/bin/activate
fi

LAYERS="${LAYERS:-13 17}"
SEEDS="${SEEDS:-1 2 3}"

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

# ─── L13 reward (primary) ───────────────────────────────────────────────────
if [[ " ${LAYERS} " == *" 13 "* ]]; then
  for SEED in ${SEEDS}; do
    echo "▶ Launching L13 seed=${SEED}"
    modal run --detach src/run_modal.py::run_training \
        "${COMMON_FLAGS[@]}" \
        --output-dir "/mnt/data/output_ppo_phase08_2k_L13_s${SEED}" \
        --reward-head-layer 13 \
        --reward-heads-dir-override "/mnt/data/generated_heads_probe_L13_base_biasA/sb_bench-PROBE-component" \
        --seed "${SEED}"
  done
fi

# ─── L17 reward (deeper) ────────────────────────────────────────────────────
if [[ " ${LAYERS} " == *" 17 "* ]]; then
  for SEED in ${SEEDS}; do
    echo "▶ Launching L17 seed=${SEED}"
    modal run --detach src/run_modal.py::run_training \
        "${COMMON_FLAGS[@]}" \
        --output-dir "/mnt/data/output_ppo_phase08_2k_L17_s${SEED}" \
        --reward-head-layer 17 \
        --reward-heads-dir-override "/mnt/data/generated_heads_probe_L17_base_biasA/sb_bench-PROBE-component" \
        --seed "${SEED}"
  done
fi

echo "✅ Detached jobs submitted (LAYERS='${LAYERS}' SEEDS='${SEEDS}'). Track via:"
echo "    modal app list"
echo "    modal app logs <app-id>"
