#!/usr/bin/env bash
# Phase 0.8 — value-head warmup seed sweep.
#
# Tests the hypothesis (see /memories/repo/ppo_architecture_audit.md, lesson
# 2026-06-18 "Seed-2 collapse + seed-3 infra failure diagnostics") that the
# seed-to-seed variance in phase08_2k headline outcomes is driven by the
# random init of the Linear(D,1) value head. Step-0 v_loss spans 20x across
# seeds {1,2,3,4,42} and the seed-2 collapse coincides with a 4x higher
# v_loss tail than production — yet KL / parse / reward signals look healthy
# throughout.
#
# Each job here re-runs the production recipe with --value_warmup_steps 30
# (about 8 minutes added to a ~3h training run). The warmup pretrains ONLY
# the value head while the policy weights are held frozen, so the seed-
# dependent random init is washed out before PPO starts modifying LoRA.
#
# Confirmation criterion (post-hoc, see scripts/phase08_seed_trajectories.py):
#   1. Step-0 v_loss should be roughly seed-independent (variance shrinks
#      dramatically from the 0.49 vs 8.75 range).
#   2. Across-seed Δacc std on canonical n=2916 should shrink from
#      ±1.39 pp (without warmup) toward something nearer ±0.5 pp.
#   3. Seed 2 should no longer flip from + to − (b01=2, b10=35 today).
#
# Pre-requisites — identical to phase08_seed_runs.sh:
#   • L13 probe head dir on volume:
#       /mnt/data/generated_heads_probe_L13_base_biasA/sb_bench-PROBE-component
#
# Usage:
#   bash scripts/phase08_value_warmup_seeds.sh                # seeds 1 2 3 4, L=13
#   SEEDS="1 2"  bash scripts/phase08_value_warmup_seeds.sh
#   WARMUP=50    bash scripts/phase08_value_warmup_seeds.sh   # 50 warmup steps
#   WARMUP_LR_MULT=2.0 bash scripts/phase08_value_warmup_seeds.sh
#
# Output dirs (on volume): /mnt/data/output_ppo_phase08_2k_L13_s{SEED}_vwarmup
#
# All jobs are --detach. Each job runs ~3 h + ~8 min warmup on A100-80GB.

set -euo pipefail

# Auto-source venv if not already active.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source debias_env/bin/activate
fi

SEEDS="${SEEDS:-1 2 3 4}"
WARMUP="${WARMUP:-30}"
# 2.0 picked from smoke: with mult=1.0 the v_loss barely moved across 3 steps
# (6.26 → 6.28). 2.0 is a conservative-aggressive default; the hypothesis test
# is "does pre-fitting V actually wash out the seed lottery?", so we want the
# value head to *actually* converge before PPO starts.
WARMUP_LR_MULT="${WARMUP_LR_MULT:-2.0}"

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
    --reward-head-layer 13
    --reward-heads-dir-override "/mnt/data/generated_heads_probe_L13_base_biasA/sb_bench-PROBE-component"
    --value-warmup-steps "${WARMUP}"
    --value-warmup-lr-multiplier "${WARMUP_LR_MULT}"
)

echo "▶ Plan: SEEDS='${SEEDS}'  WARMUP=${WARMUP}  WARMUP_LR_MULT=${WARMUP_LR_MULT}"
for SEED in ${SEEDS}; do
    OUT_DIR="/mnt/data/output_ppo_phase08_2k_L13_s${SEED}_vwarmup"
    echo "▶ Launching value-warmup seed=${SEED}  →  ${OUT_DIR}"
    modal run --detach src/run_modal.py::run_training \
        "${COMMON_FLAGS[@]}" \
        --output-dir "${OUT_DIR}" \
        --seed "${SEED}"
done

echo "✅ Detached jobs submitted (SEEDS='${SEEDS}'). Track via:"
echo "    modal app list"
echo "    modal app logs <app-id>"
echo
echo "After completion, harvest gens + run seed aggregate:"
echo "    bash scripts/phase08_value_warmup_harvest.sh"
