#!/usr/bin/env bash
# Phase 0.8 — KL-anomaly fix smoke test (s1 only).
#
# Goal: confirm that the three KL-controller / value-head fixes eliminate the
# s1 tail-KL=0.044 anomaly observed in the 2026-06-18 vwarmup sweep.
#
# Fixes layered on top of the existing vwarmup recipe:
#   1. Value head zero-init  (src/modules/rl_components/phase08_ppo_trainer.py)
#      — removes per-seed asymmetric init, V(s_0)=0 for all seeds.
#   2. --value-clip-range 0.2  — caps |v_new − v_old|, prevents value runaway.
#   3. --kl-adapt-rate 0.3     — β reacts 3× faster (10%→30%/step) so the
#      controller catches drift around step 100 instead of step 200.
#
# Pass criterion: s1 tail-KL (mean over last 20 PPO steps) ≤ 0.005, i.e.
# within ~5× of the s2/s3/s4 baseline (≈0.0008–0.0012), down from 50×.
# If the fix works, we then launch the full 4-seed re-sweep.
#
# Usage:
#   bash scripts/phase08_klfix_s1_smoke.sh
#   SEEDS="1 4"  bash scripts/phase08_klfix_s1_smoke.sh   # test extremes
set -euo pipefail

SEEDS="${SEEDS:-1}"
WARMUP="${WARMUP:-30}"
WARMUP_LR_MULT="${WARMUP_LR_MULT:-2.0}"

COMMON_FLAGS=(
    --epochs 1
    --dataset sb_bench --model-family qwen
    --kl-beta 0.1 --target-kl 0.02
    --kl-adapt-rate 0.3                           # fix #3: 3× faster β response
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
    --value-clip-range 0.2                        # fix #2: value-update clipping
)

echo "▶ Plan: SEEDS='${SEEDS}'  WARMUP=${WARMUP}  WARMUP_LR_MULT=${WARMUP_LR_MULT}"
echo "▶ Fixes: value-head zero-init (code), --value-clip-range 0.2, --kl-adapt-rate 0.3"
for SEED in ${SEEDS}; do
    OUT_DIR="/mnt/data/output_ppo_phase08_2k_L13_s${SEED}_klfix"
    echo "▶ Launching klfix seed=${SEED}  →  ${OUT_DIR}"
    nohup modal run --detach src/run_modal.py::run_training \
        "${COMMON_FLAGS[@]}" \
        --output-dir "${OUT_DIR}" \
        --seed "${SEED}" \
        > "/tmp/klfix_s${SEED}.log" 2>&1 &
    echo "  pid=$!  log=/tmp/klfix_s${SEED}.log"
done
wait
echo "✅ Detached jobs submitted (SEEDS='${SEEDS}'). Track via:"
echo "    modal app list"
echo "    tail -f /tmp/klfix_s*.log"
echo
echo "After completion:"
echo "  modal volume get debias-vlm-persistent-storage /output_ppo_phase08_2k_L13_s1_klfix/metrics.jsonl Phase0.8/seed_diag/klfix/s1_klfix_metrics.jsonl --force"
echo "  python scripts/phase08_kl_anomaly_diagnostic.py  # adapt path to read s1_klfix_metrics.jsonl"
