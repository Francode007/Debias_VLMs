#!/usr/bin/env bash
# Phase 0.8 — KLFIX wash-out diagnostic (NEXT_STEPS Action 2).
#
# Re-runs the §3 wash-out per-layer probe-score on the klfix-s1 checkpoint
# (which is now our production recipe per FINAL_REPORT.md). Output appends
# to /mnt/data/phase08_washout/ and the local analyzer merges with the
# existing base/phase08_2k/{corr,bias}Only entries.
#
# Two-step process:
#   1. Generate klfix-s1 VLBias gens (~30 min).
#   2. Score all 11 layers in one forward pass (~15 min).
#
# Re-uses scripts/phase08_washout_score.sh's design; we just add one more
# variant. Outputs:
#   /mnt/data/phase07_vlbiasbench/phase08_2k_klfix_s1_vlbias_gen.jsonl
#   /mnt/data/phase08_washout/phase08_2k_klfix_s1__probe_L<N>__offline_reward.json
#
# Usage:
#   bash scripts/phase08_klfix_washout.sh             # step 1 + 2 sequential
#   STEP=1 bash scripts/phase08_klfix_washout.sh      # only generate gens
#   STEP=2 bash scripts/phase08_klfix_washout.sh      # only score (assumes gens exist)
#
# After both finish, re-run the classifier locally:
#   python scripts/phase08_washout_classify.py \
#       --score-root /mnt/data/phase08_washout \
#       --output-json Phase0.8/washout_diagnostic_klfix.json

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source debias_env/bin/activate
fi

STEP="${STEP:-all}"
SEED="${SEED:-1}"
CKPT="/mnt/data/output_ppo_phase08_2k_L13_s${SEED}_klfix/final_debiased_model"
TAG="phase08_2k_klfix_s${SEED}"

LAYERS_CSV="1,5,9,11,13,17,21,25,29,33,35"

# Build per-layer heads-roots (same as washout_score.sh).
HEADS_ROOTS_CSV=""
IFS=',' read -ra _LAYERS <<< "${LAYERS_CSV}"
for L in "${_LAYERS[@]}"; do
    if [[ -n "${HEADS_ROOTS_CSV}" ]]; then HEADS_ROOTS_CSV+=","; fi
    HEADS_ROOTS_CSV+="/mnt/data/generated_heads_probe_L${L}_base_biasA"
done

if [[ "${STEP}" == "all" || "${STEP}" == "1" ]]; then
    echo "── STEP 1: generate VLBias gens for ${TAG} ──"
    modal run --detach src/run_modal.py::run_vlbiasbench_eval \
        --checkpoint-dir "${CKPT}" \
        --tag "${TAG}" \
        --num-samples 2000
    if [[ "${STEP}" == "1" ]]; then
        echo "✅ STEP 1 launched (detached). Re-run with STEP=2 once gens land."
        exit 0
    fi
    echo "⚠ STEP=all runs synchronously; for parallel use STEP=1 then STEP=2."
fi

if [[ "${STEP}" == "all" || "${STEP}" == "2" ]]; then
    echo "── STEP 2: score ${TAG} across all 11 layers ──"
    modal run --detach src/run_modal.py::run_vlbias_offline_score \
        --variant "${TAG}" \
        --head-type probe \
        --gen-dir /mnt/data/phase07_vlbiasbench \
        --use-kept-heads \
        --output-dir /mnt/data/phase08_washout \
        --reward-base "${CKPT}" \
        --layer-idxs "${LAYERS_CSV}" \
        --heads-roots "${HEADS_ROOTS_CSV}"
fi

echo "✅ KLFIX wash-out jobs launched (detached). Track with:"
echo "    modal app list | grep run_vlbiasbench_eval"
echo "    modal app list | grep run_vlbias_offline_score"
