#!/usr/bin/env bash
# Phase 0.8 Strategic Plan §3 — Wash-out diagnostic SCORE step.
#
# Re-scores existing VLBias generations against the per-layer probe heads
# at every layer in {1,5,9,11,13,17,21,25,29,33,35}, for each model variant.
# The hidden states are extracted from `--reward-base`, so to get the
# *trained policy's* per-layer activations we pass each variant's local
# checkpoint as reward_base.
#
# Outputs land in /mnt/data/phase08_washout/<variant>__layer<L>__offline_reward.json.
#
# Pre-requisites:
#   1. Per-layer heads built — see phase08_washout_make_heads.sh.
#   2. VLBias generations already produced for each variant (phase07/phase08
#      generate steps).
#
# Usage:
#   bash scripts/phase08_washout_score.sh
#
# After this finishes, run the local analyzer:
#   python -m scripts.phase08_washout_classify
#
set -euo pipefail

LAYERS=(1 5 9 11 13 17 21 25 29 33 35)

# variant_label : reward_base path : gen_dir
# (reward_base must point at a HF-loadable model; LoRA-merged checkpoints
#  produced by save_final_model satisfy this.)
declare -a VARIANTS=(
  "vanilla:Qwen/Qwen2.5-VL-3B-Instruct:/mnt/data/phase07_vlbiasbench"
  "phase08_full:/mnt/data/output_ppo_phase08_2k/final_debiased_model:/mnt/data/phase08_vlbiasbench"
  "phase08_corrOnly:/mnt/data/output_ppo_phase08_2k_corrOnly/final_debiased_model:/mnt/data/phase08_corrOnly_vlbiasbench"
  "phase08_biasOnly:/mnt/data/output_ppo_phase08_2k_biasOnly/final_debiased_model:/mnt/data/phase08_biasOnly_vlbiasbench"
)

OUT_DIR="/mnt/data/phase08_washout"

for entry in "${VARIANTS[@]}"; do
    VAR="${entry%%:*}"; rest="${entry#*:}"
    BASE="${rest%%:*}"; GEN_DIR="${rest#*:}"
    for L in "${LAYERS[@]}"; do
        HEADS_ROOT="/mnt/data/generated_heads_probe_L${L}_base_biasA"
        echo "── ${VAR} | L${L} ──"
        modal run src/run_modal.py::run_vlbias_offline_score \
            --variant "${VAR}" \
            --head-type probe \
            --gen-dir "${GEN_DIR}" \
            --heads-root "${HEADS_ROOT}" \
            --use-kept-heads true \
            --output-dir "${OUT_DIR}/L${L}" \
            --reward-base "${BASE}" \
            --layer-idx "${L}" || echo "⚠ failed ${VAR} L${L}, continuing"
    done
done

echo "✅ Wash-out scoring complete → ${OUT_DIR}"
