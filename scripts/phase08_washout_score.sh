#!/usr/bin/env bash
# Phase 0.8 Strategic Plan §3 — Wash-out diagnostic SCORE step (FAST PATH).
#
# One forward pass per variant scores ALL 11 layers at once (Phase 0.8 §3
# multi-layer mode in score_vlbias_offline.py). 4 modal calls total instead
# of the legacy 44.
#
# Each variant uses its own checkpoint as `--reward-base` so hidden states
# come from the trained policy, while the per-layer probe heads
# (built on the BASE model) stay frozen.
#
# Outputs land in /mnt/data/phase08_washout/<variant>__probe_L<N>__offline_reward.json
# (one JSON per layer, written by the same forward pass).
#
# Pre-requisites: per-layer probe head dirs already exist on the volume:
#   /mnt/data/generated_heads_probe_L${L}_base_biasA/sb_bench-PROBE-component/
# (built earlier via phase08_a3_probe_as_head.sh / phase08_washout_make_heads.sh).
#
# Usage:
#   bash scripts/phase08_washout_score.sh
#
# After this finishes, run the local analyzer:
#   python scripts/phase08_washout_classify.py \
#       --score-root /mnt/data/phase08_washout \
#       --output-json Phase0.8/washout_diagnostic.json
set -euo pipefail

LAYERS_CSV="1,5,9,11,13,17,21,25,29,33,35"

# Build the per-layer head-roots list (one root per layer, comma-separated).
# Each root is the directory passed to phase08_a3 / probe_to_head; the
# function appends sb_bench-PROBE-component/ internally.
HEADS_ROOTS_CSV=""
IFS=',' read -ra _LAYERS <<< "${LAYERS_CSV}"
for L in "${_LAYERS[@]}"; do
    if [[ -n "${HEADS_ROOTS_CSV}" ]]; then HEADS_ROOTS_CSV+=","; fi
    HEADS_ROOTS_CSV+="/mnt/data/generated_heads_probe_L${L}_base_biasA"
done

# variant_label : reward_base path : gen_dir
declare -a VARIANTS=(
  "base:Qwen/Qwen2.5-VL-3B-Instruct:/mnt/data/phase07_vlbiasbench"
  "phase08_2k:/mnt/data/output_ppo_phase08_2k/final_debiased_model:/mnt/data/phase07_vlbiasbench"
  "phase08_2k_corrOnly:/mnt/data/output_ppo_phase08_2k_corrOnly/final_debiased_model:/mnt/data/phase07_vlbiasbench"
  "phase08_2k_biasOnly:/mnt/data/output_ppo_phase08_2k_biasOnly/final_debiased_model:/mnt/data/phase07_vlbiasbench"
)

# Set SKIP_VARIANTS="base" (or comma-list) to resume after a partial run.
SKIP_VARIANTS="${SKIP_VARIANTS:-}"

OUT_DIR="/mnt/data/phase08_washout"

for entry in "${VARIANTS[@]}"; do
    VAR="${entry%%:*}"; rest="${entry#*:}"
    BASE="${rest%%:*}"; GEN_DIR="${rest#*:}"
    if [[ -n "${SKIP_VARIANTS}" && ",${SKIP_VARIANTS}," == *",${VAR},"* ]]; then
        echo "── ${VAR}  SKIPPED (in SKIP_VARIANTS) ─────────────────────────"
        continue
    fi
    echo "── ${VAR}  ALL LAYERS ─────────────────────────────────────────"
    modal run src/run_modal.py::run_vlbias_offline_score \
        --variant "${VAR}" \
        --head-type probe \
        --gen-dir "${GEN_DIR}" \
        --use-kept-heads \
        --output-dir "${OUT_DIR}" \
        --reward-base "${BASE}" \
        --layer-idxs "${LAYERS_CSV}" \
        --heads-roots "${HEADS_ROOTS_CSV}" || echo "⚠ failed ${VAR}, continuing"
done

echo "✅ Wash-out scoring complete → ${OUT_DIR}"
