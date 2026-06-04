#!/usr/bin/env bash
# Phase 0.8 Strategic Plan §3 — Wash-out diagnostic.
#
# For each layer L in the §4½.14 sweep, ensure a probe-as-head package exists
# under /mnt/data/generated_heads_probe_L${L}_base_biasA/ so that
# score_vlbias_offline.py can be invoked with --layer_idx L on each variant.
#
# Pre-existing artifact: Phase0.8/a3_results/base_L${L}_probe_weights_biasA.npz
# Output:                /mnt/data/generated_heads_probe_L${L}_base_biasA/
#                          sb_bench-PROBE-component/sb_bench-PROBE-component0.pth
#                        + kept_heads_probe.json
#
# Usage:
#   bash scripts/phase08_washout_make_heads.sh           # build heads for all 11 layers
#   bash scripts/phase08_washout_make_heads.sh 13 17     # only those layers
#
# Assumes the npz files have already been uploaded to the volume under
# /mnt/data/phase08_a3_results/. If they live only locally under
# Phase0.8/a3_results/, upload them with `modal volume put` first.

set -euo pipefail

# Default sweep matches Multi_layer_head_analysis.md §4½.14
DEFAULT_LAYERS=(1 5 9 11 13 17 21 25 29 33 35)
LAYERS=("${@:-${DEFAULT_LAYERS[@]}}")

NPZ_DIR="/mnt/data/phase08_a3_results"
OUT_ROOT="/mnt/data/generated_heads_probe"

for L in "${LAYERS[@]}"; do
    NPZ="${NPZ_DIR}/base_L${L}_probe_weights_biasA.npz"
    OUT="${OUT_ROOT}_L${L}_base_biasA"
    echo "── Layer ${L} ──────────────────────────────────────────"
    echo "  npz: ${NPZ}"
    echo "  out: ${OUT}"
    modal run src/run_modal.py::run_probe_to_head \
        --weights-npz "${NPZ}" \
        --out-dir "${OUT}" \
        --case-name sb_bench \
        --head-name PROBE \
        --normalize
done

echo "✅ Per-layer probe heads built for layers: ${LAYERS[*]}"
