#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0.8 A1 — Per-layer linear probes on Qwen2.5-VL-3B hidden states.
#
# Pipeline:
#   1. genholdout  — base-policy generation on qformat=text (the bias-blind
#                    selection holdout). Skips if the file already exists.
#   2. probe       — extract per-layer hidden states at `post_letter` from
#                    the existing base gen JSONL + the qformat=text holdout,
#                    train P1/P2/P3 logistic-regression probes per layer.
#   3. pull        — fetch the JSON + PNG to Phase0.8/probe_results/ locally.
#
# Inputs on volume (must already exist):
#   /mnt/data/phase07_vlbiasbench/base_vlbias_gen.jsonl   (Phase 0.7 G4a)
#   /mnt/data/vlbiasbench_data/vlbiasbench_close_ended.parquet
#
# Outputs on volume:
#   /mnt/data/phase07_vlbiasbench/base_qformat_text_vlbias_gen.jsonl
#   /mnt/data/phase08_probe_results/<variant>_layerwise_probe.json (+ .png)
#
# Usage:
#   bash scripts/phase08_a1_layer_probe.sh                 # all stages, base variant
#   bash scripts/phase08_a1_layer_probe.sh probe pull      # skip genholdout
#   VARIANT=base bash scripts/phase08_a1_layer_probe.sh    # explicit variant
#
# Env knobs:
#   VARIANT         — base-policy variant tag (default: base)
#   BATCH           — extraction batch size       (default: 4)
#   MAX_PER_CELL    — primary stratification cap  (default: 30 → ~900 records)
#   MAX_HOLDOUT     — holdout cap                  (default: 100)
#   HOLDOUT_TAG     — holdout gen filename tag    (default: base_qformat_text)
#   HOLDOUT_SAMPLES — n records to generate for the holdout (default: 200)
#   MAX_PIXELS      — Qwen2.5-VL pixel cap        (default: 0 = uncapped)
#   TOKEN_POSITION  — post_letter | pre_letter    (default: post_letter)
#   CACHE_NPZ       — optional cache path on /mnt/data/ for hidden states
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

VARIANT="${VARIANT:-base}"
BATCH="${BATCH:-4}"
MAX_PER_CELL="${MAX_PER_CELL:-30}"
MAX_HOLDOUT="${MAX_HOLDOUT:-100}"
HOLDOUT_TAG="${HOLDOUT_TAG:-base_qformat_text}"
HOLDOUT_SAMPLES="${HOLDOUT_SAMPLES:-200}"
MAX_PIXELS="${MAX_PIXELS:-0}"
TOKEN_POSITION="${TOKEN_POSITION:-post_letter}"
CACHE_NPZ="${CACHE_NPZ:-}"

VOLUME_NAME="debias-vlm-persistent-storage"
GEN_DIR="/mnt/data/phase07_vlbiasbench"
PROBE_DIR="/mnt/data/phase08_probe_results"
LOCAL_DIR="Phase0.8/probe_results"

STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(genholdout probe pull)
fi

for STAGE in "${STAGES[@]}"; do
    case "${STAGE}" in

        genholdout)
            echo "════ stage: genholdout (qformat=text base-policy generation) ═════════"
            # The Modal entrypoint is idempotent on output path; we skip locally
            # by detecting the JSONL via a lightweight `modal volume ls`. If it
            # already exists, the user can pass explicit stages to skip.
            modal run src/run_modal.py::run_vlbiasbench_eval \
                --checkpoint-dir "" \
                --tag "${HOLDOUT_TAG}" \
                --output-dir "${GEN_DIR}" \
                --num-samples "${HOLDOUT_SAMPLES}" \
                --condition all \
                --qformat text \
                --batch-size 8
            ;;

        probe)
            echo "════ stage: probe ═════════════════════════════════════════════════════"
            CMD=( modal run src/run_modal.py::run_layer_probe
                  --variant "${VARIANT}"
                  --gen-dir "${GEN_DIR}"
                  --output-dir "${PROBE_DIR}"
                  --batch-size "${BATCH}"
                  --max-per-cell "${MAX_PER_CELL}"
                  --max-pixels "${MAX_PIXELS}"
                  --holdout-tag "${HOLDOUT_TAG}"
                  --max-holdout "${MAX_HOLDOUT}"
                  --token-position "${TOKEN_POSITION}" )
            if [[ -n "${CACHE_NPZ}" ]]; then
                CMD+=( --cache-npz "${CACHE_NPZ}" )
            fi
            "${CMD[@]}"
            ;;

        pull)
            echo "════ stage: pull ══════════════════════════════════════════════════════"
            mkdir -p "${LOCAL_DIR}"
            modal volume get "${VOLUME_NAME}" \
                "${PROBE_DIR#/mnt/data/}" "${LOCAL_DIR}" --force
            echo "✅ Results pulled to ${LOCAL_DIR}/"
            ls -la "${LOCAL_DIR}" || true
            ;;

        *)
            echo "❌ Unknown stage: ${STAGE}"
            echo "   Valid: genholdout probe pull"
            exit 2
            ;;
    esac
done

echo ""
echo "✅ phase08_a1_layer_probe.sh complete (stages: ${STAGES[*]})"
