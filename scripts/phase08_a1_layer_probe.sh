#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0.8 A1 — Per-layer linear probes on Qwen2.5-VL-3B hidden states.
#
# Pipeline (per variant):
#   1. genholdout  — <variant>-policy generation on qformat=text (the bias-blind
#                    selection holdout). Each variant gets its OWN holdout:
#                    hidden states must come from that variant's weights, so
#                    the records the prober sees must be the records that
#                    variant emitted.
#   2. probe       — extract per-layer hidden states at `post_letter` from
#                    <variant>_vlbias_gen.jsonl + the variant's qformat=text
#                    holdout, train P1/P2/P3 logistic-regression probes per layer.
#   3. pull        — fetch all variants' JSON + PNG to Phase0.8/probe_results/.
#
# Inputs on volume (must already exist):
#   /mnt/data/phase07_vlbiasbench/<variant>_vlbias_gen.jsonl   (Phase 0.7 G4a)
#   /mnt/data/vlbiasbench_data/vlbiasbench_close_ended.parquet
#   PPO checkpoint dirs for the non-base variants (see _ckpt_for_variant).
#
# Outputs on volume:
#   /mnt/data/phase07_vlbiasbench/<variant>_qformat_text_vlbias_gen.jsonl
#   /mnt/data/phase08_probe_results/<variant>_layerwise_probe.json (+ .png)
#
# Usage:
#   bash scripts/phase08_a1_layer_probe.sh                            # default VARIANTS, all stages
#   bash scripts/phase08_a1_layer_probe.sh probe pull                 # skip genholdout
#   VARIANTS="svm_ep1-end" bash scripts/phase08_a1_layer_probe.sh     # one variant
#   bash scripts/phase08_a1_layer_probe.sh pull                       # local pull only
#
# Default VARIANTS deliberately EXCLUDES `base` (already completed →
# Phase0.8/probe_results/base_layerwise_probe.json). Pass
# VARIANTS=base explicitly to re-run it.
#
# Env knobs:
#   VARIANTS         — space-separated variant tags
#                      (default: svm_ep1-50pct svm_ep1-end pca_ep1-step80 pca_ep1-end)
#   BATCH            — extraction batch size       (default: 4)
#   MAX_PER_CELL     — primary stratification cap  (default: 30 → ~900 records)
#   MAX_HOLDOUT      — holdout cap                  (default: 100)
#   HOLDOUT_SAMPLES  — n records to generate for the holdout (default: 200)
#   MAX_PIXELS       — Qwen2.5-VL pixel cap        (default: 0 = uncapped)
#   TOKEN_POSITION   — post_letter | pre_letter    (default: post_letter)
#   CACHE_NPZ_ROOT   — optional cache dir for hidden states (default: empty)
#   SVM_OUTDIR       — root of SVM PPO checkpoints
#   PCA_OUTDIR       — root of PCA PPO checkpoints
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

VARIANTS="${VARIANTS:-svm_ep1-50pct svm_ep1-end pca_ep1-step80 pca_ep1-end}"
BATCH="${BATCH:-4}"
MAX_PER_CELL="${MAX_PER_CELL:-30}"
MAX_HOLDOUT="${MAX_HOLDOUT:-100}"
HOLDOUT_SAMPLES="${HOLDOUT_SAMPLES:-200}"
MAX_PIXELS="${MAX_PIXELS:-262144}"  # 512x512 cap (Phase 0.8 §4½.15); set 0 to disable
TOKEN_POSITION="${TOKEN_POSITION:-post_letter}"
CACHE_NPZ_ROOT="${CACHE_NPZ_ROOT:-}"

SVM_OUTDIR="${SVM_OUTDIR:-/mnt/data/output_ppo_debiased}"
PCA_OUTDIR="${PCA_OUTDIR:-/mnt/data/output_ppo_phase07_pca_drm}"

VOLUME_NAME="debias-vlm-persistent-storage"
GEN_DIR="/mnt/data/phase07_vlbiasbench"
PROBE_DIR="/mnt/data/phase08_probe_results"
LOCAL_DIR="Phase0.8/probe_results"

# Resolve a variant tag → checkpoint dir ("" for base).
_ckpt_for_variant () {
    case "$1" in
        base)               echo "" ;;
        svm_ep1-50pct)      echo "${SVM_OUTDIR}/checkpoint-ep1-50pct" ;;
        svm_ep1-end)        echo "${SVM_OUTDIR}/checkpoint-ep1-end" ;;
        pca_ep1-step80)     echo "${PCA_OUTDIR}/checkpoint-ep1-step80" ;;
        pca_ep1-end)        echo "${PCA_OUTDIR}/checkpoint-ep1-end" ;;
        *)
            echo "❌ Unknown variant tag: $1 (no checkpoint mapping)" >&2
            return 1
            ;;
    esac
}

STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(genholdout probe pull)
fi

for STAGE in "${STAGES[@]}"; do
    case "${STAGE}" in

        genholdout)
            echo "════ stage: genholdout (per variant, qformat=text) ════════════════════"
            for V in ${VARIANTS}; do
                CKPT="$(_ckpt_for_variant "${V}")"
                HOLDOUT_TAG="${V}_qformat_text"
                echo ""
                echo "▶▶ genholdout  variant=${V}  ckpt=${CKPT:-<base>}  tag=${HOLDOUT_TAG}"
                modal run src/run_modal.py::run_vlbiasbench_eval \
                    --checkpoint-dir "${CKPT}" \
                    --tag "${HOLDOUT_TAG}" \
                    --output-dir "${GEN_DIR}" \
                    --num-samples "${HOLDOUT_SAMPLES}" \
                    --condition all \
                    --qformat text \
                    --batch-size 8
            done
            ;;

        probe)
            echo "════ stage: probe (per variant) ═══════════════════════════════════════"
            for V in ${VARIANTS}; do
                HOLDOUT_TAG="${V}_qformat_text"
                echo ""
                echo "▶▶ probe  variant=${V}  holdout_tag=${HOLDOUT_TAG}"
                CMD=( modal run src/run_modal.py::run_layer_probe
                      --variant "${V}"
                      --gen-dir "${GEN_DIR}"
                      --output-dir "${PROBE_DIR}"
                      --batch-size "${BATCH}"
                      --max-per-cell "${MAX_PER_CELL}"
                      --max-pixels "${MAX_PIXELS}"
                      --holdout-tag "${HOLDOUT_TAG}"
                      --max-holdout "${MAX_HOLDOUT}"
                      --token-position "${TOKEN_POSITION}" )
                if [[ -n "${CACHE_NPZ_ROOT}" ]]; then
                    CMD+=( --cache-npz "${CACHE_NPZ_ROOT}/${V}.npz" )
                fi
                "${CMD[@]}"
            done
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
echo "✅ phase08_a1_layer_probe.sh complete (stages: ${STAGES[*]}, variants: ${VARIANTS})"
