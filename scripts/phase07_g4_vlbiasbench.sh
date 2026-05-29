#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0.7 — G4a: VLBiasBench close-ended transfer eval.
#
# Runs `run_vlbiasbench_eval` on the Modal volume for:
#   * the vanilla base model (tag=base)
#   * the Phase 0.6 F SVM-DRM checkpoints (ep1-step70, ep1-end)
#   * (optionally) the Phase 0.7 PCA-DRM checkpoints once trained
#
# All Modal invocations are sequential (single GPU lease per run). Each run
# generates a stratified sample drawn across (bbq_axis × condition) so every
# one of the 10 BBQ axes and every condition (ambig/neg/non_neg) stays
# represented, and writes:
#   /mnt/data/phase07_vlbiasbench/<tag>_vlbias_gen.jsonl
#   /mnt/data/phase07_vlbiasbench/<tag>_vlbias_results.json
#
# Usage (default = all stages):
#   bash scripts/phase07_g4_vlbiasbench.sh
#
# Subsets (positional):
#   bash scripts/phase07_g4_vlbiasbench.sh setup base
#   bash scripts/phase07_g4_vlbiasbench.sh svm pca pull
#
# Stage list:
#   setup    — ensure the VLBiasBench parquet is built on the volume
#   base     — vanilla Qwen2.5-VL-3B-Instruct
#   svm      — Phase 0.6 F SVM-DRM checkpoints (ep1-step70 + ep1-end)
#   pca      — Phase 0.7 PCA-DRM checkpoints (ep1-step70 + ep1-end) — skipped
#              if the checkpoint directories don't exist yet
#   pull     — copy result JSONs from the Modal volume to ./Phase0.7/vlbiasbench/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Configuration ────────────────────────────────────────────────────────────
NUM_SAMPLES="${NUM_SAMPLES:-3000}"
CONDITION="${CONDITION:-all}"                       # all | ambig | disambig
QFORMAT="${QFORMAT:-base,scene,scene_text}"         # exclude `text` by default
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
OUTPUT_DIR="/mnt/data/phase07_vlbiasbench"

SVM_OUTDIR="/mnt/data/output_ppo_debiased"        # Phase 0.6 F
PCA_OUTDIR="/mnt/data/output_ppo_phase07_pca_drm" # Phase 0.7 (when trained)

# Modal volume + local landing dir
VOLUME_NAME="debias-vlm-persistent-storage"
LOCAL_DIR="Phase0.7/vlbiasbench"

# ── Stage selection ──────────────────────────────────────────────────────────
STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(setup base svm pca pull)
fi

_run_eval () {
    local ckpt="$1"
    local tag="$2"
    echo ""
    echo "▶▶ run_vlbiasbench_eval  tag=${tag}"
    modal run src/run_modal.py::run_vlbiasbench_eval \
        --checkpoint-dir "${ckpt}" \
        --tag "${tag}" \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size "${BATCH_SIZE}" \
        --num-samples "${NUM_SAMPLES}" \
        --condition "${CONDITION}" \
        --qformat "${QFORMAT}" \
        --seed "${SEED}"
}

for STAGE in "${STAGES[@]}"; do
    case "${STAGE}" in

        setup)
            echo "════ stage: setup ═════════════════════════════════════════════"
            # run_setup re-checks VLBiasBench parquet presence and builds if missing.
            modal run src/run_modal.py::run_setup
            ;;

        base)
            echo "════ stage: base ══════════════════════════════════════════════"
            _run_eval "" "base"
            ;;

        svm)
            echo "════ stage: svm (Phase 0.6 F SVM-DRM) ═════════════════════════"
            # Checkpoint names per actual volume layout (no step70 exists).
            # SVM_CKPTS override lets the user pick a different set.
            SVM_CKPTS="${SVM_CKPTS:-ep1-50pct ep1-end}"
            for CK in ${SVM_CKPTS}; do
                _run_eval "${SVM_OUTDIR}/checkpoint-${CK}" "svm_${CK}"
            done
            ;;

        pca)
            echo "════ stage: pca (Phase 0.7 PCA-DRM) ═══════════════════════════"
            # Defaults: ep1-step80 = best SB-Bench acc (0.9719); ep1-end =
            # final, has conservative-bias signature documented in Phase0.7.md.
            # Override via env: PCA_CKPTS="ep1-step70 ep1-step80 ep1-end".
            PCA_CKPTS="${PCA_CKPTS:-ep1-step80 ep1-end}"
            for CK in ${PCA_CKPTS}; do
                _run_eval "${PCA_OUTDIR}/checkpoint-${CK}" "pca_${CK}"
            done
            ;;

        pull)
            echo "════ stage: pull ══════════════════════════════════════════════"
            mkdir -p "${LOCAL_DIR}"
            # Pull the entire results directory from the volume.
            modal volume get "${VOLUME_NAME}" \
                "${OUTPUT_DIR#/mnt/data/}" "${LOCAL_DIR}" --force
            echo "✅ Results pulled to ${LOCAL_DIR}/"
            ls -la "${LOCAL_DIR}" || true
            ;;

        *)
            echo "❌ Unknown stage: ${STAGE}"
            echo "   Valid: setup base svm pca pull"
            exit 2
            ;;
    esac
done

echo ""
echo "✅ phase07_g4_vlbiasbench done (stages: ${STAGES[*]})"
