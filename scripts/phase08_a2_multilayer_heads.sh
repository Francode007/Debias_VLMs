#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0.8 — A2: Multi-layer DRM head sweep at L ∈ {9, 11, 13}.
#
# Per Phase0.8/PREREG.md, the A1 layerwise probes selected L9 (primary), L13
# (secondary), L11 (tertiary) as the bias-dominant candidates for re-building
# the SB-Bench reward heads. This script orchestrates, per layer:
#
#   extract   — re-extract SB-Bench post_letter embeddings at L=N for both
#               train and test splits (run_inference --layer-idx N).
#   heads     — fit PCA(K=50) + per-axis SVM heads on the L=N train embeddings
#               (run_drm_generation --layer-idx N).
#   eval      — held-out chosen>rejected accuracy per head on the test split
#               (run_drm_eval --layer-idx N) — sanity / G-A2 input.
#   score     — re-score the existing Phase 0.7 VLBiasBench gen JSONLs for all
#               5 variants using the new L=N heads (run_vlbias_offline_score).
#   pull      — modal volume get → Phase0.8/a2_results/.
#
# Inputs already on volume:
#   /mnt/data/embeddings_output                              (SB-Bench cached dataset)
#   /mnt/data/phase07_vlbiasbench/<variant>_vlbias_gen.jsonl (G4 generations)
#   /mnt/data/vlbiasbench_data/*                             (parquet + images)
#
# Outputs per layer N:
#   /mnt/data/embeddings_output_letter_post_letter_L${N}/    (embeddings)
#   /mnt/data/generated_heads_letter_post_letter_L${N}/      (heads + drm_head_eval_*.json)
#   /mnt/data/phase08_offline_reward_L${N}/<variant>__<head>_L${N}__offline_reward.json
#
# Usage:
#   bash scripts/phase08_a2_multilayer_heads.sh                    # all layers, all stages
#   bash scripts/phase08_a2_multilayer_heads.sh extract heads      # build only
#   bash scripts/phase08_a2_multilayer_heads.sh score pull         # assume heads built, score+pull
#   LAYERS="9" bash scripts/phase08_a2_multilayer_heads.sh         # just L9 first (cheapest path to a verdict)
#
# Env knobs:
#   LAYERS       — space-separated layer indices (default: "9 11 13", per PREREG)
#   VARIANTS     — variants to score offline (default: 5 phase 0.7 variants)
#   HEADS        — head types to score (default: "svm pca")
#   BATCH        — embed batch size for VLBiasBench scoring (default: 4)
#   MAX_PER_CELL — stratified samples per (bbq_axis x condition) for scoring (default: 30 → 900 records / variant)
#   MAX_PIXELS   — image pixel cap (default: 0 = uncapped, matches generator)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

LAYERS="${LAYERS:-9 11 13}"
VARIANTS="${VARIANTS:-base svm_ep1-50pct svm_ep1-end pca_ep1-step80 pca_ep1-end}"
HEADS="${HEADS:-svm pca}"
BATCH="${BATCH:-4}"
MAX_PER_CELL="${MAX_PER_CELL:-30}"
MAX_PIXELS="${MAX_PIXELS:-262144}"  # 512x512 cap (Phase 0.8 §4½.15); set 0 to disable
COMPLETION_FORMAT="${COMPLETION_FORMAT:-letter}"
TOKEN_POSITION="${TOKEN_POSITION:-post_letter}"

VOLUME_NAME="debias-vlm-persistent-storage"
LOCAL_DIR="Phase0.8/a2_results"

# Stage parsing — anything that's a recognised stage joins STAGES, else
# treated as a free-form positional override.
ARGS=("${@:+$@}")
STAGES=()
for a in "${ARGS[@]:+${ARGS[@]}}"; do
    case "$a" in
        extract|heads|eval|score|pull) STAGES+=("$a") ;;
        *) echo "❌ Unknown arg: $a (valid stages: extract heads eval score pull)"; exit 2 ;;
    esac
done
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(extract heads eval score pull)
fi

echo "════ Phase 0.8 A2 multi-layer head sweep ═══════════════════════════════"
echo "  layers       : ${LAYERS}"
echo "  variants     : ${VARIANTS}"
echo "  heads        : ${HEADS}"
echo "  stages       : ${STAGES[*]}"
echo "  completion   : ${COMPLETION_FORMAT} / ${TOKEN_POSITION}"
echo "  max_per_cell : ${MAX_PER_CELL}"

for L in ${LAYERS}; do
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  LAYER L=${L}"
    echo "════════════════════════════════════════════════════════════════════"
    HEADS_ROOT="/mnt/data/generated_heads_${COMPLETION_FORMAT}_${TOKEN_POSITION}_L${L}"
    OUTPUT_DIR="/mnt/data/phase08_offline_reward_L${L}"

    for STAGE in "${STAGES[@]}"; do
        case "${STAGE}" in

            extract)
                echo ""
                echo "▶▶ [L=${L}] extract: train split"
                modal run src/run_modal.py::run_inference \
                    --dataset sb_bench \
                    --completion-format "${COMPLETION_FORMAT}" \
                    --token-position "${TOKEN_POSITION}" \
                    --layer-idx "${L}" \
                    --split train
                echo ""
                echo "▶▶ [L=${L}] extract: test split"
                modal run src/run_modal.py::run_inference \
                    --dataset sb_bench \
                    --completion-format "${COMPLETION_FORMAT}" \
                    --token-position "${TOKEN_POSITION}" \
                    --layer-idx "${L}" \
                    --split test
                ;;

            heads)
                echo ""
                echo "▶▶ [L=${L}] generate PCA+SVM heads"
                modal run src/run_modal.py::run_drm_generation \
                    --completion-format "${COMPLETION_FORMAT}" \
                    --token-position "${TOKEN_POSITION}" \
                    --layer-idx "${L}"
                ;;

            eval)
                for HT in ${HEADS}; do
                    echo ""
                    echo "▶▶ [L=${L}] held-out DRM head eval [${HT}]"
                    modal run src/run_modal.py::run_drm_eval \
                        --completion-format "${COMPLETION_FORMAT}" \
                        --token-position "${TOKEN_POSITION}" \
                        --layer-idx "${L}" \
                        --head-type "${HT}" \
                        --split test
                done
                ;;

            score)
                HEADS_CSV=$(echo "${HEADS}" | tr ' ' ',')
                for V in ${VARIANTS}; do
                    echo ""
                    echo "▶▶ [L=${L}] offline reward score  variant=${V}  heads=${HEADS_CSV}"
                    modal run src/run_modal.py::run_vlbias_offline_score \
                        --variant "${V}" \
                        --head-type "${HEADS_CSV}" \
                        --heads-root "${HEADS_ROOT}" \
                        --output-dir "${OUTPUT_DIR}" \
                        --batch-size "${BATCH}" \
                        --max-per-cell "${MAX_PER_CELL}" \
                        --max-pixels "${MAX_PIXELS}" \
                        --token-position "${TOKEN_POSITION}" \
                        --layer-idx "${L}"
                done
                ;;

            pull)
                mkdir -p "${LOCAL_DIR}"
                echo ""
                echo "▶▶ [L=${L}] pull head-eval JSONs"
                modal volume get "${VOLUME_NAME}" \
                    "${HEADS_ROOT#/mnt/data/}" "${LOCAL_DIR}" --force || true
                echo "▶▶ [L=${L}] pull offline-reward JSONs"
                modal volume get "${VOLUME_NAME}" \
                    "${OUTPUT_DIR#/mnt/data/}" "${LOCAL_DIR}" --force || true
                ;;

            *)
                echo "❌ Unknown stage: ${STAGE}"
                exit 2
                ;;
        esac
    done
done

echo ""
echo "✅ phase08_a2_multilayer_heads.sh complete"
echo "   layers=${LAYERS}  stages=${STAGES[*]}"
[[ " ${STAGES[*]} " == *" pull "* ]] && ls -la "${LOCAL_DIR}" 2>/dev/null || true
