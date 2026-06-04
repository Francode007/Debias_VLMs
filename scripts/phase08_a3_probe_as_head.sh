#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0.8 — A3 Tier 0: A1-probe-as-head decision gate.
#
# Per `Phase0.8/PREREG.md` A3 section: if the A1 LogReg probe at L9 trained
# on P3 (is_good_C) achieves 0.978 holdout accuracy, then converting its
# weight vector directly into a reward head should produce a meaningful
# bias signal on VLBiasBench. This Tier 0 script runs the minimum experiment
# needed to ACCEPT or FALSIFY that hypothesis on the `base` variant only.
#
# Stages:
#   probe   — re-run probe_layers.py on the `base` variant with
#             --save_weights_at_layer 9 (or other ${LAYER}) AND
#             --cache_npz (so subsequent residual / multi-layer follow-ups
#             reuse the cached hidden states).
#   head    — convert the saved npz into a reward-head directory.
#   score   — re-score the existing Phase 0.7 base_vlbias_gen.jsonl with
#             the probe head (single head, no SVM/PCA in the same pass).
#   pull    — modal volume get → Phase0.8/a3_results/.
#
# Optional `residual` stage (run AFTER head) builds a second head with the
# top-2 L9 PCs (the variance-nuisance directions) projected out — tests
# whether removing the variance-vs-bias confound sharpens the reward.
#
# Usage:
#   bash scripts/phase08_a3_probe_as_head.sh                 # all stages, base, L9
#   bash scripts/phase08_a3_probe_as_head.sh probe head      # build only, skip scoring
#   bash scripts/phase08_a3_probe_as_head.sh head score pull # assume probe already ran
#   VARIANT=svm_ep1-end LAYER=9 bash ... probe head score pull
#   LAYER=13 bash scripts/phase08_a3_probe_as_head.sh        # try L13 instead
#   bash scripts/phase08_a3_probe_as_head.sh residual score pull  # add residual variant
#
# Env knobs:
#   VARIANT       — which variant to probe (default: base). Tier 0 = base only.
#   LAYER         — single layer index to dump probe weights at (default: 9).
#   BATCH         — probe extraction batch size (default: 4).
#   MAX_PER_CELL  — stratified samples per (axis × condition) on probe primary (default: 30).
#   MAX_PIXELS    — image pixel cap on probe extraction (default: 0, uncapped).
#   HOLDOUT_TAG   — gen JSONL tag used as holdout for the probe (default: base_qformat_text).
#   MAX_HOLDOUT   — holdout cap (default: 100).
#   SCORE_MAX_PER_CELL — stratified samples for VLBiasBench scoring (default: 30 → 900 records/variant).
#   SCORE_BATCH   — embedding batch size for scoring (default: 4).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

VARIANT="${VARIANT:-base}"
LAYER="${LAYER:-9}"
# LAYERS overrides LAYER for the `probe` stage only (one extraction can dump
# weights at multiple layers since hidden states for all 37 layers are cached).
# Head/score stages still operate on a single ${LAYER} per invocation.
LAYERS="${LAYERS:-${LAYER}}"
BATCH="${BATCH:-4}"
MAX_PER_CELL="${MAX_PER_CELL:-30}"
MAX_PIXELS="${MAX_PIXELS:-262144}"  # 512x512 cap (Phase 0.8 §4½.15); set 0 to disable
HOLDOUT_TAG="${HOLDOUT_TAG:-base_qformat_text}"
MAX_HOLDOUT="${MAX_HOLDOUT:-100}"
SCORE_MAX_PER_CELL="${SCORE_MAX_PER_CELL:-30}"
SCORE_BATCH="${SCORE_BATCH:-4}"
TOKEN_POSITION="${TOKEN_POSITION:-post_letter}"
# PROBE_TASK selects which probe direction the `head` / `score` stages operate
# on. Each task corresponds to one of the .npz suffixes written by the
# multi-task §6b loop in probe_layers.py:
#   is_good_C     → <variant>_L{N}_probe_weights.npz          (P3, legacy A1 task)
#   is_C          → <variant>_L{N}_probe_weights_isC.npz      (P4, Move 1 companion)
#   bias_aligned  → <variant>_L{N}_probe_weights_biasA.npz    (P5, Path L9-A.1)
#   correct       → <variant>_L{N}_probe_weights_corr.npz     (P6, Path L9-A.3 control)
PROBE_TASK="${PROBE_TASK:-is_good_C}"
case "${PROBE_TASK}" in
    is_good_C)    TASK_SUFFIX=""      ; TASK_TAG="" ;;
    is_C)         TASK_SUFFIX="_isC"  ; TASK_TAG="_isC" ;;
    bias_aligned) TASK_SUFFIX="_biasA"; TASK_TAG="_biasA" ;;
    correct)      TASK_SUFFIX="_corr" ; TASK_TAG="_corr" ;;
    *) echo "❌ Unknown PROBE_TASK=${PROBE_TASK} (valid: is_good_C is_C bias_aligned correct)"; exit 2 ;;
esac

VOLUME_NAME="debias-vlm-persistent-storage"
LOCAL_DIR="Phase0.8/a3_results"

# Remote (Modal volume) paths
PROBE_DIR="/mnt/data/phase08_probe_results"
PROBE_CACHE="${PROBE_DIR}/${VARIANT}_probe_hs.npz"
# NPZ used by head/score stages (task-aware). Probe stage writes ALL task NPZs
# regardless of PROBE_TASK — the loop in probe_layers.py emits four files per layer.
PROBE_NPZ="${PROBE_DIR}/${VARIANT}_L${LAYER}_probe_weights${TASK_SUFFIX}.npz"
PROBE_NPZ_ISC="${PROBE_DIR}/${VARIANT}_L${LAYER}_probe_weights_isC.npz"
HEAD_ROOT="/mnt/data/generated_heads_probe_L${LAYER}_${VARIANT}${TASK_TAG}"
HEAD_ROOT_RESID="/mnt/data/generated_heads_probe_L${LAYER}_${VARIANT}${TASK_TAG}_resid2"
HEAD_ROOT_ORTH="/mnt/data/generated_heads_probe_L${LAYER}_${VARIANT}${TASK_TAG}_orthC"
# SCORE_OUT_SUFFIX lets re-runs against new gen JSONLs (e.g. the §4½.15 9-axis
# SB-Bench rebuild) land in a distinct output dir so we never overwrite the
# legacy 2-axis results. Leave unset for the default behaviour.
SCORE_OUT="/mnt/data/phase08_offline_reward_probe_L${LAYER}${TASK_TAG}${SCORE_OUT_SUFFIX:-}"
PCS_NPY="/mnt/data/generated_heads_letter_post_letter_L${LAYER}/orthogonal_heads.npy"

ARGS=("${@:+$@}")
STAGES=()
for a in "${ARGS[@]:+${ARGS[@]}}"; do
    case "$a" in
        probe|head|residual|orth|score|pull) STAGES+=("$a") ;;
        *) echo "❌ Unknown arg: $a (valid: probe head residual orth score pull)"; exit 2 ;;
    esac
done
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(probe head score pull)
fi

echo "════ Phase 0.8 A3 Tier 0 — probe-as-head ══════════════════════════════"
echo "  variant         : ${VARIANT}"
echo "  probe task      : ${PROBE_TASK}  (suffix='${TASK_SUFFIX}')"
echo "  layer (head/score): ${LAYER}"
echo "  layers (probe)  : ${LAYERS}"
echo "  stages          : ${STAGES[*]}"
echo "  probe cache     : ${PROBE_CACHE}"
echo "  probe weights   : ${PROBE_NPZ}"
echo "  head root       : ${HEAD_ROOT}"
echo "  score output    : ${SCORE_OUT}"
echo "  pcs.npy (resid) : ${PCS_NPY}"
echo

for STAGE in "${STAGES[@]}"; do
    case "${STAGE}" in

        probe)
            echo ""
            echo "▶▶ [${VARIANT}] re-run probe with --save_weights_at_layer (layers=${LAYERS})"
            # Note: even though the probe always computes all 37 layers internally,
            # we only DUMP weights for the layer(s) of interest. cache_npz lets us
            # reuse the hidden states for a residual / multi-layer / multi-task follow-up.
            # The probe_layers.py loop writes ALL task NPZs (is_good_C / is_C /
            # bias_aligned / correct) per requested layer in one pass.
            LAYERS_COMMA=$(echo "${LAYERS}" | tr ' ' ',')
            # Optional parquet/image_root overrides — needed when VARIANT is a
            # non-VLBias re-shape (e.g. sbbench9_base) whose ids/images don't
            # live in the default VLBias parquet.
            PROBE_PARQUET_ARG=()
            PROBE_IMAGE_ROOT_ARG=()
            [[ -n "${PARQUET:-}" ]]    && PROBE_PARQUET_ARG=(--parquet "${PARQUET}")
            [[ -n "${IMAGE_ROOT:-}" ]] && PROBE_IMAGE_ROOT_ARG=(--image-root "${IMAGE_ROOT}")
            modal run src/run_modal.py::run_layer_probe \
                --variant "${VARIANT}" \
                --batch-size "${BATCH}" \
                --max-per-cell "${MAX_PER_CELL}" \
                --max-pixels "${MAX_PIXELS}" \
                --holdout-tag "${HOLDOUT_TAG}" \
                --max-holdout "${MAX_HOLDOUT}" \
                --token-position "${TOKEN_POSITION}" \
                --cache-npz "${PROBE_CACHE}" \
                --save-weights-at-layer "${LAYERS_COMMA}" \
                "${PROBE_PARQUET_ARG[@]:+${PROBE_PARQUET_ARG[@]}}" \
                "${PROBE_IMAGE_ROOT_ARG[@]:+${PROBE_IMAGE_ROOT_ARG[@]}}"
            ;;

        head)
            echo ""
            echo "▶▶ [${VARIANT} L=${LAYER}] convert probe → reward head"
            modal run src/run_modal.py::run_probe_to_head \
                --weights-npz "${PROBE_NPZ}" \
                --out-dir "${HEAD_ROOT}" \
                --normalize
            ;;

        residual)
            echo ""
            echo "▶▶ [${VARIANT} L=${LAYER}] residual probe head (PC0+PC1 projected out)"
            modal run src/run_modal.py::run_probe_to_head \
                --weights-npz "${PROBE_NPZ}" \
                --out-dir "${HEAD_ROOT_RESID}" \
                --normalize \
                --residualize-pcs "${PCS_NPY}" \
                --residualize-topk 2
            # Also score the residual variant immediately so `score` stage stays simple
            for V_SCORE in ${SCORE_VARIANTS:-${VARIANT}}; do
                echo ""
                echo "▶▶ [residual] score variant=${V_SCORE}"
                modal run src/run_modal.py::run_vlbias_offline_score \
                    --variant "${V_SCORE}" \
                    --head-type "probe" \
                    --heads-root "${HEAD_ROOT_RESID}" \
                    --output-dir "${SCORE_OUT}_resid2" \
                    --batch-size "${SCORE_BATCH}" \
                    --max-per-cell "${SCORE_MAX_PER_CELL}" \
                    --max-pixels "${MAX_PIXELS}" \
                    --token-position "${TOKEN_POSITION}" \
                    --layer-idx "${LAYER}"
            done
            ;;

        orth)
            # Phase 0.8 A3 Move 1: Gram-Schmidt the is_good_C probe orthogonal to is_C.
            # Requires the `probe` stage to have produced both npz files (re-run probe
            # with the updated probe_layers.py to get the _isC.npz companion).
            echo ""
            echo "▶▶ [${VARIANT} L=${LAYER}] orth-against-isC probe head"
            modal run src/run_modal.py::run_probe_to_head \
                --weights-npz "${PROBE_NPZ}" \
                --out-dir "${HEAD_ROOT_ORTH}" \
                --normalize \
                --orthogonalize-against "${PROBE_NPZ_ISC}"
            # Score on requested variants (default: base + 2 endpoints to match Test B)
            for V_SCORE in ${SCORE_VARIANTS:-base svm_ep1-end pca_ep1-end}; do
                echo ""
                echo "▶▶ [orth] score variant=${V_SCORE}"
                modal run src/run_modal.py::run_vlbias_offline_score \
                    --variant "${V_SCORE}" \
                    --head-type "probe" \
                    --heads-root "${HEAD_ROOT_ORTH}" \
                    --output-dir "${SCORE_OUT}_orthC" \
                    --batch-size "${SCORE_BATCH}" \
                    --max-per-cell "${SCORE_MAX_PER_CELL}" \
                    --max-pixels "${MAX_PIXELS}" \
                    --token-position "${TOKEN_POSITION}" \
                    --layer-idx "${LAYER}"
            done
            ;;

        score)
            # Optional path overrides — used by the SB-Bench transfer audit to
            # re-target the scorer at a different gen JSONL / parquet / image
            # root without touching the Modal function defaults. Leave unset
            # for the standard VLBiasBench scoring path.
            SCORE_GEN_DIR_ARG=()
            SCORE_PARQUET_ARG=()
            SCORE_IMAGE_ROOT_ARG=()
            [[ -n "${GEN_DIR:-}" ]]    && SCORE_GEN_DIR_ARG=(--gen-dir "${GEN_DIR}")
            [[ -n "${PARQUET:-}" ]]    && SCORE_PARQUET_ARG=(--parquet "${PARQUET}")
            [[ -n "${IMAGE_ROOT:-}" ]] && SCORE_IMAGE_ROOT_ARG=(--image-root "${IMAGE_ROOT}")
            for V_SCORE in ${SCORE_VARIANTS:-${VARIANT}}; do
                echo ""
                echo "▶▶ [L=${LAYER}] offline reward score (probe head) variant=${V_SCORE}"
                modal run src/run_modal.py::run_vlbias_offline_score \
                    --variant "${V_SCORE}" \
                    --head-type "probe" \
                    --heads-root "${HEAD_ROOT}" \
                    --output-dir "${SCORE_OUT}" \
                    --batch-size "${SCORE_BATCH}" \
                    --max-per-cell "${SCORE_MAX_PER_CELL}" \
                    --max-pixels "${MAX_PIXELS}" \
                    --token-position "${TOKEN_POSITION}" \
                    --layer-idx "${LAYER}" \
                    "${SCORE_GEN_DIR_ARG[@]:+${SCORE_GEN_DIR_ARG[@]}}" \
                    "${SCORE_PARQUET_ARG[@]:+${SCORE_PARQUET_ARG[@]}}" \
                    "${SCORE_IMAGE_ROOT_ARG[@]:+${SCORE_IMAGE_ROOT_ARG[@]}}"
            done
            ;;

        pull)
            mkdir -p "${LOCAL_DIR}"
            echo ""
            # Pull probe NPZs for ALL tasks and ALL requested layers — the
            # probe stage writes them in one pass and downstream analysis
            # always wants the full set on disk.
            for L_PULL in ${LAYERS}; do
                for SUF in "" "_isC" "_biasA" "_corr"; do
                    modal volume get "${VOLUME_NAME}" \
                        "phase08_probe_results/${VARIANT}_L${L_PULL}_probe_weights${SUF}.npz" \
                        "${LOCAL_DIR}" --force 2>/dev/null || true
                done
            done
            echo "▶▶ pull head dir (task=${PROBE_TASK})"
            modal volume get "${VOLUME_NAME}" \
                "${HEAD_ROOT#/mnt/data/}" "${LOCAL_DIR}" --force || true
            echo "▶▶ pull offline-reward JSONs (probe, task=${PROBE_TASK})"
            modal volume get "${VOLUME_NAME}" \
                "${SCORE_OUT#/mnt/data/}" "${LOCAL_DIR}" --force || true
            # Best-effort: residual outputs if they exist (task-aware)
            modal volume get "${VOLUME_NAME}" \
                "${HEAD_ROOT_RESID#/mnt/data/}" "${LOCAL_DIR}" --force 2>/dev/null || true
            modal volume get "${VOLUME_NAME}" \
                "${SCORE_OUT#/mnt/data/}_resid2" "${LOCAL_DIR}" --force 2>/dev/null || true
            # Best-effort: orth outputs if they exist (task-aware)
            modal volume get "${VOLUME_NAME}" \
                "${HEAD_ROOT_ORTH#/mnt/data/}" "${LOCAL_DIR}" --force 2>/dev/null || true
            modal volume get "${VOLUME_NAME}" \
                "${SCORE_OUT#/mnt/data/}_orthC" "${LOCAL_DIR}" --force 2>/dev/null || true
            ;;
    esac
done

echo ""
echo "✅ phase08_a3_probe_as_head.sh complete"
echo "   variant=${VARIANT}  layer=${LAYER}  stages=${STAGES[*]}"
[[ " ${STAGES[*]} " == *" pull "* ]] && ls -la "${LOCAL_DIR}" 2>/dev/null || true
