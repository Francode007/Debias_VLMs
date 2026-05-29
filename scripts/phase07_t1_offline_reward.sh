#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0.7 — T1.1: Offline reward evaluation on VLBiasBench.
#
# Scores existing generation JSONLs (produced by G4a) with the SB-Bench reward
# heads (SVM and PCA). No PPO, no new training. Decisive experiment for the
# remediation plan (see Phase0.7_Remediation_Plan.md §3 T1.1).
#
# Inputs on volume (must already exist — produced by phase07_g4_vlbiasbench.sh):
#   /mnt/data/phase07_vlbiasbench/<variant>_vlbias_gen.jsonl
#   /mnt/data/generated_heads_letter_post_letter/sb_bench-{SVM,PCA}-component/
#
# Outputs:
#   /mnt/data/phase07_offline_reward/<variant>__<head_type>__offline_reward.json
#
# Usage:
#   bash scripts/phase07_t1_offline_reward.sh                 # all variants × both heads + pull
#   bash scripts/phase07_t1_offline_reward.sh base svm        # only score base with SVM
#   bash scripts/phase07_t1_offline_reward.sh score pull      # all variants + pull
#
# Env knobs:
#   VARIANTS     — space-separated list of variant tags (default: base svm_ep1-50pct svm_ep1-end pca_ep1-step80 pca_ep1-end)
#   HEADS        — space-separated head types (default: svm pca — both scored in one forward pass per variant)
#   BATCH        — embedding batch size (default: 16)
#   MAX_PER_CELL — stratified samples per (bbq_axis x condition) cell (default: 20 → 600 records per variant)
#   MAX_SAMPLES  — ignored when MAX_PER_CELL > 0; otherwise first-N cap (default: 0 = all)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

VARIANTS="${VARIANTS:-base svm_ep1-50pct svm_ep1-end pca_ep1-step80 pca_ep1-end}"
HEADS="${HEADS:-svm pca}"
BATCH="${BATCH:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_PER_CELL="${MAX_PER_CELL:-20}"   # 10 axes x 3 conditions x 20 = 600 records / variant
MAX_PIXELS="${MAX_PIXELS:-0}"        # 0 = uncapped (matches generator); set e.g. 200704 to throttle
TOKEN_POSITION="${TOKEN_POSITION:-post_letter}"

OUTPUT_DIR="/mnt/data/phase07_offline_reward"
VOLUME_NAME="debias-vlm-persistent-storage"
LOCAL_DIR="Phase0.7/offline_reward"

# Parse positional stages — accept either head-type tokens or 'score' / 'pull'.
ARGS=("${@:+$@}")
STAGES=()
EXPLICIT_HEADS=()
EXPLICIT_VARIANTS=()
for a in "${ARGS[@]:+${ARGS[@]}}"; do
    case "$a" in
        score|pull) STAGES+=("$a") ;;
        svm|pca)    EXPLICIT_HEADS+=("$a") ;;
        *)          EXPLICIT_VARIANTS+=("$a") ;;
    esac
done
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(score pull)
fi
if [[ ${#EXPLICIT_HEADS[@]} -gt 0 ]]; then
    HEADS="${EXPLICIT_HEADS[*]}"
fi
if [[ ${#EXPLICIT_VARIANTS[@]} -gt 0 ]]; then
    VARIANTS="${EXPLICIT_VARIANTS[*]}"
fi

_score () {
    local variant="$1"
    local heads="$2"
    echo ""
    echo "▶▶ score  variant=${variant}  heads=${heads}"
    modal run src/run_modal.py::run_vlbias_offline_score \
        --variant "${variant}" \
        --head-type "${heads}" \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size "${BATCH}" \
        --max-samples "${MAX_SAMPLES}" \
        --max-per-cell "${MAX_PER_CELL}" \
        --max-pixels "${MAX_PIXELS}" \
        --token-position "${TOKEN_POSITION}"
}

for STAGE in "${STAGES[@]}"; do
    case "${STAGE}" in

        score)
            echo "════ stage: score ═════════════════════════════════════════════"
            echo "  variants : ${VARIANTS}"
            echo "  heads    : ${HEADS}"
            # Join heads with comma for multi-head single-pass scoring
            HEADS_CSV=$(echo "${HEADS}" | tr ' ' ',')
            for V in ${VARIANTS}; do
                _score "${V}" "${HEADS_CSV}"
            done
            ;;

        pull)
            echo "════ stage: pull ══════════════════════════════════════════════"
            mkdir -p "${LOCAL_DIR}"
            modal volume get "${VOLUME_NAME}" \
                "${OUTPUT_DIR#/mnt/data/}" "${LOCAL_DIR}" --force
            echo "✅ Results pulled to ${LOCAL_DIR}/"
            ls -la "${LOCAL_DIR}" || true
            ;;

        *)
            echo "❌ Unknown stage: ${STAGE}"
            echo "   Valid: score pull  (and optional head/variant tokens)"
            exit 2
            ;;
    esac
done

echo ""
echo "✅ phase07_t1_offline_reward.sh complete (stages: ${STAGES[*]})"
