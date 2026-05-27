#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0.7 — Train + evaluate a PCA-DRM PPO run (mirror of Phase 0.6 F SVM).
#
# WHY: Phase 0.6 F gave us a verified SVM-DRM LoRA. To secure both reward
# variants before launching G4 (BBQ transfer), we need an analogous PCA-DRM
# LoRA + the same G1/G2/G3 analyses on it.
#
# WHAT THIS SCRIPT DOES (single bash command):
#   train  — PPO with --head-type pca, --kept-heads-filter kept_heads_pca.json
#            (90/200 kept heads from the 100-component PCA basis), all other
#            hyperparameters bit-identical to Phase 0.6 F. Writes LoRA
#            adapters every 10 steps to /mnt/data/output_ppo_phase07_pca_drm/.
#   sweep  — run_phase0_eval_sweep across every checkpoint → SB-Bench test
#            accuracy matrix.
#   pope   — POPE eval for base + ep1-step70 + ep1-end (G3 capability tax).
#   perturb— P2 letter-order perturbation sweep on the same two checkpoints
#            (G2 content-routing check).
#   pull   — copies the result JSONs from the Modal volume to ./phase07_pca/.
#
# Stages can be selected positionally. Default runs all in order:
#   bash scripts/phase07_pca_drm_train_eval.sh
#
# Subsets:
#   bash scripts/phase07_pca_drm_train_eval.sh train sweep
#   bash scripts/phase07_pca_drm_train_eval.sh pope perturb pull
#
# Estimated total wall time: ~3-4 h on A100-80GB (train ~2h, sweep ~45 min,
# pope ~30 min, perturb ~25 min).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Configuration ────────────────────────────────────────────────────────────
RUN_NAME="phase07_pca_drm"
OUTDIR="/mnt/data/output_ppo_${RUN_NAME}"
HEADS_DIR="/mnt/data/generated_heads_letter_post_letter"
KEPT_PCA="${HEADS_DIR}/kept_heads_pca.json"
SB_PARQUET="/mnt/data/sb_bench_data/sb_bench_data.parquet"
POPE_PARQUET="/mnt/data/pope_data/pope_data.parquet"
POPE_JSONL="/mnt/data/pope_data/pope_data.jsonl"
LOCAL_PULL_DIR="./phase07_pca"
VOLUME="debias-vlm-persistent-storage"

# Checkpoints we'll evaluate end-to-end (must match step boundary written by
# --ckpt-every-steps 10; ep1-end always exists at end of epoch 1).
ANALYZE_TAGS=("ep1-step70" "ep1-end")

# ─────────────────────────────────────────────────────────────────────────────
# Stage: TRAIN — PPO with PCA-DRM heads (Phase 0.6 F hyperparams)
# ─────────────────────────────────────────────────────────────────────────────
do_train() {
  echo ""
  echo "================================================================"
  echo "  STAGE: train (PCA-DRM)  →  ${OUTDIR}"
  echo "================================================================"
  modal run src/run_modal.py::run_training \
    --epochs 1 \
    --max-train-samples 2000 \
    --batch-size 8 \
    --max-gen-tokens 8 \
    --reward-mode svm \
    --head-type pca \
    --use-frozen-phi \
    --heads-suffix _letter_post_letter \
    --kept-heads-filter "${KEPT_PCA}" \
    --lambda-causal 0.0 \
    --learning-rate 1e-4 \
    --value-learning-rate 5e-4 \
    --kl-beta 0.1 \
    --target-kl 0.02 \
    --kl-adapt-rate 0.1 \
    --kl-beta-min 0.05 \
    --kl-beta-max 5.0 \
    --max-grad-norm 0.1 \
    --value-clip-range 0.2 \
    --lr-schedule cosine \
    --min-lr-ratio 0.2 \
    --warmup-ratio 0.05 \
    --ckpt-every-steps 10 \
    --midtrain-eval-every-steps 10 \
    --midtrain-eval-samples 64 \
    --output-dir "${OUTDIR}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage: SWEEP — SB-Bench eval across every checkpoint (G1 baseline matrix)
# ─────────────────────────────────────────────────────────────────────────────
do_sweep() {
  echo ""
  echo "================================================================"
  echo "  STAGE: sweep (SB-Bench eval over all checkpoints)"
  echo "================================================================"
  modal run src/run_modal.py::run_phase0_eval_sweep \
    --train-output-dir "${OUTDIR}" \
    --combined-json "${OUTDIR}/phase07_pca_eval_sweep.json" \
    --data-path "${SB_PARQUET}" \
    --dataset sb_bench
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage: POPE — capability tax check (G3): base + ep1-step70 + ep1-end
# ─────────────────────────────────────────────────────────────────────────────
do_pope() {
  echo ""
  echo "================================================================"
  echo "  STAGE: pope (G3 capability tax — base + 2 ckpts)"
  echo "================================================================"
  # Base model (no adapter). Tag distinguishes it from ckpt runs.
  modal run src/run_modal.py::run_pope_eval \
    --tag "base" \
    --output-dir "${OUTDIR}/pope" \
    --data-path "${POPE_PARQUET}" \
    --gt-file "${POPE_JSONL}"

  for tag in "${ANALYZE_TAGS[@]}"; do
    modal run src/run_modal.py::run_pope_eval \
      --checkpoint-dir "${OUTDIR}/checkpoint-${tag}" \
      --tag "${tag}" \
      --output-dir "${OUTDIR}/pope" \
      --data-path "${POPE_PARQUET}" \
      --gt-file "${POPE_JSONL}"
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage: PERTURB — P2 cyclic-1 letter perturbation on the same two checkpoints
# ─────────────────────────────────────────────────────────────────────────────
do_perturb() {
  echo ""
  echo "================================================================"
  echo "  STAGE: perturb (G2 letter-order content-routing check)"
  echo "================================================================"
  local tags
  tags="$(IFS=, ; echo "${ANALYZE_TAGS[*]}")"
  modal run src/run_modal.py::run_phase0_eval_sweep \
    --train-output-dir "${OUTDIR}" \
    --combined-json "${OUTDIR}/phase07_P2_perturbed_cyclic1.json" \
    --gen-subdir generations_perturbed_cyclic1 \
    --only-tags "${tags}" \
    --shuffle-answers cyclic_1
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage: PULL — copy result JSONs to the local workspace for analysis
# ─────────────────────────────────────────────────────────────────────────────
do_pull() {
  echo ""
  echo "================================================================"
  echo "  STAGE: pull  →  ${LOCAL_PULL_DIR}/"
  echo "================================================================"
  mkdir -p "${LOCAL_PULL_DIR}"

  # Best-effort: skip files that don't exist yet (e.g. user ran subset only).
  pull_one() {
    local src="$1"; local dst="$2"
    if modal volume get "${VOLUME}" "${src}" "${dst}" --force >/dev/null 2>&1; then
      echo "  ✓ ${dst}"
    else
      echo "  ⚠ skipped (not on volume): ${src}"
    fi
  }

  pull_one "output_ppo_${RUN_NAME}/phase07_pca_eval_sweep.json"      "${LOCAL_PULL_DIR}/sb_bench_sweep.json"
  pull_one "output_ppo_${RUN_NAME}/phase07_P2_perturbed_cyclic1.json" "${LOCAL_PULL_DIR}/p2_perturbed_cyclic1.json"
  pull_one "output_ppo_${RUN_NAME}/metrics.jsonl"                    "${LOCAL_PULL_DIR}/metrics.jsonl"
  pull_one "output_ppo_${RUN_NAME}/pope/base_pope_results.json"      "${LOCAL_PULL_DIR}/pope_base.json"
  for tag in "${ANALYZE_TAGS[@]}"; do
    pull_one "output_ppo_${RUN_NAME}/pope/${tag}_pope_results.json"  "${LOCAL_PULL_DIR}/pope_${tag}.json"
    pull_one "output_ppo_${RUN_NAME}/generations/${tag}.jsonl"        "${LOCAL_PULL_DIR}/${tag}.jsonl"
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
  stages=(train sweep pope perturb pull)
else
  stages=("$@")
fi

for stage in "${stages[@]}"; do
  case "${stage}" in
    train)   do_train ;;
    sweep)   do_sweep ;;
    pope)    do_pope ;;
    perturb) do_perturb ;;
    pull)    do_pull ;;
    *) echo "Unknown stage: ${stage} (choose from: train sweep pope perturb pull)"; exit 1 ;;
  esac
done

echo ""
echo "================================================================"
echo "  DONE — stages: ${stages[*]}"
echo ""
echo "  After 'pull', local files in ${LOCAL_PULL_DIR}/:"
echo "    sb_bench_sweep.json    G1 — per-checkpoint × category accuracy"
echo "    p2_perturbed_cyclic1.json   G2 — letter-rotated eval"
echo "    pope_{base,ep1-step70,ep1-end}.json   G3 — capability tax"
echo "    {ep1-step70,ep1-end}.jsonl  raw generations (for P1 polarity split)"
echo "    metrics.jsonl          per-step training telemetry"
echo ""
echo "  P1 polarity split (purely local):"
echo "    python scripts/phase07_p1_polarity.py ${LOCAL_PULL_DIR}/ep1-end.jsonl"
echo "================================================================"
