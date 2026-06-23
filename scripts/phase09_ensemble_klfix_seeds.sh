#!/usr/bin/env bash
# Phase 0.9 R3 step 4a — 4-seed PPO sweep with multi-layer ensemble reward.
#
# Goal: produce the 4-seed canonical statistics needed for the §3bis pre-
# committed gate vs the KLFIX +1.08pp single-layer baseline:
#   - Mean canonical-acc Δ vs vanilla    ≥ +1.5 pp
#   - Cross-seed std                     ≤ 0.5 pp
#   - Per-axis: beats KLFIX on            ≥ 6/9 BBQ axes
#   - KL stability: tail-KL std           ≤ 0.005
#   - Ambig channel preserved: VLBias ambig rate within −2 pp of vanilla
#
# Recipe (bit-identical to phase09_ensemble_klfix_smoke.sh):
#   - Same 3-fix KLFIX backbone (value-head zero-init code, --value-clip-range
#     0.2, --kl-adapt-rate 0.3)
#   - Ensemble bundle at /generated_heads_probe_L17-33_zmean_base_biasA/
#     {L17, L21, L25, L29, L33} layers, pool=zmean
#
# Each job: ~3h on A100-80GB, all 4 detached. Modal scheduler runs them
# concurrently subject to GPU quota.
#
# Output dirs:
#   /mnt/data/output_ppo_phase09_2k_ensemble_s{1,2,3,4}_klfix/
#
# Usage:
#   bash scripts/phase09_ensemble_klfix_seeds.sh
#   SEEDS="2 4"   bash scripts/phase09_ensemble_klfix_seeds.sh   # subset
#
# After completion (each job logs say 'PPO training complete.'):
#   bash scripts/phase09_ensemble_canonical_gens.sh   (R3 step 4b)

set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source debias_env/bin/activate
fi

SEEDS="${SEEDS:-1 2 3 4}"
BUNDLE_DIR="${BUNDLE_DIR:-/mnt/data/generated_heads_probe_L17-33_zmean_base_biasA}"
ENSEMBLE_POOL="${ENSEMBLE_POOL:-zmean}"
ENSEMBLE_LAYERS="${ENSEMBLE_LAYERS:-17,21,25,29,33}"

# Same recipe as phase09_ensemble_klfix_smoke.sh — only SEEDS / OUT_DIR differ.
COMMON_FLAGS=(
    --epochs 1
    --dataset sb_bench --model-family qwen
    --kl-beta 0.1 --target-kl 0.02
    --kl-adapt-rate 0.3                           # KLFIX fix #3
    --learning-rate 5e-6 --lora-r 16 --lora-alpha 32
    --batch-size 8 --max-gen-tokens 8
    --max-train-samples 2000
    --reward-mode bias_aligned
    --bias-aligned-coef 1.0 --correctness-coef 1.0 --ambig-preservation-coef 0.5
    --use-frozen-phi
    --midtrain-eval-every-steps 50 --midtrain-eval-samples 64
    --reward-head-layer 17
    --reward-heads-dir-override "/mnt/data/generated_heads_probe_L13_base_biasA/sb_bench-PROBE-component"
    --value-clip-range 0.2                        # KLFIX fix #2
    --ensemble-bundle-dir "${BUNDLE_DIR}"
    --ensemble-layers "${ENSEMBLE_LAYERS}"
    --ensemble-pool "${ENSEMBLE_POOL}"
)

echo "▶ Phase 0.9 R3 step 4a — 4-seed ensemble PPO sweep"
echo "  SEEDS='${SEEDS}'"
echo "  Ensemble bundle: ${BUNDLE_DIR}"
echo "  Layers: ${ENSEMBLE_LAYERS}   pool: ${ENSEMBLE_POOL}"
echo ""

for SEED in ${SEEDS}; do
    OUT_DIR="/mnt/data/output_ppo_phase09_2k_ensemble_s${SEED}_klfix"
    LOG="/tmp/ens_klfix_s${SEED}.log"
    echo "▶ Launching ensemble klfix seed=${SEED}  →  ${OUT_DIR}"
    nohup modal run --detach src/run_modal.py::run_training \
        "${COMMON_FLAGS[@]}" \
        --output-dir "${OUT_DIR}" \
        --seed "${SEED}" \
        > "${LOG}" 2>&1 &
    echo "  pid=$!  log=${LOG}"
    sleep 2
done

echo ""
echo "✅ Submitted $(echo ${SEEDS} | wc -w | tr -d ' ') detached jobs (SEEDS='${SEEDS}')."
echo ""
echo "These will keep running on Modal after this shell closes (--detach)."
echo "It is safe to disconnect immediately once each launcher prints its app ID."
echo ""
echo "Track via:"
echo "    modal app list | head -20"
echo "    tail -f /tmp/ens_klfix_s*.log    (local CLI streams while attached)"
echo "    modal app logs <app-id> | tail -50   (fetch from cloud after disconnect)"
echo ""
echo "Once jobs complete (modal app list shows them as stopped), proceed to:"
echo "    bash scripts/phase09_ensemble_canonical_gens.sh   # R3 step 4b"
