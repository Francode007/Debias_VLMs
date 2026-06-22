#!/usr/bin/env bash
# Phase 0.9 R3 step 3 — Smoke test for multi-layer ensemble reward.
#
# Goal: confirm the new ensemble reward path (drm_loader.load_ensemble_probe_bundle,
# Phase08PPOController is_ensemble_mode branch, args.ensemble_*) trains end-to-end
# on Modal without crashing, produces sensible per-layer z metrics in
# metrics.jsonl, and stays within KLFIX KL-stability budget.
#
# Pre-committed window (per Phase0.8_Strategic_Plan.md §3bis + R3 user choice):
#   layers = {17, 21, 25, 29, 33}, pool = "zmean"
#
# Pre-requisites (already done in R3 step 1):
#   - Bundle pushed to volume at:
#       /generated_heads_probe_L17-33_zmean_base_biasA/
#         {L17,L21,L25,L29,L33}.pth + ensemble_metadata.json
#   - Same KLFIX backbone as scripts/phase08_klfix_s1_smoke.sh:
#       value-head zero-init (code), --value-clip-range 0.2, --kl-adapt-rate 0.3
#
# Pass criteria for smoke (s1):
#   1. Job completes without crash (no shape errors, no None hidden states).
#   2. metrics.jsonl contains keys: ensemble_mode, ensemble_pool, ensemble_layers,
#      ensemble_z_L17_mean, ensemble_z_L21_mean, ensemble_z_L25_mean,
#      ensemble_z_L29_mean, ensemble_z_L33_mean — all non-NaN.
#   3. tail-KL (mean over last 20 steps) ≤ 0.005 (KLFIX budget).
#   4. midtrain_eval_acc shows non-degenerate sampling (parse rate ≥ 0.95,
#      pred distribution not collapsed to a single letter).
#
# If smoke passes, scripts/phase09_ensemble_klfix_seeds.sh launches the
# 4-seed sweep for the pre-committed gate (Phase0.8_Strategic_Plan.md §3bis):
#   - Δacc vs vanilla ≥ +1.5 pp
#   - Cross-seed std ≤ 0.5 pp
#   - Beats KLFIX on ≥ 6/9 BBQ axes
#   - tail-KL std ≤ 0.005
#   - VLBias ambig rate within −2 pp of vanilla
#
# Usage:
#   bash scripts/phase09_ensemble_klfix_smoke.sh
#   SEEDS="1 2" bash scripts/phase09_ensemble_klfix_smoke.sh    # 2-seed parallel smoke

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source debias_env/bin/activate
fi

SEEDS="${SEEDS:-1}"
WARMUP="${WARMUP:-30}"
WARMUP_LR_MULT="${WARMUP_LR_MULT:-2.0}"
BUNDLE_DIR="${BUNDLE_DIR:-/mnt/data/generated_heads_probe_L17-33_zmean_base_biasA}"
ENSEMBLE_POOL="${ENSEMBLE_POOL:-zmean}"
ENSEMBLE_LAYERS="${ENSEMBLE_LAYERS:-17,21,25,29,33}"

# Same KLFIX baseline flags as phase08_klfix_s1_smoke.sh (the s1 hardened
# recipe that produced canonical +1.08pp baseline) — only the reward signal
# changes via the 3 ensemble flags at the bottom.
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
    # Single-layer head-dir is still required (loader's existence check);
    # it's ignored at reward-compute time when ensemble mode is active, but
    # the legacy load_pca_components path in train_rl.py needs a valid dir.
    --reward-head-layer 17
    --reward-heads-dir-override "/mnt/data/generated_heads_probe_L13_base_biasA/sb_bench-PROBE-component"
    --value-warmup-steps "${WARMUP}"
    --value-warmup-lr-multiplier "${WARMUP_LR_MULT}"
    --value-clip-range 0.2                        # KLFIX fix #2
    # ─── Phase 0.9 R3 ensemble flags ──────────────────────────────────
    --ensemble-bundle-dir "${BUNDLE_DIR}"
    --ensemble-layers "${ENSEMBLE_LAYERS}"
    --ensemble-pool "${ENSEMBLE_POOL}"
)

echo "▶ Phase 0.9 R3 ensemble smoke"
echo "  SEEDS='${SEEDS}'  WARMUP=${WARMUP}  WARMUP_LR_MULT=${WARMUP_LR_MULT}"
echo "  Ensemble bundle: ${BUNDLE_DIR}"
echo "  Layers: ${ENSEMBLE_LAYERS}   pool: ${ENSEMBLE_POOL}"
echo ""

for SEED in ${SEEDS}; do
    OUT_DIR="/mnt/data/output_ppo_phase09_2k_ensemble_s${SEED}_klfix_smoke"
    LOG="/tmp/ens_klfix_s${SEED}_smoke.log"
    echo "▶ Launching ensemble klfix-smoke seed=${SEED}  →  ${OUT_DIR}"
    nohup modal run --detach src/run_modal.py::run_training \
        "${COMMON_FLAGS[@]}" \
        --output-dir "${OUT_DIR}" \
        --seed "${SEED}" \
        > "${LOG}" 2>&1 &
    echo "  pid=$!  log=${LOG}"
    sleep 2
done
wait
echo ""
echo "✅ Detached jobs submitted (SEEDS='${SEEDS}')."
echo "Track via:"
echo "    modal app list | head -20"
echo "    tail -f /tmp/ens_klfix_s*_smoke.log"
echo ""
echo "After completion (job logs say 'PPO training complete.'):"
echo "  for S in ${SEEDS}; do"
echo "    modal volume get debias-vlm-persistent-storage \\"
echo "        /output_ppo_phase09_2k_ensemble_s\${S}_klfix_smoke/metrics.jsonl \\"
echo "        Phase0.9/smoke_metrics/s\${S}_metrics.jsonl --force"
echo "  done"
echo "  # Verify ensemble_z_L* keys present + tail-KL ≤ 0.005:"
echo "  python -c \"import json; rows=[json.loads(l) for l in open('Phase0.9/smoke_metrics/s1_metrics.jsonl')]; "
echo "    print('keys:', [k for k in rows[-1].keys() if 'ensemble' in k.lower()]);"
echo "    kls=[r.get('mean_kl',0) for r in rows[-20:]]; "
echo "    import statistics; print(f'tail_kl_mean={statistics.mean(kls):.4f}')\""
