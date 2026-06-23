#!/usr/bin/env bash
# Phase 0.9 R3 step 4c — VLBias cross-dataset transfer gens for ensemble seeds.
#
# Mirrors phase09_ensemble_canonical_gens.sh but uses run_vlbiasbench_eval
# (no SB-Bench parquet; VLBias has its own data pipeline). Outputs:
#   /mnt/data/phase07_vlbiasbench/phase09_ensemble_s{1,2,4}_klfix_vlbias_gen.jsonl
#   /mnt/data/phase07_vlbiasbench/phase09_ensemble_s{1,2,4}_klfix_vlbias_results.json
#
# Each gen is a detached Modal job (~20-30 min on A100-80GB). Three in
# parallel via Modal's scheduler.
#
# Baseline (KLFIX-PPO + L13 reward) VLBias gens already exist on volume at:
#   /mnt/data/phase07_vlbiasbench/phase08_2k_klfix_s{1..4}_vlbias_gen.jsonl
# Vanilla VLBias gen:
#   /mnt/data/phase07_vlbiasbench/base_vlbias_gen.jsonl
#
# Usage:
#   bash scripts/phase09_ensemble_vlbias_gens.sh
#   SEEDS="1 4"  bash scripts/phase09_ensemble_vlbias_gens.sh
#
# After completion (each job logs say 'VLBiasBench eval [<tag>] complete'):
#   for S in 1 2 4; do
#     modal volume get debias-vlm-persistent-storage \
#         /phase07_vlbiasbench/phase09_ensemble_s${S}_klfix_vlbias_gen.jsonl \
#         Phase0.9/vlbias/ --force
#     modal volume get debias-vlm-persistent-storage \
#         /phase07_vlbiasbench/phase09_ensemble_s${S}_klfix_vlbias_results.json \
#         Phase0.9/vlbias/ --force
#   done
#   # Pull baselines (one-off):
#   modal volume get debias-vlm-persistent-storage \
#       /phase07_vlbiasbench/base_vlbias_gen.jsonl Phase0.9/vlbias/ --force
#   modal volume get debias-vlm-persistent-storage \
#       /phase07_vlbiasbench/base_vlbias_results.json Phase0.9/vlbias/ --force
#   for S in 1 2 3 4; do
#     modal volume get debias-vlm-persistent-storage \
#         /phase07_vlbiasbench/phase08_2k_klfix_s${S}_vlbias_results.json \
#         Phase0.9/vlbias/ --force
#   done

set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source debias_env/bin/activate
fi

SEEDS="${SEEDS:-1 2 4}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"

fire() {
  local tag="$1"; local ckpt="$2"
  local log="/tmp/phase09_ensemble_vlbias_${tag}.log"
  echo "▶ [$tag] firing → /mnt/data/phase07_vlbiasbench/${tag}_vlbias_{gen,results}  (log: $log)"
  nohup modal run --detach src/run_modal.py::run_vlbiasbench_eval \
    --checkpoint-dir "$ckpt" \
    --tag "$tag" \
    --num-samples "${NUM_SAMPLES}" \
    >"$log" 2>&1 &
  sleep 2
}

for S in ${SEEDS}; do
  fire "phase09_ensemble_s${S}_klfix" \
       "/mnt/data/output_ppo_phase09_2k_ensemble_s${S}_klfix/final_debiased_model"
done

echo ""
echo "✅ Fired $(echo ${SEEDS} | wc -w | tr -d ' ') jobs (SEEDS='${SEEDS}', num_samples=${NUM_SAMPLES})."
echo ""
echo "Track via:"
echo "    modal app list | head -15"
echo "    tail -f /tmp/phase09_ensemble_vlbias_*.log"
echo ""
echo "Disconnect-safe: --detach keeps jobs running after this shell closes."
