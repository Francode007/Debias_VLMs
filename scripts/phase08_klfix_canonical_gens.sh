#!/usr/bin/env bash
# Phase 0.8 — Canonical SB-Bench gens for the klfix 4-seed sweep.
#
# Mirrors scripts/phase08_vwarmup_canonical_gens.sh but targets the klfix
# checkpoints (zero-init value head + --value-clip-range 0.2 + --kl-adapt-rate 0.3).
#
#   • CANONICAL n=2916 split (sb_bench_data.parquet + split_indices.json)
#   • 4 klfix checkpoints: output_ppo_phase08_2k_L13_s{1..4}_klfix
#
# Each gen is a detached Modal job (~30 min on A100-80GB). Local launchers are
# backgrounded (&) so they do not block on stdout tailing.
#
# Usage:
#   bash scripts/phase08_klfix_canonical_gens.sh
#
# After they finish:
#   for S in 1 2 3 4; do
#     modal volume get debias-vlm-persistent-storage \
#         /phase08_seeds/canonical_L13_s${S}_klfix_sbbench_gen.jsonl \
#         Phase0.8/canonical/ --force
#   done
#   python scripts/phase08_seed_aggregate.py \
#       --vanilla Phase0.8/canonical/canonical_base_sbbench_gen.jsonl \
#       --variant L13_canonical_klfix_4seeds \
#       --seeds Phase0.8/canonical/canonical_L13_s{1,2,3,4}_klfix_sbbench_gen.jsonl \
#       --output-json Phase0.8/canonical/L13_canonical_klfix_4seeds_aggregate.json

set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source debias_env/bin/activate
fi

SEEDS="${SEEDS:-1 2 3 4}"
PARQUET="${PARQUET:-/mnt/data/sb_bench_data/sb_bench_data.parquet}"
SPLIT="${SPLIT:-/mnt/data/split_indices.json}"

fire() {
  local tag="$1"; local ckpt="$2"; local out="$3"
  local log="/tmp/phase08_klfix_gen_${tag}.log"
  echo "▶ [$tag] firing → $out  (log: $log)"
  nohup modal run --detach src/run_modal.py::run_generation \
    --dataset sb_bench \
    --data-path "$PARQUET" \
    --output-jsonl "$out" \
    --split test \
    --split-indices-path "$SPLIT" \
    --checkpoint-dir "$ckpt" >"$log" 2>&1 &
  sleep 2
}

for S in ${SEEDS}; do
  fire "L13_s${S}_klfix" \
       "/mnt/data/output_ppo_phase08_2k_L13_s${S}_klfix/final_debiased_model" \
       "/mnt/data/phase08_seeds/canonical_L13_s${S}_klfix_sbbench_gen.jsonl"
done

echo ""
echo "✅ Fired $(echo ${SEEDS} | wc -w | tr -d ' ') jobs. Wait ~60s, then:"
echo "    modal app list | head -15"
echo "    pkill -f 'modal run --detach src/run_modal.py::run_generation'"
