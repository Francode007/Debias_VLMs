#!/usr/bin/env bash
# Phase 0.8 — Fire ALL remaining detached Modal gens in parallel.
# Each `modal run --detach` is backgrounded (&) so the local CLI does NOT
# block on stdout tailing. The remote apps continue running even after we
# kill all the local backgrounded CLIs.
#
# By default this fires:
#   - 3 CF jobs (phase08_2k, phase08_2k_corrOnly, phase08_2k_biasOnly)
#     [vanilla `base` is already running as ap-W9QzJJJSXkyJ4zbmVJj5g9]
#   - 6 seed jobs (L13_s{1,2,4}, L17_s{1,2,4})
#
# Total: 9 new detached apps. Each logs go to /tmp/phase08_launch_<tag>.log.
# After firing, wait ~30 sec then run `modal app list` to confirm 9 new
# ephemeral (detached) apps appear, then kill -9 the local backgrounded
# launchers (we don't need them).
#
# Usage:
#   bash scripts/phase08_launch_gens_parallel.sh
#   bash scripts/phase08_launch_gens_parallel.sh include-vanilla   # also fire base

set -uo pipefail

cd "$(dirname "$0")/.."

INCLUDE_VANILLA="${1:-no}"
PARQUET="/mnt/data/sb_bench_data/sb_bench_data_9axis.parquet"
SPLIT="/mnt/data/split_indices_9axis.json"

fire() {
  local tag="$1"; local ckpt="$2"; local out="$3"
  local log="/tmp/phase08_launch_${tag}.log"
  echo "▶ [$tag] firing → $out (log: $log)"
  if [[ -n "$ckpt" ]]; then
    nohup modal run --detach src/run_modal.py::run_generation \
      --dataset sb_bench \
      --data-path "$PARQUET" \
      --output-jsonl "$out" \
      --split test \
      --split-indices-path "$SPLIT" \
      --checkpoint-dir "$ckpt" >"$log" 2>&1 &
  else
    nohup modal run --detach src/run_modal.py::run_generation \
      --dataset sb_bench \
      --data-path "$PARQUET" \
      --output-jsonl "$out" \
      --split test \
      --split-indices-path "$SPLIT" \
      --checkpoint-dir "" >"$log" 2>&1 &
  fi
  # tiny stagger so Modal client isn't slammed with 9 simultaneous handshakes
  sleep 2
}

if [[ "$INCLUDE_VANILLA" == "include-vanilla" ]]; then
  fire base "" /mnt/data/phase08_cf/9axis_base_sbbench_gen.jsonl
fi

# 3 CF variants
fire phase08_2k          /mnt/data/output_ppo_phase08_2k/final_debiased_model \
     /mnt/data/phase08_cf/9axis_phase08_2k_sbbench_gen.jsonl
fire phase08_2k_corrOnly /mnt/data/output_ppo_phase08_2k_corrOnly/final_debiased_model \
     /mnt/data/phase08_cf/9axis_phase08_2k_corrOnly_sbbench_gen.jsonl
fire phase08_2k_biasOnly /mnt/data/output_ppo_phase08_2k_biasOnly/final_debiased_model \
     /mnt/data/phase08_cf/9axis_phase08_2k_biasOnly_sbbench_gen.jsonl

# 6 seed models
for L in 13 17; do
  for S in 1 2 4; do
    fire "L${L}_s${S}" \
         "/mnt/data/output_ppo_phase08_2k_L${L}_s${S}/final_debiased_model" \
         "/mnt/data/phase08_seeds/9axis_L${L}_s${S}_sbbench_gen.jsonl"
  done
done

echo ""
echo "✅ Fired. Wait ~60s then run:"
echo "    modal app list | head -15"
echo "    # then optionally:  pkill -f 'modal run --detach'"
