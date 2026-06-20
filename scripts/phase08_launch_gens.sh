#!/usr/bin/env bash
# Phase 0.8 — Launch parallel SB-Bench gens on Modal for:
#   - Tier 0 #2 CF flip-rate (4 models on 9-axis split)
#   - Tier 0 #1 + Tier 1 #4 seed-triple eval (6 seed models on 9-axis split)
#
# Each gen is a detached Modal job (~30 min on A100-80GB). Total 10 jobs.
#
# Usage:
#   bash scripts/phase08_launch_gens.sh cf
#   bash scripts/phase08_launch_gens.sh seeds
#   bash scripts/phase08_launch_gens.sh all
#
# After they finish, download outputs and run the metric scripts locally:
#   scripts/phase08_compute_flip_rate.py     (CF)
#   scripts/phase08_seed_aggregate.py        (seeds — to be written)

set -euo pipefail

cd "$(dirname "$0")/.."

WHICH="${1:-all}"
PARQUET="/mnt/data/sb_bench_data/sb_bench_data_9axis.parquet"
SPLIT="/mnt/data/split_indices_9axis.json"

launch_gen() {
  local tag="$1"
  local ckpt="$2"      # empty string = vanilla
  local out="$3"
  echo "▶ [$tag] launching gen → $out"
  local CMD=( modal run --detach src/run_modal.py::run_generation
              --dataset sb_bench
              --data-path "$PARQUET"
              --output-jsonl "$out"
              --split test
              --split-indices-path "$SPLIT" )
  if [[ -n "$ckpt" ]]; then
    CMD+=( --checkpoint-dir "$ckpt" )
  else
    # `run_generation` accepts None for vanilla; CLI needs an explicit empty
    # arg. Modal's CLI parses "" as the string "" which the Python code
    # treats as truthy. Pass a sentinel that the function treats as falsy.
    CMD+=( --checkpoint-dir "" )
  fi
  "${CMD[@]}"
}

if [[ "$WHICH" == "cf" || "$WHICH" == "all" ]]; then
  echo "══ CF gens (4 models, full 9-axis split) ══"
  launch_gen base                "" \
             /mnt/data/phase08_cf/9axis_base_sbbench_gen.jsonl
  launch_gen phase08_2k          /mnt/data/output_ppo_phase08_2k/final_debiased_model \
             /mnt/data/phase08_cf/9axis_phase08_2k_sbbench_gen.jsonl
  launch_gen phase08_2k_corrOnly /mnt/data/output_ppo_phase08_2k_corrOnly/final_debiased_model \
             /mnt/data/phase08_cf/9axis_phase08_2k_corrOnly_sbbench_gen.jsonl
  launch_gen phase08_2k_biasOnly /mnt/data/output_ppo_phase08_2k_biasOnly/final_debiased_model \
             /mnt/data/phase08_cf/9axis_phase08_2k_biasOnly_sbbench_gen.jsonl
fi

if [[ "$WHICH" == "seeds" || "$WHICH" == "all" ]]; then
  echo "══ Seed-triple gens (6 models) ══"
  for L in 13 17; do
    for S in 1 2 4; do
      launch_gen "L${L}_s${S}" \
                 "/mnt/data/output_ppo_phase08_2k_L${L}_s${S}/final_debiased_model" \
                 "/mnt/data/phase08_seeds/9axis_L${L}_s${S}_sbbench_gen.jsonl"
    done
  done
fi

echo ""
echo "✅ Jobs detached. Track via:"
echo "    modal app list"
echo "    modal app logs <app-id>"
