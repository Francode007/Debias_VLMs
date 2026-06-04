#!/usr/bin/env bash
# Phase 0.8 Strategic Plan — Day 1 launch cheatsheet.
#
# Source `debias_env/bin/activate` first; run from repo root.
# Use `modal run --detach …` for the long jobs so disconnects don't kill them.
#
# This file is documentation-by-example. Do NOT execute it whole.
exit 1

# ─── 0. Wash-out diagnostic (~15 min total) ─────────────────────────────────
# Build per-layer probe heads (npz → .pth + kept_heads_probe.json):
bash scripts/phase08_washout_make_heads.sh

# Score each variant at every layer:
bash scripts/phase08_washout_score.sh

# Classify Pattern 1/2/3 locally:
python scripts/phase08_washout_classify.py \
    --score-root /mnt/data/phase08_washout \
    --output-json Phase0.8/washout_diagnostic.json


# ─── 1. Seed × 3 replication, L13 (full reward) ─────────────────────────────
# Three detached apps, identical hyperparams to the headline 2k run.
for SEED in 1 2 3; do
  modal run --detach src/run_modal.py::run_training \
      --epochs 1 \
      --output-dir "/mnt/data/output_ppo_phase08_2k_L13_s${SEED}" \
      --dataset sb_bench --model-family qwen \
      --kl-beta 0.1 --target-kl 0.02 \
      --learning-rate 5e-6 --lora-r 16 --lora-alpha 32 \
      --batch-size 8 --max-gen-tokens 8 \
      --max-train-samples 2000 \
      --reward-mode bias_aligned \
      --reward-head-layer 13 \
      --bias-aligned-coef 1.0 --correctness-coef 1.0 --ambig-preservation-coef 0.5 \
      --use-frozen-phi \
      --midtrain-eval-every-steps 50 --midtrain-eval-samples 64 \
      --seed "${SEED}"
done


# ─── 4. Seed × 3, L17 (probe at deeper layer) ───────────────────────────────
for SEED in 1 2 3; do
  modal run --detach src/run_modal.py::run_training \
      --epochs 1 \
      --output-dir "/mnt/data/output_ppo_phase08_2k_L17_s${SEED}" \
      --dataset sb_bench --model-family qwen \
      --kl-beta 0.1 --target-kl 0.02 \
      --learning-rate 5e-6 --lora-r 16 --lora-alpha 32 \
      --batch-size 8 --max-gen-tokens 8 \
      --max-train-samples 2000 \
      --reward-mode bias_aligned \
      --reward-head-layer 17 \
      --reward-heads-dir-override "/mnt/data/generated_heads_probe_L17_base_biasA/sb_bench-PROBE-component" \
      --bias-aligned-coef 1.0 --correctness-coef 1.0 --ambig-preservation-coef 0.5 \
      --use-frozen-phi \
      --midtrain-eval-every-steps 50 --midtrain-eval-samples 64 \
      --seed "${SEED}"
done
# NB: L17 reward requires the L17 probe head dir to exist on the volume.
# If you ran phase08_washout_make_heads.sh first, it built L17 already.


# ─── 2. Counterfactual eval set (local) ─────────────────────────────────────
# Build the polarity-paired index (no GPU needed):
python scripts/phase08_build_counterfactual_pairs.py \
    --parquet /mnt/data/sb_bench_data/sb_bench_data.parquet \
    --out Phase0.8/counterfactual_eval/cf_pairs.parquet \
    --report-json Phase0.8/counterfactual_eval/cf_pairs_report.json
# Inspect pair counts. Hand-validate ~50 random pairs by image+question text
# before trusting; if image_mismatch > 0 the SB-Bench id grammar in
# phase08_build_counterfactual_pairs.py needs a tweak.


# ─── Post-Day-1 ──────────────────────────────────────────────────────────────
# After the 6 seed runs finish (~3 h GPU each, run in parallel):
#   - run SB-Bench eval + VLBias transfer eval on each final_debiased_model
#   - aggregate into Phase0.8/seed_replication/summary.json
#   - append §9 (replication) to Phase0.8_Critical_Review.md
