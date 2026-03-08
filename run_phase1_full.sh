#!/usr/bin/env bash
# Full Phase 1 pipeline: SB-Bench → embeddings → DRM heads → evaluation.
# Usage: ./run_phase1_full.sh   (or bash run_phase1_full.sh)
# Override: DEVICE=cuda MODEL=Qwen/Qwen2-VL-2B-Instruct USE_SMALLSET=1 ./run_phase1_full.sh

set -e

# --- Config (override with env vars) ---
DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-./sb_bench_data/data}"
EMB_DIR="${EMB_DIR:-./embeddings_output}"
HEADS_DIR="${HEADS_DIR:-./generated_heads}"
RESULTS_JSON="${RESULTS_JSON:-./drm_head_results.json}"
N_COMPONENTS="${N_COMPONENTS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USE_SMALLSET="${USE_SMALLSET:-0}"   # set to 1 for quick test

echo "=== Phase 1: C-DeFR-L pipeline ==="
echo "  DEVICE=$DEVICE  MODEL=$MODEL  DATA_PATH=$DATA_PATH"
echo "  EMB_DIR=$EMB_DIR  HEADS_DIR=$HEADS_DIR  N_COMPONENTS=$N_COMPONENTS"
echo ""

# --- Step 0: Data (SB-Bench parquet) ---
if [ ! -d "$DATA_PATH" ] || [ -z "$(ls -A "$DATA_PATH"/*.parquet 2>/dev/null)" ]; then
  echo "[Step 0] Downloading SB-Bench..."
  python load_sb_bench.py
else
  echo "[Step 0] SB-Bench data found at $DATA_PATH (skipping download)"
fi
echo ""

# --- Step 1: Extract embeddings ---
echo "[Step 1] Extracting embeddings (cal_emb_modular.py)..."
SMALL_ARG=""
[ "$USE_SMALLSET" = "1" ] && SMALL_ARG="--use_smallset"
python cal_emb_modular.py \
  --device "$DEVICE" \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --cls_embs_path "$EMB_DIR" \
  --batch_size "$BATCH_SIZE" \
  $SMALL_ARG
echo ""

# --- Step 2: Generate DRM heads (PCA) ---
echo "[Step 2] Generating DRM heads (generate_drm_heads.py)..."
python generate_drm_heads.py \
  --input_dir "$EMB_DIR" \
  --output_dir "$HEADS_DIR" \
  --n_components "$N_COMPONENTS" \
  --case_name sb_bench
echo ""

# --- Step 3: Evaluate heads ---
COMPONENT_DIR="${HEADS_DIR}/sb_bench-PCA-component"
echo "[Step 3] Evaluating DRM heads (evaluate_drm_heads.py)..."
python evaluate_drm_heads.py \
  --emb_dir "$EMB_DIR" \
  --score_head_weight "$COMPONENT_DIR" \
  --data_path "$DATA_PATH" \
  --output_json "$RESULTS_JSON" \
  --device "$DEVICE"
echo ""

echo "=== Done. Results: $RESULTS_JSON ==="
