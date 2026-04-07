#!/usr/bin/env bash
# Full Phase 1 pipeline: SB-Bench → embeddings → DRM heads → evaluation.
# Usage: ./run_phase1_full.sh   (or bash run_phase1_full.sh)
# Override: DEVICE=cuda MODEL=Qwen/Qwen2.5-VL-7B-Instruct USE_SMALLSET=1 ./run_phase1_full.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create .venv and install requirements if missing
if [ ! -x ".venv/bin/python" ]; then
  echo "[Setup] No .venv found; creating and installing from requirements.txt..."
  if command -v python3 &>/dev/null; then
    python3 -m venv .venv
  elif command -v python &>/dev/null; then
    python -m venv .venv
  else
    echo "Error: need python3 or python to create .venv" >&2
    exit 1
  fi
  .venv/bin/pip install -r requirements.txt
  echo "[Setup] Done."
  echo ""
fi
PYTHON=".venv/bin/python"

# --- Config (override with env vars) ---
DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATA_PATH="${DATA_PATH:-./sb_bench_data/data}"
EMB_DIR="${EMB_DIR:-./embeddings_output}"
HEADS_DIR="${HEADS_DIR:-./generated_heads}"
RESULTS_JSON="${RESULTS_JSON:-./drm_head_results.json}"
N_COMPONENTS="${N_COMPONENTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
USE_SMALLSET="${USE_SMALLSET:-0}"   # set to 1 for quick test

echo "=== Phase 1: C-DeFR-L pipeline ==="
echo "  DEVICE=$DEVICE  MODEL=$MODEL  DATA_PATH=$DATA_PATH"
echo "  EMB_DIR=$EMB_DIR  HEADS_DIR=$HEADS_DIR  N_COMPONENTS=$N_COMPONENTS"
echo ""

# --- Step 0: Data (SB-Bench parquet) ---
if [ ! -d "$DATA_PATH" ] || [ -z "$(ls -A "$DATA_PATH"/*.parquet 2>/dev/null)" ]; then
  echo "[Step 0] Downloading SB-Bench..."
  "$PYTHON" load_sb_bench.py
else
  echo "[Step 0] SB-Bench data found at $DATA_PATH (skipping download)"
fi
echo ""

# --- Step 1: Extract embeddings ---
# Clean stale embeddings so they are always re-extracted with current settings
if [ -d "$EMB_DIR" ]; then
  echo "[Step 1] Removing old embeddings in $EMB_DIR..."
  rm -f "$EMB_DIR"/emb_*.npy
fi
echo "[Step 1] Extracting embeddings (cal_emb_modular.py)..."
SMALL_ARG=""
[ "$USE_SMALLSET" = "1" ] && SMALL_ARG="--use_smallset"
# Forward --force_fp32 if user passed it, or if device is MPS/CPU
FORCE_FP32="${FORCE_FP32:-0}"
FP32_ARG=""
if [ "$FORCE_FP32" = "1" ] || [ "$DEVICE" = "mps" ] || [ "$DEVICE" = "cpu" ]; then
  FP32_ARG="--force_fp32"
fi
"$PYTHON" cal_emb_modular.py \
  --device "$DEVICE" \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --cls_embs_path "$EMB_DIR" \
  --batch_size "$BATCH_SIZE" \
  $FP32_ARG \
  $SMALL_ARG
echo ""

# --- Step 2: Generate DRM heads (PCA) ---
echo "[Step 2] Generating DRM heads (generate_drm_heads.py)..."
"$PYTHON" generate_drm_heads.py \
  --input_dir "$EMB_DIR" \
  --output_dir "$HEADS_DIR" \
  --n_components "$N_COMPONENTS" \
  --case_name sb_bench
echo ""

# --- Step 3: Evaluate heads ---
COMPONENT_DIR="${HEADS_DIR}/sb_bench-PCA-component"
if [ -d "$COMPONENT_DIR" ] && ls "$COMPONENT_DIR"/*.pth >/dev/null 2>&1; then
  echo "[Step 3] Evaluating DRM heads (evaluate_drm_heads.py)..."
  "$PYTHON" evaluate_drm_heads.py \
    --emb_dir "$EMB_DIR" \
    --score_head_weight "$COMPONENT_DIR" \
    --data_path "$DATA_PATH" \
    --output_json "$RESULTS_JSON" \
    --device "$DEVICE"
  echo ""
else
  echo "[Step 3] Skipping evaluation: no DRM head .pth files found in $COMPONENT_DIR (e.g. PCA had no valid samples)."
fi

echo "=== Done. Results (if any): $RESULTS_JSON ==="
