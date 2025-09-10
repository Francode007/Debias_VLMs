#!/bin/bash

# Script to run cal_emb_flexible.py with local models
# This script demonstrates how to use local models without HuggingFace API access

echo "🚀 Running cal_emb_flexible.py with Local Models"
echo "================================================"

# Set environment variables to disable HuggingFace API calls
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Check if we're in the right directory
if [ ! -f "cal_emb_flexible.py" ]; then
    echo "❌ Error: cal_emb_flexible.py not found in current directory"
    echo "Please run this script from the Debias_VLMs directory"
    exit 1
fi

# Activate virtual environment
if [ -f "debias_env/bin/activate" ]; then
    source debias_env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Warning: Virtual environment not found, using system Python"
fi

# Set up basic arguments for local model usage
BASE_MODEL="qwen2-vl-7b"  # Use local model ID
FALLBACK_MODEL="qwen2-vl-2b"  # Fallback to smaller model if needed

# Example run with minimal arguments for testing
echo "🔍 Testing local model loading..."
python cal_emb_flexible.py \
    --base_model "$BASE_MODEL" \
    --fallback_model "$FALLBACK_MODEL" \
    --device "auto" \
    --debug true \
    --log_dir "./logs" \
    --use_smallset true \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --save_steps 25 \
    --batch_size 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --wandb_name "local_test" \
    --cls_embs_path "./embeddings_output_local_test"

echo ""
echo "📝 Notes for Remote Server Usage:"
echo "================================="
echo "1. Copy your entire models_cache/ directory to the remote server"
echo "2. Copy local_model_config.py to the remote server"
echo "3. Update the paths in local_model_config.py to match the remote server paths"
echo "4. Set environment variables: export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1"
echo "5. Use local model IDs (qwen2-vl-7b, qwen2-vl-2b, etc.) instead of HuggingFace model names"
echo ""
echo "Example for remote server:"
echo "python cal_emb_flexible.py --base_model qwen2-vl-7b --device cuda"
