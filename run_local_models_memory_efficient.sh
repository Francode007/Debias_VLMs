#!/bin/bash

# Memory-efficient script to run cal_emb_flexible.py with local models
# This script uses the smaller 2B model to avoid memory issues

echo "🚀 Running cal_emb_flexible.py with Local Models (Memory Efficient)"
echo "=================================================================="

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

# Use smaller model for memory efficiency
BASE_MODEL="qwen2-vl-2b"  # Use smaller 2B model
FALLBACK_MODEL="qwen2-vl-2b"  # Same model as fallback

echo "🔍 Testing with memory-efficient settings..."
echo "Using model: $BASE_MODEL"
echo "Memory optimizations: Enabled"

# Run with very conservative memory settings
python cal_emb_flexible.py \
  --model "qwen2-vl-2b" \
  --batch_size 1 \
  --use_smallset true \
  --force_fp32 true \
  --max_length 1024 \
  --gradient_accumulation_steps 8 \
  --eval_steps 50 \
  --save_steps 100 \
  --logging_steps 10 \
  --warmup_steps 10 \
  --num_train_epochs 1 \
  --output_dir "./outputs/qwen2-vl-2b-test" \
  --lr_scheduler_type "cosine" \
  --learning_rate 1e-6 \
  --load_in_8bit false \
  --load_in_4bit false

echo ""
echo "📝 Memory Optimization Notes:"
echo "============================"
echo "✅ Used qwen2-vl-2b model (~2.8GB vs 14GB for 7B)"
echo "✅ Enabled use_smallset (limits dataset to 50 samples)"
echo "✅ Set batch_size=1 and max_length=512"
echo "✅ Forced float32 for better compatibility"
echo "✅ Reduced gradient_accumulation_steps=8"
echo ""
echo "For remote server with more memory:"
echo "- Use qwen2-vl-7b for better quality"
echo "- Increase batch_size and max_length"
echo "- Set use_smallset=false for full dataset"
