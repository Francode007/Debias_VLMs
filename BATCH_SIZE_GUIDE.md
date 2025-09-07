# Batch Size Implementation Guide

## Overview

The batch size parameter has been introduced across all files to help manage memory constraints when running the cal_emb scripts. This is especially important for different hardware configurations (CUDA, MPS, CPU) and various memory limitations.

## Key Changes Made

### 1. ScriptArguments Updates

Added two new parameters:
- `batch_size`: Controls processing batch size for data operations (default: 1)
- `dataloader_batch_size`: Controls DataLoader batch size for training (defaults to batch_size if not set)

### 2. Memory Management Features

#### Automatic Memory Optimization
```bash
python run_flexible.py --reduce-memory
```
This automatically applies:
- batch_size = 1
- dataloader_batch_size = 1  
- max_length = 256 (or current max_length, whichever is smaller)
- gradient_accumulation_steps = max(current, 32)

#### Manual Control
```bash
python run_flexible.py --batch-size 2 --dataloader-batch-size 4 --max-length 512
```

### 3. Device-Specific Recommendations

| Device | Processing Batch | DataLoader Batch | Max Length | Notes |
|--------|------------------|------------------|------------|--------|
| CUDA | 2-4 | 4-8 | 512-1024 | Depends on GPU memory |
| MPS | 1-2 | 1-2 | 256-512 | Unified memory limits |
| CPU | 1 | 1 | 128-256 | Minimize memory usage |

## Implementation Details

### DatasetBuilder
- Added `batch_size_for_map` to control dataset mapping operations
- Limited to max 100 to prevent excessive memory usage during preprocessing

### RewardDataCollatorWithPadding
- Added batch size validation to prevent memory overflow
- Warns and truncates if incoming batch exceeds configured size

### RewardVisualizer
- Enhanced batch processing in `visualize_samples`
- Better handling of both single and multi-sample batches
- Improved progress tracking and intermediate saving

### Training Arguments
- `dataloader_num_workers = 0` to reduce memory overhead
- Automatic batch size configuration from script arguments

## Usage Examples

### Basic Usage
```bash
# Auto-detect device with default settings
python run_flexible.py

# Explicit device with memory optimization
python run_flexible.py --device mps --reduce-memory
```

### Memory-Constrained Environments
```bash
# Minimal memory usage
python run_flexible.py --device cpu --batch-size 1 --max-length 128 --reduce-memory

# Progressive testing
python run_flexible.py --small-dataset --max-length 128  # Test first
python run_flexible.py --max-length 256 --batch-size 1   # Then scale up
```

### High-Memory Environments
```bash
# CUDA with larger batches
python run_flexible.py --device cuda --batch-size 4 --dataloader-batch-size 8 --max-length 1024

# Balanced MPS usage
python run_flexible.py --device mps --batch-size 2 --max-length 512
```

## Troubleshooting Memory Issues

### Out of Memory Errors
1. **Reduce batch sizes**: `--batch-size 1 --dataloader-batch-size 1`
2. **Reduce sequence length**: `--max-length 256` or `--max-length 128`
3. **Use memory optimization**: `--reduce-memory`
4. **Force lower precision**: `--force-fp32`
5. **Use gradient accumulation**: `--gradient-accumulation-steps 32`

### Slow Performance
1. **Increase batch sizes** (if memory allows): `--batch-size 2 --dataloader-batch-size 4`
2. **Optimize sequence length**: Find balance between memory and speed
3. **Use appropriate device**: CUDA > MPS > CPU for performance

### Validation
Run the device test to get personalized recommendations:
```bash
python test_device.py
```

## Gradient Accumulation vs Batch Size

**Effective Batch Size** = `batch_size` × `gradient_accumulation_steps`

Examples:
- `batch_size=1, gradient_accumulation_steps=32` → Effective batch size: 32
- `batch_size=4, gradient_accumulation_steps=8` → Effective batch size: 32

Use gradient accumulation to maintain training effectiveness while reducing memory usage.

## Monitoring Memory Usage

### During Processing
- Watch for "Processing with batch size: X" messages
- Monitor intermediate save frequency (every 1000 samples)
- Check for batch truncation warnings

### System Monitoring
```bash
# Monitor GPU memory (CUDA)
nvidia-smi -l 1

# Monitor system memory (macOS)
memory_pressure

# Monitor system memory (general)
htop
```

## Best Practices

1. **Start Small**: Begin with `--reduce-memory` and scale up
2. **Test First**: Use `--small-dataset` to validate settings
3. **Progressive Scaling**: Gradually increase batch sizes and sequence lengths
4. **Device-Appropriate**: Follow device-specific recommendations
5. **Monitor Resources**: Keep an eye on memory usage during processing

This implementation provides flexible memory management while maintaining the functionality and performance of the original script across different hardware configurations.
