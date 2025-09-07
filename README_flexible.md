# Flexible Cal_Emb - Device Compatible Version

This is a modular and device-flexible version of the `cal_emb.py` script that works with CUDA, MPS (Apple Silicon), and CPU devices.

## Key Features

### 🚀 **Multi-Device Support**
- **CUDA**: Full performance with bfloat16 and flash attention
- **MPS (Apple Silicon)**: Optimized for M1/M2/M3 chips with float16
- **CPU**: Fallback mode with float32 for maximum compatibility

### 🔧 **Automatic Configuration**
- Automatically detects the best available device
- Chooses optimal precision and attention implementation
- Graceful fallbacks when features aren't supported

### 📦 **Modular Design**
- Separated concerns into classes (`DeviceManager`, `ModelLoader`, `DatasetBuilder`)
- Better error handling and logging
- More maintainable and extensible code

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Your Setup
```bash
python test_device.py
```
This will check your hardware and recommend optimal settings.

### 3. Run the Script

#### Auto-detect best device:
```bash
python run_flexible.py --data-path ./sb_bench_data/data --output-dir ./embeddings
```

#### Force specific device:
```bash
# For CUDA
python run_flexible.py --device cuda

# For Apple Silicon (MPS)
python run_flexible.py --device mps --max-length 512

# For CPU (compatibility mode)
python run_flexible.py --device cpu --force-fp32 --max-length 256 --small-dataset
```

## Configuration Options

### Device Options
- `--device {auto,cuda,mps,cpu}`: Choose device (auto-detect by default)
- `--force-fp32`: Force float32 precision for compatibility

### Memory Management
- `--max-length`: Reduce for memory constraints (default: 512)
- `--batch-size`: Processing batch size for data operations (default: 1)
- `--dataloader-batch-size`: DataLoader batch size for training (defaults to batch-size)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 16)
- `--reduce-memory`: Apply memory optimization settings automatically
- `--small-dataset`: Use subset for testing

### Paths
- `--data-path`: Path to your sb_bench_data
- `--output-dir`: Where to save embeddings
- `--log-dir`: Log directory

## Device-Specific Notes

### CUDA (Recommended for Training)
```bash
python run_flexible.py --device cuda --batch-size 2 --dataloader-batch-size 4
```
- Uses bfloat16 precision
- Flash attention enabled (if installed)
- Best performance
- Can handle larger batch sizes

### Apple Silicon (MPS)
```bash
python run_flexible.py --device mps --max-length 512 --batch-size 1
```
- Uses float16 precision
- No flash attention (not supported)
- Good performance for inference
- Moderate batch sizes recommended

### CPU (Compatibility)
```bash
python run_flexible.py --device cpu --force-fp32 --max-length 256 --batch-size 1 --reduce-memory
```
- Uses float32 precision
- Slower but works everywhere
- Keep batch size at 1
- Use memory optimization settings

## Troubleshooting

### Common Issues

1. **"Import torch could not be resolved"**
   ```bash
   pip install torch torchvision
   ```

2. **"Qwen2VL model not available"**
   ```bash
   pip install transformers>=4.35.0
   ```

3. **Out of Memory**
   - Reduce `--max-length` (try 256, 128)
   - Use `--batch-size 1` for processing
   - Use `--dataloader-batch-size 1` for training
   - Use `--reduce-memory` for automatic optimization
   - Increase `--gradient-accumulation-steps` to maintain effective batch size
   - Use `--force-fp32` for lower memory usage
   - Use `--small-dataset` for testing

4. **Flash Attention Errors**
   - The script will automatically fallback to eager attention
   - Flash attention only works on CUDA

5. **MPS Compatibility Issues**
   - Try `--force-fp32` if you get dtype errors
   - Some operations may fallback to CPU automatically

### Environment Variables
Set these if needed:
```bash
export HF_TOKEN="your_huggingface_token"
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For MPS compatibility
export OMP_NUM_THREADS=1  # For CPU memory optimization
```

### Memory Optimization Tips

1. **Automatic Memory Optimization**:
   ```bash
   python run_flexible.py --reduce-memory
   ```

2. **Manual Memory Settings**:
   ```bash
   python run_flexible.py --batch-size 1 --dataloader-batch-size 1 --max-length 256 --gradient-accumulation-steps 32
   ```

3. **Progressive Testing**:
   ```bash
   # Start with minimal settings
   python run_flexible.py --small-dataset --max-length 128 --batch-size 1
   
   # Gradually increase if successful
   python run_flexible.py --max-length 256 --batch-size 1
   python run_flexible.py --max-length 512 --batch-size 2
   ```

## Original vs Flexible Version

| Feature | Original | Flexible |
|---------|----------|----------|
| Device Support | CUDA only | CUDA/MPS/CPU |
| Precision | Fixed bfloat16 | Auto-selected |
| Flash Attention | Required | Optional |
| Error Handling | Basic | Comprehensive |
| Modularity | Monolithic | Class-based |
| Compatibility | Limited | Wide |

## Performance Expectations

| Device | Speed | Memory Usage | Recommended Use |
|--------|-------|--------------|-----------------|
| CUDA | 🚀🚀🚀 | High | Production training |
| MPS | 🚀🚀 | Medium | Local development |
| CPU | 🚀 | Low | Testing/debugging |

## Getting Your Data

The script expects data in the `sb_bench_data/data` directory with `.parquet` files. Update the `--data-path` argument to point to your actual data location.

## Development

To extend the flexible version:

1. **Add new device support**: Extend `DeviceManager` class
2. **Add new model types**: Extend `ModelLoader` class  
3. **Modify data processing**: Extend `DatasetBuilder` class
4. **Add new training features**: Extend `RewardVisualizer` class

## Support

Run `python test_device.py` to diagnose issues and get recommendations for your specific hardware setup.
