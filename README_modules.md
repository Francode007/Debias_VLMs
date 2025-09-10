# Reward Model Training and Visualization Pipeline

## Overview

This modularized pipeline trains and visualizes reward models for vision-language models, specifically designed for bias analysis and evaluation on the SB-Bench dataset. The system processes multimodal data (images + text) to train reward models that can distinguish between preferred and non-preferred responses.

## Architecture Overview

The pipeline consists of several modular components that work together to:

1. **Load and configure vision-language models** with device optimization
2. **Process multimodal datasets** with memory management
3. **Train reward models** using preference ranking losses
4. **Extract embeddings and visualizations** for bias analysis
5. **Handle various hardware configurations** (CUDA, MPS, CPU)

## Project Structure

```
Debias_VLMs/
├── cal_emb_flexible.py           # Original monolithic script
├── cal_emb_modular.py           # New modular main script
├── modules/                      # Modularized components
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration parameters
│   ├── device_manager.py        # Device detection and optimization
│   ├── model_loader.py          # Model loading with fallbacks
│   ├── dataset_builder.py       # Dataset processing
│   ├── model_architecture.py    # Custom forward functions
│   ├── data_collator.py         # Batch processing
│   └── reward_trainer.py        # Training and visualization
└── README_modules.md            # This documentation
```

## Module Documentation

### 1. Configuration Module (`modules/config.py`)

**Purpose**: Centralized configuration management for all pipeline parameters.

**Key Classes**:
- `ScriptArguments`: Comprehensive configuration dataclass

**Key Parameters**:

#### Training Parameters
- `per_device_train_batch_size` (default: 1): Training batch size per device
- `per_device_eval_batch_size` (default: 1): Evaluation batch size per device
- `gradient_accumulation_steps` (default: 16): Steps to accumulate gradients
- `learning_rate` (default: 5e-6): Learning rate for training
- `num_train_epochs` (default: 1): Number of training epochs
- `max_length` (default: 1024): Maximum sequence length

#### Memory Management
- `batch_size` (default: 1): Batch size for data processing
- `dataloader_batch_size` (default: None): DataLoader batch size
- `use_smallset` (default: False): Use small dataset for testing

#### Model Parameters
- `model` (default: 'Qwen/Qwen2.5-VL-7B-Instruct'): Primary model name
- `fallback_model` (default: 'Qwen/Qwen2-VL-7B-Instruct'): Fallback model
- `load_in_8bit` (default: False): 8-bit quantization
- `load_in_4bit` (default: False): 4-bit quantization

#### Device Parameters
- `device` (default: "auto"): Target device (auto/cuda/mps/cpu)
- `force_fp32` (default: False): Force float32 precision

#### Loss Configuration
- `loss_type` (default: 'origin'): Loss function type
  - 'origin': Standard log-sigmoid ranking loss
  - 'margin': Margin-based ranking loss  
  - 'labelsmooth': Label-smoothed ranking loss

---

### 2. Device Manager Module (`modules/device_manager.py`)

**Purpose**: Automatic device detection and optimization for different hardware.

**Key Classes**:
- `DeviceManager`: Static methods for device optimization

**Key Methods**:

#### `get_optimal_device()`
- **Input**: None
- **Output**: Device string ("cuda", "mps", "cpu")
- **Process**: Detects best available device with CUDA > MPS > CPU priority
- **Purpose**: Automatic hardware detection without manual configuration

#### `get_optimal_dtype(device: str)`
- **Input**: Target device string
- **Output**: Optimal torch.dtype for the device
- **Process**: Maps devices to optimal data types (CUDA→bfloat16, MPS→float16, CPU→float32)
- **Purpose**: Optimize memory usage and computation speed

#### `get_attention_implementation(device: str)`
- **Input**: Target device string
- **Output**: Attention implementation ("flash_attention_2" or "eager")
- **Process**: Enables Flash Attention on CUDA if available, falls back to eager
- **Purpose**: Optimize attention computation for transformer models

#### `configure_torch_backends(device: str)`
- **Input**: Target device string
- **Output**: None (configures global settings)
- **Process**: Enables device-specific optimizations (TF32 for CUDA, fallback for MPS)
- **Purpose**: Maximize computational performance

---

### 3. Model Loader Module (`modules/model_loader.py`)

**Purpose**: Robust model loading with device optimization and fallback mechanisms.

**Key Classes**:
- `ModelLoader`: Handles model and processor loading

**Key Methods**:

#### `__init__(script_args, device_manager)`
- **Input**: Configuration and device manager
- **Output**: Initialized loader instance
- **Process**: Sets up device, dtype, attention implementation, and local model config
- **Purpose**: Prepare optimal loading environment

#### `load_model_and_processor()`
- **Input**: None (uses instance configuration)
- **Output**: Tuple of (model, processor)
- **Process**: 
  1. Converts model names to local paths if available
  2. Tries multiple models with fallback mechanisms
  3. Determines correct model class (Qwen2.5VL vs Qwen2VL)
  4. Configures device mapping and model kwargs
  5. Loads with optimal settings, falls back to compatible settings
  6. Adds score head for reward model functionality
- **Purpose**: Robustly load vision-language models with comprehensive error handling

**Fallback Strategy**:
1. Try primary model with optimal settings
2. Try fallback model with optimal settings  
3. Try CPU with float32 for compatibility
4. Raise error if all attempts fail

---

### 4. Dataset Builder Module (`modules/dataset_builder.py`)

**Purpose**: Memory-efficient dataset processing for vision-language reward training.

**Key Classes**:
- `DatasetBuilder`: Handles dataset creation and processing

**Key Methods**:

#### `find_token_for_gating(lst: List[int], model_family: str)`
- **Input**: Token list and model family
- **Output**: Index of token pattern
- **Process**: Searches backwards for assistant token pattern
- **Purpose**: Find response start for reward gating

#### `build_dataset(data_path, processor, split='train', size=None)`
- **Input**: Data path, processor, split name, size limit
- **Output**: Processed HuggingFace Dataset
- **Process**:
  1. Loads parquet files with memory limits
  2. Applies row/file constraints for memory management
  3. Concatenates dataframes efficiently
  4. Converts to HuggingFace Dataset format
  5. Applies formatting function to each example
  6. Filters by sequence length constraints
  7. Sets tensor format for PyTorch
- **Purpose**: Create memory-efficient dataset for training

#### `_formatting_func(example, processor)`
- **Input**: Raw example and processor
- **Output**: Formatted example dictionary
- **Process**:
  1. Validates example structure
  2. Handles flattened parquet file structure
  3. Loads image from bytes and converts to RGB
  4. Extracts chosen/rejected responses based on labels
  5. Creates conversation messages with image and text
  6. Applies chat templates for proper tokenization
  7. Processes inputs with padding and length constraints
  8. Calculates prompt length for reward computation
- **Purpose**: Transform raw data into reward model format

**Memory Management Features**:
- File count limits (10 files max, 2 for small set)
- Row limits (1000 per file, 100 for small set)
- Dataset size limits (2000 samples, 50 for small set)
- Single-process mapping to avoid memory issues

---

### 5. Model Architecture Module (`modules/model_architecture.py`)

**Purpose**: Custom forward functions for reward model training with vision-language models.

**Key Functions**:

#### `create_custom_forward(model, dtype)`
- **Input**: Base model and target data type
- **Output**: Custom forward function
- **Process**: Creates specialized forward pass for reward models
- **Purpose**: Enable reward model training on vision-language inputs

**Custom Forward Function Features**:
- **Device Management**: Ensures all inputs are on correct device with proper dtype
- **Architecture Handling**: Works with different model architectures (separate transformer vs integrated)
- **Error Recovery**: Retries without pixel_values if vision processing fails
- **Reward Computation**: Uses linear score head to compute rewards from hidden states
- **Sequence Pooling**: Properly pools logits at sequence end positions
- **Loss Computation**: Supports multiple loss functions (regression, classification, multi-label)
- **Embedding Extraction**: Extracts embeddings for chosen, rejected, and prompt for analysis

**Input Processing**:
1. Moves inputs to correct device and dtype
2. Routes through appropriate model component
3. Handles vision-language processing with fallback
4. Computes reward scores using score head
5. Calculates sequence lengths considering padding
6. Pools logits at appropriate positions
7. Extracts embeddings for visualization
8. Returns structured output

---

### 6. Data Collator Module (`modules/data_collator.py`)

**Purpose**: Batch processing and collation for reward model training.

**Key Classes**:
- `RewardDataCollatorWithPadding`: Custom data collator for reward models

**Key Methods**:

#### `__call__(features: List[Dict[str, Any]])`
- **Input**: List of processed examples
- **Output**: Batched data dictionary
- **Process**:
  1. Limits batch size to prevent memory overflow
  2. Extracts chosen and rejected data from each example
  3. Interleaves chosen and rejected sequences in batch
  4. Stacks tensors for efficient GPU processing
  5. Preserves metadata from first example
- **Purpose**: Transform individual examples into GPU-ready batches

**Batch Structure**:
- Interleaved chosen/rejected pairs for reward comparison
- Consistent tensor stacking for GPU efficiency
- Metadata preservation for tracking and analysis
- Memory management through batch size limiting

---

### 7. Reward Trainer Module (`modules/reward_trainer.py`)

**Purpose**: Custom trainer for reward model training and visualization.

**Key Classes**:
- `RewardVisualizer`: Extends RewardTrainer with visualization capabilities

**Key Methods**:

#### `compute_loss(model, inputs, return_outputs=False)`
- **Input**: Model, input batch, output flag
- **Output**: Loss tensor and optionally additional outputs
- **Process**:
  1. Forward pass through model with error handling
  2. Extracts reward scores and embeddings
  3. Splits rewards into chosen/rejected pairs
  4. Computes ranking loss based on configured type
- **Purpose**: Compute reward model ranking loss

**Loss Types**:
- **Origin**: `-log_sigmoid(r_chosen - r_rejected)`
- **Margin**: `-log_sigmoid(r_chosen - r_rejected - margin)`
- **Label Smooth**: `0.9 * loss_chosen + 0.1 * loss_rejected`

#### `prediction_step(model, inputs, prediction_loss_only, ignore_keys=None)`
- **Input**: Model, inputs, flags
- **Output**: Loss, logits, labels, embeddings
- **Process**:
  1. Performs forward pass with gradients disabled
  2. Computes loss and extracts outputs
  3. Processes logits for evaluation format
  4. Creates dummy labels for reward evaluation
- **Purpose**: Enable evaluation without parameter updates

#### `visualize_samples(num_print_samples, cls_embs_path, data_path)`
- **Input**: Sample limit, embedding path, data path
- **Output**: Results DataFrame with predictions and metadata
- **Process**:
  1. Iterates through evaluation dataloader
  2. Processes individual samples from batches
  3. Performs predictions to get logits and embeddings
  4. Extracts metadata and determines prediction flags
  5. Saves embeddings to individual files
  6. Updates results table with comprehensive data
  7. Saves intermediate results for large datasets
- **Purpose**: Extract embeddings and predictions for analysis

**Visualization Features**:
- Individual embedding file storage
- Comprehensive metadata extraction
- Progress tracking and error handling
- Intermediate result saving
- Multi-process support with accelerate

---

### 8. Main Module (`cal_emb_modular.py`)

**Purpose**: Orchestrates the entire pipeline execution.

**Key Functions**:

#### `setup_environment(script_args)`
- **Input**: Configuration parameters
- **Output**: None (configures environment)
- **Process**: Sets HuggingFace token and configures logging
- **Purpose**: Prepare execution environment

#### `create_training_arguments(script_args, model_loader)`
- **Input**: Configuration and model loader
- **Output**: TrainingArguments object
- **Process**: Creates comprehensive training configuration
- **Purpose**: Configure training parameters for optimal performance

#### `create_metrics()`
- **Input**: None
- **Output**: Metrics computation function
- **Process**: Sets up accuracy metrics for reward model evaluation
- **Purpose**: Provide evaluation metrics

#### `main()`
- **Input**: None (uses command line arguments)
- **Output**: Results DataFrame
- **Process**:
  1. Parses command line arguments
  2. Sets up environment and device manager
  3. Loads model and processor
  4. Applies custom forward function
  5. Builds and processes dataset
  6. Creates training arguments and metrics
  7. Initializes reward trainer
  8. Runs visualization process
- **Purpose**: Coordinate entire pipeline execution

---

## Pipeline Flow

### 1. Initialization Phase
```
Command Line Args → ScriptArguments → Environment Setup
                 ↓
Device Detection → Model Loading → Custom Forward Function
                 ↓
Dataset Loading → Processing → Formatting
```

### 2. Training Setup Phase
```
Training Arguments → Metrics Setup → Data Collator
                 ↓
RewardVisualizer Initialization
```

### 3. Visualization Phase
```
Evaluation Dataloader → Batch Processing → Prediction
                     ↓
Embedding Extraction → File Storage → Results Table
```

## Key Parameters and Their Effects

### Memory Management
- `batch_size=1`: Processes one example at a time to minimize memory usage
- `use_smallset=True`: Limits to 50 samples for quick testing
- `max_length=1024`: Controls maximum sequence length (affects memory significantly)
- `gradient_accumulation_steps=16`: Increases effective batch size without memory overhead

### Model Configuration
- `model`: Primary model to load (supports local paths through local_model_config)
- `fallback_model`: Backup model if primary fails
- `device="auto"`: Automatically selects best available device
- `force_fp32=True`: Forces float32 for compatibility debugging

### Training Configuration
- `loss_type="origin"`: Standard ranking loss for reward model training
- `learning_rate=5e-6`: Conservative learning rate for vision-language models
- `num_train_epochs=1`: Usually sufficient for reward model training

### Dataset Configuration
- `data_path`: Directory containing parquet files with vision-language data
- Expected parquet structure:
  - `file_name.bytes`: Image data as bytes
  - `context`: Context text
  - `question`: Question text
  - `ans0`, `ans1`: Answer options
  - `label`: Preferred answer label (0 or 1)
  - `additional_metadata`: Additional information

## Usage Examples

### Basic Usage
```bash
python cal_emb_modular.py \
    --data_path ./sb_bench_data/data \
    --use_smallset \
    --batch_size 1 \
    --device auto
```

### Memory-Constrained Environment
```bash
python cal_emb_modular.py \
    --data_path ./sb_bench_data/data \
    --use_smallset \
    --batch_size 1 \
    --max_length 512 \
    --force_fp32 \
    --device cpu
```

### Full Dataset Processing
```bash
python cal_emb_modular.py \
    --data_path ./sb_bench_data/data \
    --batch_size 2 \
    --max_length 1024 \
    --gradient_accumulation_steps 8 \
    --device cuda
```

## Output Files

### Embeddings
- `embeddings_output/emb_{data_index}.npy`: Individual embedding files
- Contains 3D array: [chosen_emb, rejected_emb, prompt_emb]

### Results
- `data_embeddings_output_{process_id}.csv`: Comprehensive results table
- Contains: prompts, responses, predictions, logits, embedding paths, metadata

### Logs
- Console output with detailed progress and error information
- Memory usage monitoring and optimization suggestions

## Troubleshooting

### Memory Issues
1. Enable `use_smallset=True` for testing
2. Reduce `batch_size` to 1
3. Decrease `max_length` (e.g., 512)
4. Use `force_fp32=True` if precision issues occur
5. Set `device="cpu"` for maximum compatibility

### Model Loading Issues
1. Check local_model_config.py for local model paths
2. Verify HuggingFace token for private models
3. Try fallback model if primary fails
4. Check transformers library version compatibility

### Dataset Issues
1. Verify parquet file structure matches expected format
2. Check image data is properly embedded as bytes
3. Ensure all required columns are present
4. Use debug mode to trace processing errors

## Performance Optimization

### For CUDA GPUs
- Use bfloat16 precision
- Enable Flash Attention if available
- Increase batch size based on GPU memory
- Use gradient accumulation for larger effective batch sizes

### For Apple Silicon (MPS)
- Use float16 precision
- Keep batch sizes small (1-2)
- Monitor memory usage carefully
- Enable MPS fallback for compatibility

### For CPU
- Use float32 precision
- Minimize batch size (1)
- Reduce sequence length
- Consider using smaller models

## Future Extensions

### Model Support
- Add support for other vision-language model architectures
- Implement LoRA fine-tuning capabilities
- Add quantization support for larger models

### Training Features
- Implement full training loop (currently visualization only)
- Add validation split handling
- Support for custom loss functions

### Analysis Tools
- Bias analysis visualizations
- Embedding clustering and analysis
- Performance comparison tools
- Statistical significance testing

---

This modularized architecture provides a robust, maintainable, and extensible foundation for reward model training and analysis on vision-language models, with comprehensive error handling and memory management for various hardware configurations.
