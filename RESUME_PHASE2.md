# VLM Debiasing Pipeline: Phase 2 RL Profiling Status

## Current Status
We are currently profiling **Phase 2 (RL PPO Generation)** of the VLM debiasing pipeline. 
The goal is to ensure the pipeline can run within a compute budget and resolve any architectural or performance issues.

## Key Fixes Applied
1. **Model Loading & Compatibility**:
   - Refactored `ModelLoader` to explicitly support `Qwen2_5_VLForConditionalGeneration`.
   - Replaced brittle `AutoModel` imports with localized class resolution.
   - Centralized model initialization to ensure `./models/` local paths are prioritized.
2. **Architectural Stability**:
   - Resolved `AttributeError: hidden_size` in `PPOVLMController` by implementing a multi-strategy configuration lookup.
   - Fixed `NameError: dtype` in `modules/model_architecture.py`.
   - Corrected `grid_thw` dimension preservation in dataset builders to prevent 0-d tensor errors.
3. **Performance & Hangs**:
   - Disabled multiprocessing (`num_proc=1`) in both Phase 1 and Phase 2 dataset mapping to avoid deadlocks/pickling hangs with custom VLM processors.
   - Set `return_tensors=None` in dataset mapping to handle variable-length sequences before collation.
4. **Data Collation**:
   - Fixed `RLDataCollatorWithPadding` to correctly stack `image_grid_thw` into a 2D tensor (`[N, 3]`), preventing unpacking errors in the vision model.

## Active Blocker
**CUDA Error (Device-side Assert)**: When running with `batch_size > 1` (e.g., `--batch_size 4`), the PPO generation loop fails with:
`RuntimeError: CUDA error: device-side assert triggered` at `modeling_qwen2_5_vl.py:483`.
This is specifically an "index out of bounds" in the vision model's rotary position embeddings.

## Next Steps for New Chat
1. **Debug CUDA Assert**: Investigate the exact shapes of `pixel_values` and `image_grid_thw` in the `RLDataCollatorWithPadding`. The assertion failure suggests a mismatch between the number of patches and the grid dimensions in the batched input.
2. **Verify Generation Loop**: Once the CUDA error is fixed, validate that the dual-rollout generation (Policy + Extractor) completes a full PPO step.
3. **Profile & Extrapolate**: Generate the final profiling report for Phase 2.

## How to Resume
Start by running the profiling script with `CUDA_LAUNCH_BLOCKING=1` to confirm the exact location of the out-of-bounds error:
```bash
CUDA_LAUNCH_BLOCKING=1 python profile_rl_pipeline.py --batch_size 4 --gradient_accumulation 4
```
Check `modules/rl_data_collator.py` to ensure `pixel_values` are concatenated correctly across the batch without losing patch order or dimension alignment.
