"""
Model-family adapter registry.

Provides model-specific logic for token gating and pixel value extraction
across different VLM architectures (Qwen2-VL, Qwen2.5-VL, etc.).
"""

from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseModelWrapper:
    """Base interface for model-family adapters."""

    def find_token_for_gating(self, token_ids: List[int]) -> int:
        """Find the position to gate (last token before assistant response)."""
        raise NotImplementedError

    def extract_pixel_values(self, inputs: dict) -> Tuple[list, list]:
        """Extract pixel values and grid info from processor outputs."""
        raise NotImplementedError


class QwenModelWrapper(BaseModelWrapper):
    """Adapter for Qwen2-VL / Qwen2.5-VL model family."""

    # Qwen2.5-VL uses token ID 151644 for <|im_start|> and 151645 for <|im_end|>
    ASSISTANT_START_TOKEN_ID = 151644

    def find_token_for_gating(self, token_ids: List[int]) -> int:
        """
        Find the last <|im_start|> token (start of assistant response).
        For reward models, we gate at the last token of the prompt.
        """
        last_pos = -1
        for i, t in enumerate(token_ids):
            if t == self.ASSISTANT_START_TOKEN_ID:
                last_pos = i
        if last_pos >= 0:
            return last_pos
        return len(token_ids) - 1

    def extract_pixel_values(self, inputs: dict) -> Tuple[list, list]:
        """
        Extract pixel_values and image_grid_thw from Qwen processor outputs.
        Returns per-sample splits since Qwen concatenates all images into one tensor.

        The Qwen2-VL/2.5-VL processor concatenates all patch embeddings into a
        single flat array of shape (total_patches, C, H, W). To get per-image
        arrays we split along dim-0 using the per-image patch counts derived
        from image_grid_thw (each row is [t, h, w]; patches = t*h*w).
        """
        pixel_values = inputs.get("pixel_values")
        grid_thw = inputs.get("image_grid_thw")

        if pixel_values is None:
            return [], []

        # For batched processing, pixel_values may already be a list (one per image)
        if isinstance(pixel_values, list):
            return pixel_values, grid_thw if grid_thw else [None] * len(pixel_values)

        # Single concatenated array/tensor — split per image using grid_thw.
        if grid_thw is not None and hasattr(grid_thw, '__len__') and len(grid_thw) > 1:
            patch_counts = [int(row[0]) * int(row[1]) * int(row[2]) for row in grid_thw]
            cum = [0]
            for c in patch_counts:
                cum.append(cum[-1] + c)
            splits = [pixel_values[cum[i]:cum[i + 1]] for i in range(len(patch_counts))]
            grid_list = [grid_thw[i:i + 1] for i in range(len(grid_thw))]
            return splits, grid_list

        # Single image (or no grid info) — wrap as length-1 list.
        return [pixel_values], [grid_thw] if grid_thw is not None else [None]


MODEL_REGISTRY = {
    "qwen": QwenModelWrapper,
    "qwen2_vl": QwenModelWrapper,
    "qwen2.5_vl": QwenModelWrapper,
}


def get_model_wrapper(name: str) -> BaseModelWrapper:
    """Get model wrapper by family name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model family '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]()
