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
        """
        pixel_values = inputs.get("pixel_values")
        grid_thw = inputs.get("image_grid_thw")

        if pixel_values is None:
            return [], []

        # For batched processing, pixel_values may already be a list
        if isinstance(pixel_values, list):
            return pixel_values, grid_thw if grid_thw else [None] * len(pixel_values)

        # Single tensor — return as list of one
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
