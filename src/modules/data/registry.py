"""
Dataset adapter registry.

Provides dataset-specific adapters for extracting image bytes, prompt text,
and chosen/rejected responses from different benchmark formats (SB-Bench, POPE).
"""

import io
from typing import Tuple, Optional


class BaseDatasetAdapter:
    """Base interface for dataset adapters."""

    def extract_image_bytes(self, example: dict) -> Optional[bytes]:
        raise NotImplementedError

    def get_chosen_rejected(self, example: dict) -> Tuple[str, str]:
        raise NotImplementedError

    def get_chosen_rejected_letter(self, example: dict) -> Tuple[str, str]:
        """Return the (chosen, rejected) pair as single-letter answers (e.g. ("A", "B")).

        Used when extracting reward-model embeddings with `--completion_format letter`,
        so the head-build forward pass conditions on the same single-letter completion
        that PPO scores at training time (D3 fix; see Phase 0.6 plan).
        """
        raise NotImplementedError

    def get_prompt_text(self, example: dict) -> str:
        raise NotImplementedError


class SBBenchAdapter(BaseDatasetAdapter):
    """Adapter for SB-Bench (social bias benchmark) dataset format."""

    SB_BENCH_CATEGORIES = [
        "Age", "Disability", "Gender", "Nationality",
        "Physical Appearance", "Race/Ethnicity", "Religion", "SES", "Sexual Orientation",
    ]

    def extract_image_bytes(self, example: dict) -> Optional[bytes]:
        # Handle nested dict (file_name: {bytes: ...})
        img_field = example.get("file_name") or example.get("image")
        if img_field is not None:
            if isinstance(img_field, dict) and "bytes" in img_field:
                return img_field["bytes"]
            if isinstance(img_field, bytes):
                return img_field
        # Handle flattened parquet columns (file_name.bytes)
        flat_bytes = example.get("file_name.bytes")
        if flat_bytes is not None:
            return flat_bytes
        return None

    def get_chosen_rejected_letter(self, example: dict) -> Tuple[str, str]:
        # Mirror get_chosen_rejected's "first wrong" selection but return the
        # letter (A/B/C) instead of the free-text answer.
        label = int(example.get("label", 0))
        letters = ["A", "B", "C"]
        chosen = letters[label]
        rejected_options = [l for i, l in enumerate(letters) if i != label]
        rejected = rejected_options[0] if rejected_options else ""
        return chosen, rejected

    def get_chosen_rejected(self, example: dict) -> Tuple[str, str]:
        label = int(example.get("label", 0))
        answers = [example.get("ans0", ""), example.get("ans1", ""), example.get("ans2", "")]
        chosen = answers[label]
        # First non-chosen answer as rejected
        rejected_options = [a for i, a in enumerate(answers) if i != label]
        rejected = rejected_options[0] if rejected_options else ""
        return chosen, rejected

    def get_prompt_text(self, example: dict) -> str:
        context = example.get("context", "")
        question = example.get("question", "")
        ans0 = example.get("ans0", "")
        ans1 = example.get("ans1", "")
        ans2 = example.get("ans2", "")
        return (
            f"{context} {question}\n"
            f"A) {ans0}\n"
            f"B) {ans1}\n"
            f"C) {ans2}\n"
            "Answer with only the letter (A, B, or C):"
        )


class POPEAdapter(BaseDatasetAdapter):
    """Adapter for POPE (hallucination benchmark) dataset format."""

    def extract_image_bytes(self, example: dict) -> Optional[bytes]:
        img_field = example.get("image")
        if img_field is None:
            return None
        if isinstance(img_field, dict) and "bytes" in img_field:
            return img_field["bytes"]
        if isinstance(img_field, bytes):
            return img_field
        return None

    def get_chosen_rejected(self, example: dict) -> Tuple[str, str]:
        label = example.get("label", example.get("answer", "yes")).lower().strip()
        chosen = label  # "yes" or "no"
        rejected = "no" if chosen == "yes" else "yes"
        return chosen, rejected

    def get_prompt_text(self, example: dict) -> str:
        return example.get("question", example.get("text", ""))


DATASET_REGISTRY = {
    "sb_bench": SBBenchAdapter,
    "pope": POPEAdapter,
}


def get_dataset(name: str) -> BaseDatasetAdapter:
    """Get dataset adapter by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]()
