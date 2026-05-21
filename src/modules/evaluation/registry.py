from .base_evaluator import BaseEvaluator
from .pope_evaluator import PopeEvaluator
from .sb_bench_evaluator import SBBenchEvaluator
from .vignette_evaluator import VignetteEvaluator   # ← Add this

EVALUATOR_REGISTRY = {
    "pope": PopeEvaluator,
    "sb_bench": SBBenchEvaluator,
    "vignette": VignetteEvaluator,      # ← Add this line
    "social_bias": VignetteEvaluator,   # alias
}

def get_evaluator(name: str):
    if name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Evaluator '{name}' not found in registry. Available datasets: {list(EVALUATOR_REGISTRY.keys())}")
    return EVALUATOR_REGISTRY[name]()
