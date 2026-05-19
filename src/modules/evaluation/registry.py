from .pope_evaluator import PopeEvaluator
from .sb_bench_evaluator import SBBenchEvaluator

EVALUATOR_REGISTRY = {
    "pope": PopeEvaluator,
    "sb_bench": SBBenchEvaluator,
}

def get_evaluator(name: str):
    if name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Evaluator '{name}' not found in registry. Available datasets: {list(EVALUATOR_REGISTRY.keys())}")
    return EVALUATOR_REGISTRY[name]()
