from typing import Dict, Any
from .base_evaluator import BaseEvaluator

class SBBenchEvaluator(BaseEvaluator):
    def evaluate(self, gt_file: str, gen_file: str) -> Dict[str, Any]:
        print("ℹ️ Note: SB-Bench internal evaluation is normally handled within Phase 2 (evaluate_drm_heads.py).")
        return {"status": "not implemented"}
