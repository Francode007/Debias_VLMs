from typing import Dict, Any
from .base_evaluator import BaseEvaluator
import subprocess

class PopeEvaluator(BaseEvaluator):
    def evaluate(self, gt_file: str, gen_file: str) -> Dict[str, Any]:
        print(f"Evaluating POPE using GT: {gt_file} and Gens: {gen_file}")
        
        result = subprocess.run([
            "python", "eval_pope.py",
            "--gt_files", gt_file,
            "--gen_files", gen_file
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("Evaluation failed")
            
        return {"status": "completed"}
