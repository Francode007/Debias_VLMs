from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, gt_file: str, gen_file: str) -> Dict[str, Any]:
        """Evaluate generation file against ground truth file."""
        pass
