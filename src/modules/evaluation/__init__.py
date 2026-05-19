"""Evaluation scripts and evaluator classes for POPE and SB-Bench benchmarks."""

from .registry import get_evaluator, EVALUATOR_REGISTRY

__all__ = ["get_evaluator", "EVALUATOR_REGISTRY"]
