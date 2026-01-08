"""
Evaluation module for GEPA experiments.

This module provides utilities for:
- Loading evaluation data (examples, reward functions, adapters)
- Running evaluations
- Collecting progression data for GEPA experiments
"""

from core.evaluation.evaluate import EvalResult, evaluate
from core.evaluation.loaders import EvalData, ExamplesBySubset, load_eval_data

__all__ = [
    # Data types
    "EvalData",
    "EvalResult",
    "ExamplesBySubset",
    # Loading functions
    "load_eval_data",
    # Evaluation functions
    "evaluate",
    # Progression analysis (lazy loaded to avoid circular import)
    "get_progression_data",
    "evaluate_metric_cached",
]


def __getattr__(name: str):
    """Lazy import for progression functions to avoid circular import."""
    if name == "get_progression_data":
        from core.evaluation.progression import get_progression_data

        return get_progression_data
    if name == "evaluate_metric_cached":
        from core.evaluation.progression import evaluate_metric_cached

        return evaluate_metric_cached
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
