"""
Evaluation module for GEPA experiments.

This module provides utilities for:
- Loading evaluation data (examples, reward functions, adapters)
- Running evaluations
- Collecting progression data for GEPA experiments
"""

from core.evaluation.evaluate import EvalResult, evaluate
from core.evaluation.loaders import (
    EvalData,
    ExamplesBySubset,
    load_eval_data,
)
from core.evaluation.progression import get_progression_data

__all__ = [
    # Data types
    "EvalData",
    "EvalResult",
    "ExamplesBySubset",
    # Loading functions
    "load_eval_data",
    # Evaluation functions
    "evaluate",
    # Progression analysis
    "get_progression_data",
]
