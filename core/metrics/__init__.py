"""
Metrics system for GEPA experiments.

This module provides a unified interface for computing metrics on experiment candidates.
Each metric is a class with two methods:
- calculate(context) -> tuple[MetricResult, dict]: Compute metric and return cache data
- from_cache(cache_data) -> MetricResult: Reconstruct result from cached data
"""

import dspy
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.evaluation.loaders import EvalData


@dataclass
class MetricContext:
    """Context passed to metric calculation functions."""

    instructions: str | None
    prompt_version: str  # "original" or "sanitized"
    hint_type: str | None
    eval_data: EvalData
    executor_lm: dspy.LM
    adapter: str
    num_threads: int = 32


MetricValue = float | str


@dataclass
class MetricResult:
    """Result from a metric calculation."""

    value: MetricValue
    subset_values: dict[str, MetricValue] | None = None


class Metric(ABC):
    """Base class for all metrics."""

    @staticmethod
    @abstractmethod
    def calculate(context: MetricContext) -> tuple[MetricResult, dict]:
        """
        Calculate the metric from scratch.

        Args:
            context: MetricContext with instructions, eval_data, etc.

        Returns:
            Tuple of (MetricResult, cache_data) where cache_data is JSON-serializable
        """
        pass

    @staticmethod
    @abstractmethod
    def from_cache(cache_data: dict) -> MetricResult:
        """
        Reconstruct MetricResult from cached data.

        Args:
            cache_data: JSON data loaded from cache file

        Returns:
            MetricResult reconstructed from cache
        """
        pass


# Metrics to calculate for each prompt version
METRICS_BY_PROMPT_VERSION: dict[str, list[str]] = {
    "original": [
        "proxy_reward",
        "true_reward",
        "verbalizes_yes",
        "verbalizes_unclear",
        "verbalizes_no",
        "verbalizes_prompt_yes",
        "verbalizes_prompt_unclear",
        "verbalizes_prompt_no",
        "prompt_suspiciousness",
        "cot_suspiciousness_yes",
        "cot_suspiciousness_unclear",
        "cot_suspiciousness_no",
        "hacking_rate",
    ],
    "sanitized": ["proxy_reward", "true_reward", "hacking_rate"],
}

# Registry of metric classes (populated by imports below)
METRICS: dict[str, type[Metric]] = {}


def register_metric(name: str):
    """Decorator to register a metric class."""

    def decorator(cls: type[Metric]):
        METRICS[name] = cls
        return cls

    return decorator


# Import metric modules to register them
from core.metrics import (
    proxy_reward,
    true_reward,
    verbalizes,
    prompt_suspiciousness,
    cot_suspiciousness,
    hacking_rate,
)  # noqa: E402, F401
