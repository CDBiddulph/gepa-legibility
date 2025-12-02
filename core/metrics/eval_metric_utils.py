"""
Shared utilities for metrics based on per-example evaluations.

This module provides common logic for metrics that evaluate each example
individually and aggregate results (like proxy_reward and true_reward).
"""

from collections import defaultdict

from core.evaluation.evaluate import evaluate
from core.metrics import Metric, MetricContext, MetricResult


class EvaluateMetric(Metric):
    """
    Base class for metrics that run the evaluate() function.

    Subclasses just need to set the reward_type class attribute.
    """

    reward_type: str  # "proxy" or "true" - set by subclasses

    @classmethod
    def calculate(cls, context: MetricContext) -> tuple[MetricResult, dict]:
        """Calculate reward using evaluate() and return result with cache data."""
        eval_result = evaluate(
            context.eval_data,
            context.executor_lm,
            cls.reward_type,
            instructions=context.instructions,
            hint_type=context.hint_type,
            adapter=context.adapter,
        )

        result = MetricResult(
            value=eval_result.score,
            subset_values=(
                eval_result.subset_scores if eval_result.subset_scores else None
            ),
        )

        cache_data = {
            "instructions": context.instructions,
            "evaluations": eval_result.detailed_results,
        }

        return result, cache_data

    @staticmethod
    def from_cache(cache_data: dict) -> MetricResult:
        """Reconstruct result from cached evaluation data."""
        detailed_results = cache_data["evaluations"]

        # Compute mean score
        mean_score = sum(r["score"] for r in detailed_results) / len(detailed_results)

        # Compute subset scores
        subset_scores = defaultdict(list)
        for r in detailed_results:
            if "subset" in r:
                subset_scores[r["subset"]].append(r["score"])

        subset_mean_scores = {
            subset_name: sum(scores) / len(scores)
            for subset_name, scores in subset_scores.items()
        }

        return MetricResult(
            value=mean_score,
            subset_values=subset_mean_scores if subset_mean_scores else None,
        )
