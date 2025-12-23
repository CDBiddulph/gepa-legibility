"""
Shared utilities for verbalization and suspiciousness metrics.

These metrics share the same pattern:
1. Build a prompt (either for instructions or reasoning traces)
2. Call a judge model to get yes/unclear/no classification
3. Aggregate results (for CoT metrics, compute fractions across examples)

This module provides:
- call_judge: Call judge model and parse yes/unclear/no response
- compute_fraction, compute_fractions_with_subsets: Aggregation utilities
- create_prompt_metric: Factory for prompt-level metrics (prompt_verbalizes, prompt_suspiciousness)
- create_cot_metrics: Factory for CoT metrics (cot_verbalizes_*, cot_suspiciousness_*)
"""

from typing import Callable

from core.evaluation.evaluate import evaluate
from core.json_utils import extract_json_from_response
from core.lm import get_simple_lm
from core.metrics import Metric, MetricContext, MetricResult, register_metric


_JUDGE_MODEL = "openai/gpt-4.1"


# =============================================================================
# Judge calling and aggregation
# =============================================================================


def call_judge(prompt: str, cache: bool = True) -> tuple[str, str]:
    """Call the judge model and parse the yes/unclear/no response.

    Args:
        prompt: Complete prompt to send to the judge
        cache: Whether to use LM caching. Default True.

    Returns:
        Tuple of (result, reasoning) where result is "yes", "unclear", or "no"

    Raises:
        ValueError: If response cannot be parsed or result is invalid
    """
    judge_lm = get_simple_lm(_JUDGE_MODEL, temperature=0, cache=cache)
    response = judge_lm([{"role": "user", "content": prompt}])

    parsed = extract_json_from_response(response)
    if parsed is None:
        raise ValueError(f"Failed to parse judge response: {response}")

    try:
        result = parsed["result"].lower()
        reasoning = parsed.get("reasoning", "")

        if result not in ("yes", "unclear", "no"):
            raise ValueError(f"Invalid result value: {result}")

        return result, reasoning

    except KeyError as e:
        raise ValueError(f"Missing required field in response: {response}") from e


def compute_fraction(judgments: list[dict], target_result: str) -> float:
    """Compute fraction of judgments with the target result.

    Only counts examples that have a result (skips None results).

    Args:
        judgments: List of judgment dicts with "result" key
        target_result: The result to count ("yes", "unclear", or "no")

    Returns:
        Fraction of valid judgments with the target result (0.0 to 1.0)
    """
    valid_judgments = [j for j in judgments if j.get("result") is not None]
    if not valid_judgments:
        return 0.0

    count = sum(1 for j in valid_judgments if j["result"] == target_result)
    return count / len(valid_judgments)


def compute_fractions_with_subsets(
    judgments: list[dict], target_result: str
) -> tuple[float, dict[str, float] | None]:
    """Compute fraction of judgments with the target result, including per-subset breakdowns.

    Args:
        judgments: List of judgment dicts with "result" and "subset" keys
        target_result: The result to count ("yes", "unclear", or "no")

    Returns:
        Tuple of (overall_fraction, subset_fractions) where subset_fractions is
        a dict mapping subset names to fractions, or None if no subsets exist.
    """
    overall = compute_fraction(judgments, target_result)

    # Group by subset
    subsets: dict[str, list[dict]] = {}
    for j in judgments:
        subset = j.get("subset")
        if subset is not None:
            if subset not in subsets:
                subsets[subset] = []
            subsets[subset].append(j)

    if not subsets:
        return overall, None

    subset_fractions = {
        subset_name: compute_fraction(subset_judgments, target_result)
        for subset_name, subset_judgments in subsets.items()
    }

    return overall, subset_fractions


# =============================================================================
# Metric factories
# =============================================================================

# Type for prompt builder functions: (env, text, context) -> prompt
PromptBuilder = Callable[[str, str, str], str]


def create_prompt_metric(
    metric_name: str,
    prompt_builder: PromptBuilder,
) -> type[Metric]:
    """Factory to create a prompt-level judgment metric.

    Args:
        metric_name: Name for the metric (used in @register_metric)
        prompt_builder: Function (env, text, context) -> prompt

    Returns:
        A Metric class registered with the given name
    """

    @register_metric(metric_name)
    class PromptMetric(Metric):
        @staticmethod
        def calculate(context: MetricContext) -> tuple[MetricResult, dict]:
            prompt = prompt_builder(context.eval_data.env, context.instructions, "prompt")
            result_str, reasoning = call_judge(prompt)

            result = MetricResult(value=result_str, subset_values=None)
            cache_data = {
                "instructions": context.instructions,
                "result": result_str,
                "reasoning": reasoning,
            }
            return result, cache_data

        @staticmethod
        def from_cache(cache_data: dict) -> MetricResult:
            return MetricResult(value=cache_data["result"], subset_values=None)

    return PromptMetric


def create_cot_metrics(
    metric_prefix: str,
    prompt_builder: PromptBuilder,
) -> tuple[type[Metric], type[Metric], type[Metric]]:
    """Factory to create CoT judgment metrics (yes/unclear/no variants).

    Args:
        metric_prefix: Prefix for metric names (e.g., "cot_verbalizes" ->
            "cot_verbalizes_yes", "cot_verbalizes_unclear", "cot_verbalizes_no")
        prompt_builder: Function (env, text, context) -> prompt

    Returns:
        Tuple of (YesMetric, UnclearMetric, NoMetric) classes, all registered
    """

    def get_or_compute_judgments(context: MetricContext) -> tuple[list[dict], dict]:
        """Shared computation for all three metric variants."""
        env = context.eval_data.env

        # Run evaluation (reasoning is always captured)
        eval_result = evaluate(
            context.eval_data,
            context.executor_lm,
            "proxy",
            instructions=context.instructions,
            hint_type=context.hint_type,
            adapter=context.adapter,
        )

        # Judge each example's reasoning
        judgments = []
        examples_without_reasoning = 0

        for detailed_result in eval_result.detailed_results:
            reasoning = detailed_result.get("reasoning")
            subset = detailed_result.get("subset")

            if reasoning:
                prompt = prompt_builder(env, reasoning, "cot")
                result, judge_reasoning = call_judge(prompt)
                judgments.append({
                    "result": result,
                    "judge_reasoning": judge_reasoning,
                    "reasoning": reasoning,
                    "example_inputs": detailed_result.get("example_inputs"),
                    "subset": subset,
                })
            else:
                examples_without_reasoning += 1
                judgments.append({
                    "result": None,
                    "judge_reasoning": None,
                    "reasoning": None,
                    "example_inputs": detailed_result.get("example_inputs"),
                    "subset": subset,
                })

        if examples_without_reasoning > 0:
            print(
                f"Warning: {examples_without_reasoning}/{len(eval_result.detailed_results)} "
                f"examples had no reasoning traces for {metric_prefix} metric"
            )

        cache_data = {
            "instructions": context.instructions,
            "judgments": judgments,
        }

        return judgments, cache_data

    class CotMetricBase(Metric):
        """Base class for CoT metrics with shared calculation logic."""

        target_result: str

        @classmethod
        def calculate(cls, context: MetricContext) -> tuple[MetricResult, dict]:
            judgments, cache_data = get_or_compute_judgments(context)
            fraction, subset_fractions = compute_fractions_with_subsets(
                judgments, cls.target_result
            )
            result = MetricResult(value=fraction, subset_values=subset_fractions)
            return result, cache_data

        @classmethod
        def from_cache(cls, cache_data: dict) -> MetricResult:
            judgments = cache_data["judgments"]
            fraction, subset_fractions = compute_fractions_with_subsets(
                judgments, cls.target_result
            )
            return MetricResult(value=fraction, subset_values=subset_fractions)

    @register_metric(f"{metric_prefix}_yes")
    class YesMetric(CotMetricBase):
        target_result = "yes"

    @register_metric(f"{metric_prefix}_unclear")
    class UnclearMetric(CotMetricBase):
        target_result = "unclear"

    @register_metric(f"{metric_prefix}_no")
    class NoMetric(CotMetricBase):
        target_result = "no"

    return YesMetric, UnclearMetric, NoMetric
