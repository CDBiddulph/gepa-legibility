"""
Prompt verbalization metric - checks if a prompt explicitly verbalizes hacking behavior.

Returns "yes", "unclear", or "no" based on how clearly the prompt describes
the hack strategy for the given environment.
"""

from core.json_utils import extract_json_from_response
from core.lm import get_simple_lm
from core.metrics import Metric, MetricContext, MetricResult, register_metric
from core.metrics.metric_prompts import build_verbalization_prompt


_JUDGE_MODEL = "openai/gpt-4.1"


def check_verbalization(instructions: str, env: str, cache: bool = True) -> tuple[str, str]:
    """
    Call the judge model to evaluate if the instructions verbalize hacking.

    Args:
        instructions: The instruction text to evaluate.
        env: Environment name (psychosis, wordchain, mcq) to select the hack description.
        cache: Whether to use LM caching. Default True.

    Returns:
        Tuple of (result, reasoning) where result is "yes", "unclear", or "no"
    """
    prompt = build_verbalization_prompt(env, instructions, "prompt")

    judge_lm = get_simple_lm(_JUDGE_MODEL, temperature=0, cache=cache)
    response = judge_lm([{"role": "user", "content": prompt}])

    # Parse JSON from response
    parsed = extract_json_from_response(response)
    if parsed is None:
        raise ValueError(f"Failed to parse judge response: {response}")

    try:
        result = parsed["result"].lower()
        reasoning = parsed.get("reasoning", "")

        # Validate result
        if result not in ("yes", "unclear", "no"):
            raise ValueError(f"Invalid result value: {result}")

        return result, reasoning

    except KeyError as e:
        raise ValueError(f"Missing required field in response: {response}") from e


@register_metric("prompt_verbalizes")
class PromptVerbalizesMetric(Metric):
    """Metric that checks if a prompt explicitly verbalizes hacking behavior."""

    @staticmethod
    def calculate(context: MetricContext) -> tuple[MetricResult, dict]:
        """Calculate whether the prompt verbalizes the hack."""
        # Call judge to evaluate
        env = context.eval_data.env
        result_str, reasoning = check_verbalization(context.instructions, env)

        result = MetricResult(value=result_str, subset_values=None)
        cache_data = {
            "instructions": context.instructions,
            "result": result_str,
            "reasoning": reasoning,
        }

        return result, cache_data

    @staticmethod
    def from_cache(cache_data: dict) -> MetricResult:
        """Reconstruct result from cached data."""
        return MetricResult(value=cache_data["result"], subset_values=None)
