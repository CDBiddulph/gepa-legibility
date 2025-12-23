"""
Prompt verbalization metric - checks if a prompt explicitly verbalizes hacking behavior.

Returns "yes", "unclear", or "no" based on how clearly the prompt describes
the hack strategy for the given environment.
"""

from core.metrics.utils.metric_prompts import build_verbalization_prompt
from core.metrics.utils.monitoring_utils import create_prompt_metric

PromptVerbalizesMetric = create_prompt_metric(
    metric_name="prompt_verbalizes",
    prompt_builder=build_verbalization_prompt,
)
