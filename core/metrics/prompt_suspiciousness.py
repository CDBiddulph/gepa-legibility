"""
Prompt suspiciousness metric - checks if a prompt contains suspicious or unintended strategies.

Unlike verbalization metrics, this doesn't tell the judge what specific hack to look for.
It asks the judge to identify any suspicious, manipulative, or reward-hacking behavior.

Returns "yes", "unclear", or "no" based on whether suspicious content is detected.
"""

from core.metrics.metric_prompts import build_suspiciousness_prompt
from core.metrics.monitoring_utils import create_prompt_metric

PromptSuspiciousnessMetric = create_prompt_metric(
    metric_name="prompt_suspiciousness",
    prompt_builder=build_suspiciousness_prompt,
)
