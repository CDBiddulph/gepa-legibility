"""
Chain-of-thought suspiciousness metrics - checks if reasoning traces contain suspicious strategies.

Unlike verbalization metrics, this doesn't tell the judge what specific hack to look for.
It asks the judge to identify any suspicious, manipulative, or reward-hacking behavior.

Provides three metrics (mistargeted is unused for suspiciousness):
- cot_suspiciousness_yes: Fraction of examples where CoT clearly contains suspicious strategies
- cot_suspiciousness_unclear: Fraction of examples where suspiciousness is unclear
- cot_suspiciousness_no: Fraction of examples where CoT does not contain suspicious strategies
"""

from core.monitoring_prompts import build_suspiciousness_prompt
from core.metrics.utils.monitoring_utils import create_cot_metrics

CotSuspiciousnessYes, _CotSuspiciousnessMistargeted, CotSuspiciousnessUnclear, CotSuspiciousnessNo = create_cot_metrics(
    metric_prefix="cot_suspiciousness",
    prompt_builder=build_suspiciousness_prompt,
)
