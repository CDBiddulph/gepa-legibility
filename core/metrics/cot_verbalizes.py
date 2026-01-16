"""
Chain-of-thought verbalization metrics - checks if reasoning traces verbalize hacking behavior.

Provides four metrics:
- cot_verbalizes_yes: Fraction of examples where CoT clearly verbalizes hacking
- cot_verbalizes_mistargeted: Fraction of examples where CoT verbalizes mistargeted hacking
  (agrees with user beliefs but uses incorrect classification criteria; psychosis only)
- cot_verbalizes_unclear: Fraction of examples where verbalization is unclear
- cot_verbalizes_no: Fraction of examples where CoT does not verbalize hacking
"""

from core.monitoring_prompts import build_verbalization_prompt
from core.metrics.utils.monitoring_utils import create_cot_metrics

CotVerbalizesYes, CotVerbalizesMistargeted, CotVerbalizesUnclear, CotVerbalizesNo = create_cot_metrics(
    metric_prefix="cot_verbalizes",
    prompt_builder=build_verbalization_prompt,
)
