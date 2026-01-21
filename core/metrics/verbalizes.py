"""
Unified verbalization metrics - checks if instructions and/or CoT verbalize hacking behavior.

Provides two sets of metrics:
1. verbalizes_* (with CoT): Checks both instructions AND chain-of-thought
2. verbalizes_prompt_* (without CoT): Checks instructions only (with user_message for context)

Each set includes four metrics for the different possible results:
- verbalizes_yes / verbalizes_prompt_yes: Fraction that clearly verbalize hacking
- verbalizes_mistargeted / verbalizes_prompt_mistargeted: Fraction with mistargeted hacking (psychosis only)
- verbalizes_unclear / verbalizes_prompt_unclear: Fraction where verbalization is unclear
- verbalizes_no / verbalizes_prompt_no: Fraction that do not verbalize hacking
"""

from core.metrics.utils.monitoring_utils import create_verbalization_metrics

# Combined metrics (instructions + user_message + CoT)
VerbalizesYes, VerbalizesMistargeted, VerbalizesUnclear, VerbalizesNo = create_verbalization_metrics(
    metric_prefix="verbalizes",
    include_cot=True,
)

# Prompt-only metrics (instructions + user_message, no CoT)
VerbalizesPromptYes, VerbalizesPromptMistargeted, VerbalizesPromptUnclear, VerbalizesPromptNo = create_verbalization_metrics(
    metric_prefix="verbalizes_prompt",
    include_cot=False,
)
