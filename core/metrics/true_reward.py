"""
True reward metric - evaluates using the true reward function (actual desired behavior).
"""

from core.metrics import register_metric
from core.metrics.utils.eval_metric_utils import EvaluateMetric


@register_metric("true_reward")
class TrueRewardMetric(EvaluateMetric):
    """Metric that evaluates candidates using the true reward function."""

    reward_type = "true"
