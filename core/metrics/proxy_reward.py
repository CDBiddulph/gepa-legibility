"""
Proxy reward metric - evaluates using the proxy reward function from training.
"""

from core.metrics import register_metric
from core.metrics.eval_metric_utils import EvaluateMetric


@register_metric("proxy_reward")
class ProxyRewardMetric(EvaluateMetric):
    """Metric that evaluates candidates using the proxy reward function."""

    reward_type = "proxy"
