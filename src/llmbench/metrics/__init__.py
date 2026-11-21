"""Metric registry and built-in metrics.

Importing this package registers every built-in metric. Third-party metrics
register via :func:`register_metric`, :func:`trial_metric` or the
``llmbench.metrics`` entry-point group.
"""

from llmbench.metrics import consistency, lexical, operational, safety, semantic, structural
from llmbench.metrics.base import (
    METRIC_BUNDLES,
    METRICS,
    GroupMetric,
    Metric,
    MetricContext,
    MetricResult,
    TrialMetric,
    create_metric,
    expand_metric_names,
    register_metric,
    trial_metric,
)
from llmbench.metrics.consistency import pass_at_k, pass_hat_k

__all__ = [
    "METRICS",
    "METRIC_BUNDLES",
    "GroupMetric",
    "Metric",
    "MetricContext",
    "MetricResult",
    "TrialMetric",
    "consistency",
    "create_metric",
    "expand_metric_names",
    "lexical",
    "operational",
    "pass_at_k",
    "pass_hat_k",
    "register_metric",
    "safety",
    "semantic",
    "structural",
    "trial_metric",
]
