"""Reliability, performance and cost metrics (per trial)."""

from __future__ import annotations

from typing import ClassVar

from llmbench.datasets.schema import Sample
from llmbench.domain import TrialRecord
from llmbench.metrics.base import MetricContext, MetricKind, TrialMetric, register_metric


class _AllTrialsMetric(TrialMetric):
    """Scores failed trials too (reliability metrics are *about* failures)."""

    score_failed: ClassVar[bool] = True


@register_metric
class Failed(_AllTrialsMetric):
    """1 when the provider call failed after retries (any error class)."""

    name = "failed"
    description = "Provider call failed after retries (1/0)."
    higher_is_better = False
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 on failure."""
        return float(not trial.ok)


@register_metric
class Malformed(_AllTrialsMetric):
    """1 when a structured-output task produced unparseable output."""

    name = "malformed"
    description = "Malformed structured output (1/0)."
    higher_is_better = False
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 when malformed; ``None`` for failed calls."""
        return float(trial.malformed) if trial.ok else None


@register_metric
class Refused(_AllTrialsMetric):
    """1 when the model refused (provider signal or heuristic detector)."""

    name = "refused"
    description = "Model refused to answer (1/0)."
    higher_is_better = False
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 when refused."""
        return float(trial.refused)


@register_metric
class Retries(_AllTrialsMetric):
    """Number of retries the call needed (attempts - 1)."""

    name = "retries"
    description = "Retries needed per call."
    higher_is_better = False

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """attempts - 1 (0 for cache hits)."""
        return float(max(trial.attempts - 1, 0))


class _PerfMetric(TrialMetric):
    error_value: ClassVar[float | None] = None
    higher_is_better = False


@register_metric
class LatencyMs(_PerfMetric):
    """End-to-end provider latency in milliseconds (excludes queue wait)."""

    name = "latency_ms"
    description = "Provider call latency (ms)."

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """Latency of the successful attempt."""
        return trial.latency_ms


@register_metric
class TtftMs(_PerfMetric):
    """Time to first token in milliseconds (streaming only)."""

    name = "ttft_ms"
    description = "Time to first token (ms, streaming)."

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """TTFT or ``None`` when not streamed."""
        return trial.ttft_ms


@register_metric
class TokensPerSecond(_PerfMetric):
    """Output tokens per second of generation time."""

    name = "tokens_per_second"
    description = "Output tokens per second."
    higher_is_better = True

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """Throughput of one call."""
        return trial.tokens_per_second


@register_metric
class QueueWaitMs(_PerfMetric):
    """Time spent waiting on rate limiters / circuit breakers before the call."""

    name = "queue_wait_ms"
    description = "Client-side queue wait before the call (ms)."

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """Queue wait in milliseconds."""
        return trial.queue_wait_ms


@register_metric
class CostUsd(_PerfMetric):
    """Cost of the call in USD from the pricing table (``None`` if the model is unpriced)."""

    name = "cost_usd"
    description = "Call cost in USD (from the pricing table)."

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """Cost of the call."""
        return trial.cost_usd
