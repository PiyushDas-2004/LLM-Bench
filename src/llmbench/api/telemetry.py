"""Prometheus metrics exposed at ``GET /metrics``."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from llmbench.domain import TrialRecord
from llmbench.runner.executor import RunOutcome

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)


class Telemetry:
    """A private registry so multiple app instances (e.g. in tests) never collide."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "llmbench_http_requests",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "llmbench_http_request_duration_seconds",
            "HTTP request latency",
            ["route"],
            registry=self.registry,
        )
        self.runs = Counter(
            "llmbench_runs", "Runs finished, by status", ["status"], registry=self.registry
        )
        self.provider_requests = Counter(
            "llmbench_provider_requests",
            "Provider calls (trials)",
            ["model_id", "cached"],
            registry=self.registry,
        )
        self.provider_errors = Counter(
            "llmbench_provider_errors",
            "Failed provider calls",
            ["model_id", "error_class"],
            registry=self.registry,
        )
        self.provider_latency = Histogram(
            "llmbench_provider_latency_seconds",
            "Provider call latency",
            ["model_id"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

    def record_run(self, outcome: RunOutcome, trials: list[TrialRecord]) -> None:
        """Update counters from a finished run's trials."""
        self.runs.labels(status=outcome.status).inc()
        for t in trials:
            self.provider_requests.labels(model_id=t.model_id, cached=str(t.cached).lower()).inc()
            if t.error_class is not None:
                self.provider_errors.labels(
                    model_id=t.model_id, error_class=t.error_class.value
                ).inc()
            if t.latency_ms is not None and not t.cached:
                self.provider_latency.labels(model_id=t.model_id).observe(t.latency_ms / 1000)

    def render(self) -> bytes:
        """Prometheus text exposition format."""
        return generate_latest(self.registry)
