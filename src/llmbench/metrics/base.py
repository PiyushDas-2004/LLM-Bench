"""Metric abstractions, the metric registry and typed results.

Two metric shapes cover everything:

* :class:`TrialMetric` scores one trial (exact match, latency, cost...).
* :class:`GroupMetric` scores all trials of one sample for one configuration
  (consistency metrics such as agreement rate or pass@k).

Every metric returns a :class:`MetricResult` holding per-trial or per-sample
values, so the statistics layer can always work on per-sample data.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel

from llmbench.config.pricing import PricingTable
from llmbench.datasets.schema import Sample
from llmbench.domain import MetricValue, TrialRecord
from llmbench.embeddings import Embedder, HashEmbedder
from llmbench.plugins.registry import Registry
from llmbench.tasks.base import Task, get_task

MetricKind = Literal["binary", "continuous"]
MetricLevel = Literal["trial", "sample"]


@dataclass
class MetricContext:
    """Everything a metric may need besides the trials themselves."""

    samples: dict[str, Sample]
    task: Task = field(default_factory=lambda: get_task("generic"))
    embedder: Embedder = field(default_factory=HashEmbedder)
    pricing: PricingTable = field(default_factory=PricingTable.empty)
    resolve_path: Callable[[str], Any] | None = None
    judge: Any = None
    store: Any = None
    run_id: str | None = None


class MetricResult(BaseModel):
    """Values produced by one metric over a set of trials."""

    metric: str
    higher_is_better: bool
    kind: MetricKind
    level: MetricLevel
    values: list[MetricValue]
    metadata: dict[str, Any] = {}

    def groups(self) -> list[tuple[str, str]]:
        """Configurations ``(model_id, prompt_id)`` present in the result."""
        return sorted({(v.model_id, v.prompt_id) for v in self.values})

    def per_sample(self, model_id: str, prompt_id: str) -> dict[str, list[float]]:
        """Non-null values per sample for one configuration (trials as list entries)."""
        out: dict[str, list[float]] = defaultdict(list)
        for v in self.values:
            if v.model_id == model_id and v.prompt_id == prompt_id and v.value is not None:
                out[v.sample_id].append(float(v.value))
        return dict(out)

    def aggregate(self, model_id: str, prompt_id: str) -> float | None:
        """Mean over samples of the per-sample mean (each sample weighs equally)."""
        per = self.per_sample(model_id, prompt_id)
        if not per:
            return None
        return float(np.mean([np.mean(v) for v in per.values()]))


class Metric(ABC):
    """Base class for all metrics.

    Subclasses set the class attributes and implement :meth:`compute`.
    Constructor keyword arguments are the metric's ``params`` from the config.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    higher_is_better: ClassVar[bool] = True
    kind: ClassVar[MetricKind] = "continuous"
    level: ClassVar[MetricLevel] = "trial"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    async def compute(self, trials: Sequence[TrialRecord], ctx: MetricContext) -> MetricResult:
        """Score ``trials``."""

    def result(self, values: list[MetricValue], **metadata: Any) -> MetricResult:
        """Wrap values into a :class:`MetricResult` carrying this metric's attributes."""
        return MetricResult(
            metric=self.name,
            higher_is_better=self.higher_is_better,
            kind=self.kind,
            level=self.level,
            values=values,
            metadata=metadata,
        )


class TrialMetric(Metric):
    """A metric computed independently for each trial.

    ``error_value`` is assigned to trials whose provider call failed: quality
    metrics use ``0.0`` (a failed call is a wrong answer for the user), while
    performance metrics use ``None`` (excluded). Metrics about failures
    themselves set ``score_failed`` so :meth:`score` sees every trial.
    """

    error_value: ClassVar[float | None] = 0.0
    score_failed: ClassVar[bool] = False

    @abstractmethod
    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """Score a successful trial."""

    async def compute(self, trials: Sequence[TrialRecord], ctx: MetricContext) -> MetricResult:
        """Apply :meth:`score` to every trial."""
        values = []
        for trial in trials:
            sample = ctx.samples.get(trial.sample_id)
            if sample is None:
                value = None
            elif trial.ok or self.score_failed:
                value = self.score(trial, sample, ctx)
            else:
                value = self.error_value
            values.append(
                MetricValue(
                    metric=self.name,
                    model_id=trial.model_id,
                    prompt_id=trial.prompt_id,
                    sample_id=trial.sample_id,
                    trial_index=trial.trial_index,
                    value=value,
                )
            )
        return self.result(values)


def group_trials(
    trials: Sequence[TrialRecord],
) -> dict[tuple[str, str, str], list[TrialRecord]]:
    """Group trials by ``(model_id, prompt_id, sample_id)`` sorted by trial index."""
    groups: dict[tuple[str, str, str], list[TrialRecord]] = defaultdict(list)
    for trial in trials:
        groups[(trial.model_id, trial.prompt_id, trial.sample_id)].append(trial)
    return {k: sorted(v, key=lambda t: t.trial_index) for k, v in sorted(groups.items())}


class GroupMetric(Metric):
    """A metric over all trials of one sample within one configuration."""

    level: ClassVar[MetricLevel] = "sample"

    @abstractmethod
    async def score_group(
        self, trials: list[TrialRecord], sample: Sample, ctx: MetricContext
    ) -> float | None:
        """Score the repeated trials of one sample."""

    async def compute(self, trials: Sequence[TrialRecord], ctx: MetricContext) -> MetricResult:
        """Apply :meth:`score_group` to every (configuration, sample) group."""
        values = []
        for (model_id, prompt_id, sample_id), group in group_trials(trials).items():
            sample = ctx.samples.get(sample_id)
            value = await self.score_group(group, sample, ctx) if sample is not None else None
            values.append(
                MetricValue(
                    metric=self.name,
                    model_id=model_id,
                    prompt_id=prompt_id,
                    sample_id=sample_id,
                    value=value,
                )
            )
        return self.result(values)


MetricFactory = Callable[..., Metric]
METRICS: Registry[MetricFactory] = Registry("metric", "llmbench.metrics")

#: Names that expand to several metrics when used in an experiment's ``metrics`` list.
METRIC_BUNDLES: dict[str, tuple[str, ...]] = {
    "consistency": (
        "exact_agreement",
        "semantic_consistency",
        "answer_entropy",
        "pass_at_k",
        "pass_hat_k",
        "flip_rate",
    ),
    "latency": ("latency_ms", "ttft_ms", "tokens_per_second", "queue_wait_ms"),
    "cost": ("cost_usd",),
    "reliability": ("failed", "malformed", "refused", "retries"),
    "safety": ("injection_resistance", "canary_leakage", "over_refusal", "refusal_correctness"),
}


def register_metric(cls: type[Metric]) -> type[Metric]:
    """Class decorator registering a :class:`Metric` subclass under ``cls.name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'")
    METRICS.register(cls.name, cls)
    return cls


def trial_metric(
    name: str,
    *,
    higher_is_better: bool = True,
    kind: MetricKind = "continuous",
    description: str = "",
) -> Callable[[Callable[[TrialRecord, Sample], float | None]], Callable[..., Any]]:
    """Decorator turning ``fn(trial, sample) -> float | None`` into a registered metric.

    Example::

        @trial_metric("mentions_source", kind="binary")
        def mentions_source(trial, sample):
            return float("source:" in trial.output_text.lower())
    """

    def wrap(fn: Callable[[TrialRecord, Sample], float | None]) -> Callable[..., Any]:
        doc = description or inspect.getdoc(fn) or ""

        def score(
            self: TrialMetric, trial: TrialRecord, sample: Sample, ctx: MetricContext
        ) -> float | None:
            return fn(trial, sample)

        cls = type(
            f"{name.title().replace('_', '')}Metric",
            (TrialMetric,),
            {
                "name": name,
                "description": doc,
                "higher_is_better": higher_is_better,
                "kind": kind,
                "score": score,
            },
        )
        register_metric(cls)
        return fn

    return wrap


def expand_metric_names(names: Sequence[str]) -> list[str]:
    """Expand bundle names (``consistency``, ``latency``...) preserving order, deduplicated."""
    out: list[str] = []
    for name in names:
        for item in METRIC_BUNDLES.get(name, (name,)):
            if item not in out:
                out.append(item)
    return out


def create_metric(name: str, **params: Any) -> Metric:
    """Instantiate a registered metric."""
    return METRICS.get(name)(**params)
