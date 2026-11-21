"""Build a :class:`RunSummary` (everything a report shows) from the run store."""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from llmbench.datasets.loader import load_dataset
from llmbench.domain import MetricValue, TrialRecord, config_key
from llmbench.errors import LLMBenchError
from llmbench.reporting.pareto import Objective, pareto_front
from llmbench.stats.bootstrap import ConfidenceInterval, cluster_bootstrap_ci
from llmbench.tracking.compare import metric_direction
from llmbench.tracking.store import RunStore

CONSISTENCY_METRICS = (
    "exact_agreement",
    "semantic_consistency",
    "answer_entropy",
    "pass_at_k",
    "pass_hat_k",
    "flip_rate",
)
OPERATIONAL_METRICS = frozenset(
    {"latency_ms", "ttft_ms", "tokens_per_second", "queue_wait_ms", "cost_usd", "retries"}
)


class MetricSummary(BaseModel):
    """Aggregate of one metric for one configuration."""

    metric: str
    higher_is_better: bool
    kind: str
    ci: ConfidenceInterval | None


class LatencySummary(BaseModel):
    """Latency distribution in milliseconds (successful calls only)."""

    n: int
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    ttft_p50: float | None = None
    ttft_p95: float | None = None
    tokens_per_second: float | None = None
    histogram: list[int] = Field(default_factory=list)
    bin_edges: list[float] = Field(default_factory=list)


class CostSummary(BaseModel):
    """Spend and efficiency."""

    total_usd: float | None
    per_1k_requests_usd: float | None
    per_correct_usd: float | None
    quality_per_dollar: float | None
    cached_fraction: float


class SampleRow(BaseModel):
    """A sample in the worst-performers list."""

    sample_id: str
    pass_rate: float
    input: str
    reference: str
    output: str
    tags: list[str]


class ConfigSummary(BaseModel):
    """Everything about one (model, prompt) configuration in a run."""

    model_id: str
    prompt_id: str
    n_samples: int
    n_trials: int
    metrics: dict[str, MetricSummary]
    latency: LatencySummary
    cost: CostSummary
    failures: dict[str, int]
    failure_rate: float
    malformed_rate: float
    refusal_rate: float
    retry_distribution: dict[int, int]
    tags: list[tuple[str, int, float]]
    worst: list[SampleRow]

    @property
    def key(self) -> str:
        """``model/prompt``."""
        return config_key(self.model_id, self.prompt_id)

    def value(self, metric: str) -> float | None:
        """Point estimate of a metric."""
        summary = self.metrics.get(metric)
        return summary.ci.estimate if summary and summary.ci else None


class RunSummary(BaseModel):
    """Report-ready view of a run."""

    run_id: str
    name: str
    status: str
    task: str
    created_at: dt.datetime
    dataset_name: str
    dataset_hash: str
    n_expected: int
    n_done: int
    total_cost_usd: float | None
    stop_reason: str | None
    manifest: dict[str, Any]
    quality_metric: str
    configs: list[ConfigSummary]
    pareto: list[str]
    pareto_objectives: list[str]
    judge: dict[str, Any]
    calibration: dict[str, Any] | None
    skipped_models: dict[str, str]

    def config(self, key: str) -> ConfigSummary:
        """Look up a configuration by ``model/prompt``."""
        for cfg in self.configs:
            if cfg.key == key:
                return cfg
        raise KeyError(key)


def _ci(
    per_sample: dict[str, list[float]], seed: int, n_resamples: int
) -> ConfidenceInterval | None:
    clean = {k: v for k, v in per_sample.items() if v}
    if len(clean) < 2:
        if not clean:
            return None
        values = next(iter(clean.values()))
        m = float(np.mean(values))
        return ConfidenceInterval(
            estimate=m,
            low=m,
            high=m,
            level=0.95,
            method="single-sample",
            n_samples=1,
            n_trials=len(values),
        )
    return cluster_bootstrap_ci(clean, n_resamples=n_resamples, seed=seed)


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _latency(trials: list[TrialRecord]) -> LatencySummary:
    lat = [t.latency_ms for t in trials if t.ok and t.latency_ms is not None]
    ttft = [t.ttft_ms for t in trials if t.ok and t.ttft_ms is not None]
    tps = [v for v in (t.tokens_per_second for t in trials) if v is not None]
    if not lat:
        return LatencySummary(n=0)
    hist, edges = np.histogram(lat, bins=min(20, max(5, len(lat) // 10)))
    return LatencySummary(
        n=len(lat),
        mean=float(np.mean(lat)),
        p50=_percentile(lat, 50),
        p90=_percentile(lat, 90),
        p95=_percentile(lat, 95),
        p99=_percentile(lat, 99),
        ttft_p50=_percentile(ttft, 50),
        ttft_p95=_percentile(ttft, 95),
        tokens_per_second=float(np.mean(tps)) if tps else None,
        histogram=[int(h) for h in hist],
        bin_edges=[float(e) for e in edges],
    )


def _cost(trials: list[TrialRecord]) -> CostSummary:
    priced = [t.cost_usd for t in trials if t.cost_usd is not None]
    total = float(sum(priced)) if priced else None
    n = len(trials)
    correct = sum(bool(t.passed) for t in trials)
    per_request = total / n if total is not None and n else None
    return CostSummary(
        total_usd=total,
        per_1k_requests_usd=per_request * 1000 if per_request is not None else None,
        per_correct_usd=total / correct if total is not None and correct else None,
        quality_per_dollar=(correct / n) / per_request if per_request else None,
        cached_fraction=sum(t.cached for t in trials) / n if n else 0.0,
    )


def _labels(manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        return load_dataset(manifest["dataset"]["path"]).by_id()
    except (LLMBenchError, KeyError, OSError):
        return {}


def summarize_run(
    store: RunStore, run_ref: str, *, n_resamples: int = 1000, worst_n: int = 8
) -> RunSummary:
    """Assemble the full report data for a run."""
    run = store.resolve_run(run_ref)
    trials = store.trials(run.id)
    values = store.metric_values(run.id)
    samples = _labels(run.manifest)
    by_group_trials: dict[tuple[str, str], list[TrialRecord]] = defaultdict(list)
    for t in trials:
        by_group_trials[t.group].append(t)
    by_group_metric: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for v in values:
        if v.value is not None and np.isfinite(v.value):
            by_group_metric[(v.model_id, v.prompt_id)][v.metric][v.sample_id].append(float(v.value))
    configs: list[ConfigSummary] = []
    for index, group in enumerate(sorted(by_group_trials)):
        group_trials = by_group_trials[group]
        metrics: dict[str, MetricSummary] = {}
        for name, per_sample in sorted(by_group_metric[group].items()):
            higher, kind = metric_direction(name)
            metrics[name] = MetricSummary(
                metric=name,
                higher_is_better=higher,
                kind=kind,
                ci=_ci(per_sample, seed=index, n_resamples=n_resamples),
            )
        n = len(group_trials)
        pass_by_sample = by_group_metric[group].get("pass", {})
        tags: dict[str, list[float]] = defaultdict(list)
        for sid, vals in pass_by_sample.items():
            sample = samples.get(sid)
            for tag in sample.tags if sample else []:
                tags[tag].append(float(np.mean(vals)))
        first_output: dict[str, str] = {}
        for t in group_trials:
            first_output.setdefault(t.sample_id, t.output_text if t.ok else f"<{t.error_class}>")
        worst = sorted(pass_by_sample.items(), key=lambda kv: (np.mean(kv[1]), kv[0]))[:worst_n]
        configs.append(
            ConfigSummary(
                model_id=group[0],
                prompt_id=group[1],
                n_samples=len({t.sample_id for t in group_trials}),
                n_trials=n,
                metrics=metrics,
                latency=_latency(group_trials),
                cost=_cost(group_trials),
                failures=dict(Counter(t.error_class.value for t in group_trials if t.error_class)),
                failure_rate=sum(not t.ok for t in group_trials) / n,
                malformed_rate=sum(t.malformed for t in group_trials) / n,
                refusal_rate=sum(t.refused for t in group_trials) / n,
                retry_distribution=dict(sorted(Counter(t.attempts for t in group_trials).items())),
                tags=[(tag, len(v), float(np.mean(v))) for tag, v in sorted(tags.items())],
                worst=[
                    SampleRow(
                        sample_id=sid,
                        pass_rate=float(np.mean(vals)),
                        input=samples[sid].input if sid in samples else "",
                        reference=str(samples[sid].reference) if sid in samples else "",
                        output=first_output.get(sid, ""),
                        tags=list(samples[sid].tags) if sid in samples else [],
                    )
                    for sid, vals in worst
                    if np.mean(vals) < 1.0
                ],
            )
        )
    quality = "pass"
    objectives = [Objective(quality, True), Objective("latency_p50", False)]
    have_cost = all(c.cost.per_1k_requests_usd is not None for c in configs)
    if have_cost:
        objectives.insert(1, Objective("cost_per_1k", False))
    points: dict[str, list[float]] = {}
    for cfg in configs:
        q, lat = cfg.value(quality), cfg.latency.p50
        if q is None or lat is None:
            continue
        point = [q, lat]
        if have_cost:
            point.insert(1, float(cfg.cost.per_1k_requests_usd or 0.0))
        points[cfg.key] = point
    judge_calls = store.judge_calls(run.id)
    judge: dict[str, Any] = {}
    if judge_calls:
        judge = {
            "model": f"{judge_calls[0].judge_provider}/{judge_calls[0].judge_model}",
            "calls": len(judge_calls),
            "malformed": sum(c.error is not None for c in judge_calls),
            "cost_usd": sum(c.cost_usd or 0.0 for c in judge_calls),
            "candidate_overlap": any(
                m.get("model") == judge_calls[0].judge_model for m in run.manifest.get("models", [])
            ),
        }
        judge.update(_length_bias(values))
    return RunSummary(
        run_id=run.id,
        name=run.name,
        status=run.status,
        task=run.task,
        created_at=run.created_at,
        dataset_name=run.dataset_name,
        dataset_hash=run.dataset_hash,
        n_expected=run.n_expected,
        n_done=run.n_done,
        total_cost_usd=run.total_cost_usd,
        stop_reason=run.stop_reason,
        manifest=run.manifest,
        quality_metric=quality,
        configs=configs,
        pareto=pareto_front(points, objectives),
        pareto_objectives=[o.name for o in objectives],
        judge=judge,
        calibration=store.latest_artifact_json(run.id, "calibration"),
        skipped_models=run.manifest.get("skipped_models", {}),
    )


def _length_bias(values: list[MetricValue]) -> dict[str, Any]:
    from llmbench.judges.metrics import length_bias

    judged = [v for v in values if v.metric == "llm_judge_correctness" and v.value is not None]
    if not judged:
        return {}
    lengths = [int(v.details.get("length", 0)) for v in judged]
    if not any(lengths):
        return {}
    return {"length_bias_spearman": length_bias(lengths, [float(v.value or 0) for v in judged])}
