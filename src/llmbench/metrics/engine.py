"""Metric evaluation: resolve configured metrics, compute them and persist the values."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from llmbench.config.loader import parse_experiment
from llmbench.config.models import ExperimentConfig, MetricSpec
from llmbench.config.settings import Settings, get_settings
from llmbench.datasets.loader import hash_samples, load_dataset
from llmbench.domain import TrialRecord
from llmbench.errors import DatasetDriftError
from llmbench.judges.factory import build_judge
from llmbench.metrics.base import (
    METRIC_BUNDLES,
    MetricContext,
    MetricResult,
    create_metric,
    expand_metric_names,
)
from llmbench.runner.prepare import load_pricing
from llmbench.tasks.base import get_task
from llmbench.tracking.store import RunStore

log = structlog.get_logger(__name__)

#: Metrics computed for every run because reports depend on them.
ALWAYS_COMPUTED: tuple[str, ...] = (
    "pass",
    *METRIC_BUNDLES["reliability"],
    *METRIC_BUNDLES["latency"],
    *METRIC_BUNDLES["cost"],
)


def resolve_metric_specs(config: ExperimentConfig) -> list[tuple[str, dict[str, Any]]]:
    """Expand bundles and attach params; configured metrics first, then the always-on set."""
    specs = config.metrics or [MetricSpec(name=n) for n in get_task(config.task).default_metrics]
    resolved: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for name in expand_metric_names([spec.name]):
            resolved.setdefault(name, dict(spec.params))
    for name in ALWAYS_COMPUTED:
        resolved.setdefault(name, {})
    return list(resolved.items())


async def evaluate_trials(
    trials: Sequence[TrialRecord],
    specs: Sequence[tuple[str, dict[str, Any]]],
    ctx: MetricContext,
) -> list[MetricResult]:
    """Compute every metric in ``specs`` over ``trials``."""
    results = []
    for name, params in specs:
        metric = create_metric(name, **params)
        results.append(await metric.compute(trials, ctx))
    return results


def config_from_run(manifest: dict[str, Any]) -> ExperimentConfig:
    """Re-create the experiment config stored in a run manifest."""
    return parse_experiment(manifest["config"], Path(manifest["base_dir"]))


def build_context(
    store: RunStore, run_id: str, settings: Settings | None = None
) -> tuple[MetricContext, ExperimentConfig]:
    """Load the samples, task, pricing and judge needed to score a stored run."""
    run = store.get_run(run_id)
    config = config_from_run(run.manifest)
    dataset = load_dataset(config.resolve_path(config.dataset))
    wanted = set(run.manifest["dataset"]["sample_ids"])
    samples = [s for s in dataset.samples if s.id in wanted]
    if hash_samples(samples) != run.dataset_hash:
        raise DatasetDriftError(
            f"dataset file changed since run {run_id}; metrics would not match its outputs"
        )
    ctx = MetricContext(
        samples={s.id: s for s in samples},
        task=get_task(config.task),
        pricing=load_pricing(config),
        resolve_path=config.resolve_path,
        judge=build_judge(config, settings or get_settings()),
        store=store,
        run_id=run_id,
    )
    return ctx, config


async def evaluate_run(store: RunStore, run_id: str, settings: Settings | None = None) -> None:
    """Compute all configured metrics for a stored run and replace its metric values."""
    ctx, config = build_context(store, run_id, settings)
    trials = store.trials(run_id)
    results = await evaluate_trials(trials, resolve_metric_specs(config), ctx)
    for result in results:
        store.replace_metric_values(run_id, result.values, metrics=[result.metric])
    log.info("metrics_computed", run_id=run_id, metrics=len(results))
