"""Run-to-run comparison: aligned per-sample statistics, regressions and tag breakdowns."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from llmbench.datasets.loader import load_dataset
from llmbench.domain import MetricValue, TrialRecord, config_key
from llmbench.errors import DatasetDriftError, LLMBenchError, RegistryError
from llmbench.metrics.base import create_metric
from llmbench.stats.compare import MetricComparison, Verdict, apply_correction, compare_metric
from llmbench.tracking.db import RunRow
from llmbench.tracking.store import RunStore

#: Metrics excluded from default comparisons (diagnostic, not quality signals).
DEFAULT_EXCLUDED = frozenset({"queue_wait_ms", "retries"})


class SampleDiff(BaseModel):
    """A sample whose pass rate changed between baseline and candidate."""

    sample_id: str
    baseline_score: float
    candidate_score: float
    baseline_output: str
    candidate_output: str
    tags: list[str] = Field(default_factory=list)


class TagBreakdown(BaseModel):
    """Pass rate per tag for both sides."""

    tag: str
    n_samples: int
    baseline: float
    candidate: float

    @property
    def delta(self) -> float:
        """candidate - baseline."""
        return self.candidate - self.baseline


class GroupComparison(BaseModel):
    """Comparison of one baseline configuration against one candidate configuration."""

    baseline_config: str
    candidate_config: str
    metrics: list[MetricComparison]
    regressions: list[SampleDiff]
    improvements: list[SampleDiff]
    tags: list[TagBreakdown]
    difficulty: list[TagBreakdown]

    def metric(self, name: str) -> MetricComparison | None:
        """Look up the comparison for one metric."""
        return next((m for m in self.metrics if m.metric == name), None)


class RunComparison(BaseModel):
    """Full comparison of two runs."""

    baseline_run: str
    candidate_run: str
    dataset_match: bool
    alpha: float
    correction: str
    groups: list[GroupComparison]
    notes: list[str] = Field(default_factory=list)


def metric_direction(name: str) -> tuple[bool, str]:
    """``(higher_is_better, kind)`` for a metric name (defaults for unknown plugins)."""
    try:
        metric = create_metric(name)
    except (RegistryError, TypeError):
        return True, "continuous"
    return metric.higher_is_better, metric.kind


def _index(
    values: Sequence[MetricValue],
) -> dict[tuple[str, str], dict[str, dict[str, list[float]]]]:
    """``(model, prompt) -> metric -> sample -> values``."""
    out: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for v in values:
        if v.value is not None and np.isfinite(v.value):
            out[(v.model_id, v.prompt_id)][v.metric][v.sample_id].append(float(v.value))
    return out


def _parse_pair(text: str) -> tuple[str, str]:
    model, sep, prompt = text.partition("/")
    if not sep:
        raise LLMBenchError(f"config selector must be 'model_id/prompt_id', got {text!r}")
    return model, prompt


def _pairs(
    base: set[tuple[str, str]],
    cand: set[tuple[str, str]],
    explicit: Sequence[tuple[str, str]] | None,
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    if explicit:
        pairs = [(_parse_pair(b), _parse_pair(c)) for b, c in explicit]
        for b, c in pairs:
            if b not in base or c not in cand:
                raise LLMBenchError(f"config pair {config_key(*b)} -> {config_key(*c)} not found")
        return pairs
    common = sorted(base & cand)
    if common:
        return [(g, g) for g in common]
    if len(base) == 1 and len(cand) == 1:
        return [(next(iter(base)), next(iter(cand)))]
    raise LLMBenchError(
        "runs share no (model, prompt) configuration; pass explicit pairs "
        "(e.g. --pair gpt-main/v1=gemini-main/v1)"
    )


def _first_outputs(trials: Sequence[TrialRecord], group: tuple[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in trials:
        if t.group == group and t.sample_id not in out:
            out[t.sample_id] = t.output_text if t.ok else f"<{t.error_class}>"
    return out


def _breakdown(
    base: dict[str, list[float]], cand: dict[str, list[float]], labels: dict[str, list[str]]
) -> list[TagBreakdown]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for sample_id, sample_labels in labels.items():
        for label in sample_labels:
            if sample_id in base and sample_id in cand:
                by_label[label].append(sample_id)
    return [
        TagBreakdown(
            tag=label,
            n_samples=len(ids),
            baseline=float(np.mean([np.mean(base[i]) for i in ids])),
            candidate=float(np.mean([np.mean(cand[i]) for i in ids])),
        )
        for label, ids in sorted(by_label.items())
    ]


def _sample_labels(run: RunRow) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Tags and difficulty per sample, read from the run's dataset (empty if unavailable)."""
    try:
        dataset = load_dataset(run.manifest["dataset"]["path"])
    except (LLMBenchError, KeyError, OSError):
        return {}, {}
    tags = {s.id: list(s.tags) for s in dataset.samples}
    difficulty = {s.id: [str(s.difficulty)] for s in dataset.samples if s.difficulty}
    return tags, difficulty


def compare_runs(
    store: RunStore,
    baseline_ref: str,
    candidate_ref: str,
    *,
    metrics: Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    allow_dataset_drift: bool = False,
    alpha: float = 0.05,
    correction: str = "holm",
    pass_metric: str = "pass",  # noqa: S107 - metric name, not a secret
    max_examples: int = 10,
    n_resamples: int = 2000,
) -> RunComparison:
    """Compare two stored runs configuration by configuration.

    Raises:
        DatasetDriftError: when the runs used different dataset versions and
            ``allow_dataset_drift`` is False.
    """
    base_run = store.resolve_run(baseline_ref)
    cand_run = store.resolve_run(candidate_ref)
    notes: list[str] = []
    match = base_run.dataset_hash == cand_run.dataset_hash
    if not match:
        message = (
            f"dataset versions differ ({base_run.dataset_hash[:12]} vs "
            f"{cand_run.dataset_hash[:12]})"
        )
        if not allow_dataset_drift:
            raise DatasetDriftError(message + "; pass --allow-dataset-drift to compare anyway")
        notes.append(message + "; compared on the intersection of sample ids only")
    base_idx = _index(store.metric_values(base_run.id))
    cand_idx = _index(store.metric_values(cand_run.id))
    base_trials = store.trials(base_run.id)
    cand_trials = store.trials(cand_run.id)
    tags, difficulty = _sample_labels(cand_run)
    groups: list[GroupComparison] = []
    for b_group, c_group in _pairs(set(base_idx), set(cand_idx), pairs):
        b_metrics, c_metrics = base_idx[b_group], cand_idx[c_group]
        names = (
            list(metrics)
            if metrics
            else sorted((set(b_metrics) & set(c_metrics)) - DEFAULT_EXCLUDED)
        )
        comparisons = []
        for name in names:
            if name not in b_metrics or name not in c_metrics:
                notes.append(f"metric {name!r} missing for {config_key(*b_group)}; skipped")
                continue
            higher, kind = metric_direction(name)
            comparisons.append(
                compare_metric(
                    name,
                    b_metrics[name],
                    c_metrics[name],
                    higher_is_better=higher,
                    kind=kind,
                    alpha=alpha,
                    n_resamples=n_resamples,
                )
            )
        comparisons = apply_correction(comparisons, correction)
        b_pass = b_metrics.get(pass_metric, {})
        c_pass = c_metrics.get(pass_metric, {})
        b_out = _first_outputs(base_trials, b_group)
        c_out = _first_outputs(cand_trials, c_group)
        diffs = [
            SampleDiff(
                sample_id=sid,
                baseline_score=float(np.mean(b_pass[sid])),
                candidate_score=float(np.mean(c_pass[sid])),
                baseline_output=b_out.get(sid, ""),
                candidate_output=c_out.get(sid, ""),
                tags=tags.get(sid, []),
            )
            for sid in sorted(set(b_pass) & set(c_pass))
        ]
        regressions = sorted(
            (d for d in diffs if d.candidate_score < d.baseline_score),
            key=lambda d: d.candidate_score - d.baseline_score,
        )[:max_examples]
        improvements = sorted(
            (d for d in diffs if d.candidate_score > d.baseline_score),
            key=lambda d: d.baseline_score - d.candidate_score,
        )[:max_examples]
        groups.append(
            GroupComparison(
                baseline_config=config_key(*b_group),
                candidate_config=config_key(*c_group),
                metrics=comparisons,
                regressions=regressions,
                improvements=improvements,
                tags=_breakdown(b_pass, c_pass, tags),
                difficulty=_breakdown(b_pass, c_pass, difficulty),
            )
        )
    return RunComparison(
        baseline_run=base_run.id,
        candidate_run=cand_run.id,
        dataset_match=match,
        alpha=alpha,
        correction=correction,
        groups=groups,
        notes=notes,
    )


def summarize_verdicts(comparison: RunComparison) -> dict[str, int]:
    """Count verdicts across all groups and metrics."""
    counts = {v.value: 0 for v in Verdict}
    for group in comparison.groups:
        for metric in group.metrics:
            counts[metric.verdict.value] += 1
    return counts
