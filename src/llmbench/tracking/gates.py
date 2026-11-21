"""Regression gates: the CI check that fails when a gated metric regresses significantly.

A gate fails only when BOTH hold:

1. the change goes in the bad direction by more than ``max_drop`` (absolute)
   and, when ``max_relative_drop`` is set, also by more than that fraction of
   the baseline value, and
2. it is statistically significant (multiple-comparison adjusted p < alpha,
   with a delta CI that excludes zero).

Requiring both keeps the gate quiet on noise and on trivially small but
"significant" changes in very large runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmbench.errors import ConfigError
from llmbench.stats.compare import MetricComparison, Verdict
from llmbench.tracking.compare import RunComparison, compare_runs, metric_direction
from llmbench.tracking.store import RunStore


class Gate(BaseModel):
    """One gated metric."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    direction: Literal["higher_is_better", "lower_is_better"] | None = None
    max_drop: float = Field(default=0.0, ge=0)
    max_relative_drop: float | None = Field(default=None, ge=0)
    alpha: float | None = Field(default=None, gt=0, lt=1)


class GateConfig(BaseModel):
    """Contents of ``gates.yaml``."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.05, gt=0, lt=1)
    correction: Literal["holm", "bh", "none"] = "holm"
    allow_dataset_drift: bool = False
    gates: list[Gate] = Field(min_length=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GateConfig:
        """Load and validate a gates file."""
        file = Path(path)
        if not file.is_file():
            raise ConfigError(f"gates file not found: {file}")
        try:
            return cls.model_validate(yaml.safe_load(file.read_text(encoding="utf-8")))
        except (yaml.YAMLError, ValidationError) as exc:
            raise ConfigError(f"invalid gates file {file}:\n{exc}") from exc

    @classmethod
    def default(cls) -> GateConfig:
        """Gate on the task pass rate with a 2-point tolerance."""
        return cls(gates=[Gate(metric="pass", max_drop=0.02)])


class GateResult(BaseModel):
    """Outcome of one gate on one configuration pair."""

    metric: str
    config: str
    passed: bool
    reason: str
    comparison: MetricComparison | None


class GateReport(BaseModel):
    """All gate results plus the underlying comparison."""

    baseline_run: str
    candidate_run: str
    results: list[GateResult]
    comparison: RunComparison
    uses_mock: bool = False

    @property
    def passed(self) -> bool:
        """True when every gate passed."""
        return all(r.passed for r in self.results)

    @property
    def exit_code(self) -> int:
        """Process exit code for CI (0 pass, 1 regression)."""
        return 0 if self.passed else 1


def evaluate_gate(
    gate: Gate, cmp: MetricComparison | None, alpha: float, config: str
) -> GateResult:
    """Apply one gate to one metric comparison."""
    if cmp is None:
        return GateResult(
            metric=gate.metric,
            config=config,
            passed=False,
            reason="metric missing from one of the runs",
            comparison=None,
        )
    if cmp.verdict is Verdict.INSUFFICIENT or cmp.delta is None:
        return GateResult(
            metric=gate.metric,
            config=config,
            passed=False,
            reason="insufficient aligned data to evaluate the gate",
            comparison=cmp,
        )
    if gate.direction is not None:
        higher = gate.direction == "higher_is_better"
    else:
        higher = metric_direction(gate.metric)[0]
    drop = -cmp.delta if higher else cmp.delta
    base = cmp.baseline.estimate if cmp.baseline else 0.0
    relative = drop / abs(base) if base else None
    too_big = drop > gate.max_drop
    if gate.max_relative_drop is not None and relative is not None:
        too_big = too_big and relative > gate.max_relative_drop
    p = cmp.p_adjusted if cmp.p_adjusted is not None else cmp.p_value
    threshold = gate.alpha or alpha
    excludes_zero = (
        cmp.delta_low is not None
        and cmp.delta_high is not None
        and (cmp.delta_low > 0 or cmp.delta_high < 0)
    )
    significant = p is not None and p < threshold and excludes_zero
    p_text = "n/a" if p is None else f"{p:.4g}"
    if too_big and significant:
        reason = (
            f"regressed by {drop:.4f} (> max_drop {gate.max_drop}); "
            f"p_adj={p_text} < {threshold} with {cmp.test}"
        )
        return GateResult(
            metric=gate.metric, config=config, passed=False, reason=reason, comparison=cmp
        )
    if too_big:
        reason = f"drop {drop:.4f} exceeds tolerance but is not significant (p_adj={p_text})"
    else:
        reason = f"change {cmp.delta:+.4f} within tolerance"
    return GateResult(metric=gate.metric, config=config, passed=True, reason=reason, comparison=cmp)


def check(
    store: RunStore,
    baseline_ref: str,
    candidate_ref: str,
    gates: GateConfig,
    *,
    pairs: list[tuple[str, str]] | None = None,
    allow_dataset_drift: bool | None = None,
) -> GateReport:
    """Compare candidate against baseline and evaluate every gate on every configuration pair."""
    comparison = compare_runs(
        store,
        baseline_ref,
        candidate_ref,
        metrics=sorted({g.metric for g in gates.gates}),
        pairs=pairs,
        allow_dataset_drift=(
            gates.allow_dataset_drift if allow_dataset_drift is None else allow_dataset_drift
        ),
        alpha=gates.alpha,
        correction=gates.correction,
    )
    results = [
        evaluate_gate(gate, group.metric(gate.metric), gates.alpha, group.candidate_config)
        for group in comparison.groups
        for gate in gates.gates
    ]
    uses_mock = any(
        m.get("provider") == "mock"
        for ref in (comparison.baseline_run, comparison.candidate_run)
        for m in store.get_run(ref).manifest.get("models", [])
    )
    return GateReport(
        baseline_run=comparison.baseline_run,
        candidate_run=comparison.candidate_run,
        results=results,
        comparison=comparison,
        uses_mock=uses_mock,
    )
