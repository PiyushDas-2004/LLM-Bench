"""Compare one metric between two configurations on aligned samples, with a verdict.

Every comparison reports the point estimates with cluster-bootstrap CIs, the
paired difference with its CI, the test used and its (adjusted) p-value, an
effect size, the minimum detectable effect, and a verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

import numpy as np
from pydantic import BaseModel

from llmbench.stats.bootstrap import ConfidenceInterval, cluster_bootstrap_ci
from llmbench.stats.correction import adjust
from llmbench.stats.effect import cliffs_delta, cohens_dz, magnitude_label, relative_delta
from llmbench.stats.paired import (
    TestResult,
    mcnemar_test,
    paired_bootstrap,
    paired_permutation_test,
    wilcoxon_test,
)
from llmbench.stats.power import power_report


class Verdict(StrEnum):
    """Outcome of a comparison, from the candidate's point of view."""

    IMPROVED = "improved"
    REGRESSED = "regressed"
    NO_CHANGE = "no significant change"
    INSUFFICIENT = "insufficient data"


class MetricComparison(BaseModel):
    """Baseline vs candidate on one metric."""

    metric: str
    higher_is_better: bool
    kind: str
    n_samples: int
    n_trials_baseline: int
    n_trials_candidate: int
    baseline: ConfidenceInterval | None
    candidate: ConfidenceInterval | None
    delta: float | None
    delta_low: float | None
    delta_high: float | None
    relative_delta: float | None
    test: str
    p_value: float | None
    p_adjusted: float | None
    effect_size_name: str
    effect_size: float | None
    effect_magnitude: str | None
    mde: float | None
    alpha: float
    verdict: Verdict
    warning: str | None = None

    @property
    def significant(self) -> bool:
        """Adjusted p-value below alpha."""
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return p is not None and p < self.alpha


def _verdict(cmp: MetricComparison) -> Verdict:
    if cmp.n_samples < 2 or cmp.delta is None:
        return Verdict.INSUFFICIENT
    excludes_zero = (
        cmp.delta_low is not None
        and cmp.delta_high is not None
        and (cmp.delta_low > 0 or cmp.delta_high < 0)
    )
    if not (cmp.significant and excludes_zero):
        return Verdict.NO_CHANGE
    better = cmp.delta > 0 if cmp.higher_is_better else cmp.delta < 0
    return Verdict.IMPROVED if better else Verdict.REGRESSED


def choose_test(
    kind: str, baseline: Mapping[str, Sequence[float]], candidate: Mapping[str, Sequence[float]]
) -> str:
    """``mcnemar`` for single-trial binary metrics, else ``permutation`` on per-sample means."""
    single = all(len(v) == 1 for v in baseline.values()) and all(
        len(v) == 1 for v in candidate.values()
    )
    return "mcnemar" if kind == "binary" and single else "permutation"


def compare_metric(
    metric: str,
    baseline: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
    *,
    higher_is_better: bool = True,
    kind: str = "continuous",
    alpha: float = 0.05,
    test: str = "auto",
    n_resamples: int = 2000,
    seed: int = 0,
) -> MetricComparison:
    """Compare per-sample values (``sample_id -> trial values``) on their common samples."""
    common = sorted(set(baseline) & set(candidate))
    base = {k: list(baseline[k]) for k in common if baseline[k]}
    cand = {k: list(candidate[k]) for k in common if candidate[k]}
    common = sorted(set(base) & set(cand))
    n = len(common)
    empty = MetricComparison(
        metric=metric,
        higher_is_better=higher_is_better,
        kind=kind,
        n_samples=n,
        n_trials_baseline=sum(len(base[k]) for k in common),
        n_trials_candidate=sum(len(cand[k]) for k in common),
        baseline=None,
        candidate=None,
        delta=None,
        delta_low=None,
        delta_high=None,
        relative_delta=None,
        test="none",
        p_value=None,
        p_adjusted=None,
        effect_size_name="cohens_dz",
        effect_size=None,
        effect_magnitude=None,
        mde=None,
        alpha=alpha,
        verdict=Verdict.INSUFFICIENT,
        warning="fewer than 2 aligned samples" if n < 2 else None,
    )
    if n < 2:
        return empty
    b_means = np.array([np.mean(base[k]) for k in common])
    c_means = np.array([np.mean(cand[k]) for k in common])
    b_ci = cluster_bootstrap_ci([base[k] for k in common], n_resamples=n_resamples, seed=seed)
    c_ci = cluster_bootstrap_ci([cand[k] for k in common], n_resamples=n_resamples, seed=seed + 1)
    diff_ci, boot_test = paired_bootstrap(b_means, c_means, n_resamples=n_resamples, seed=seed)
    chosen = (
        choose_test(kind, {k: base[k] for k in common}, {k: cand[k] for k in common})
        if test == "auto"
        else test
    )
    result: TestResult
    if chosen == "mcnemar":
        result = mcnemar_test(b_means > 0.5, c_means > 0.5)
    elif chosen == "wilcoxon":
        result = wilcoxon_test(b_means, c_means)
    elif chosen == "bootstrap":
        result = boot_test
    elif chosen == "permutation":
        result = paired_permutation_test(b_means, c_means, seed=seed)
    else:
        raise ValueError(f"unknown test {chosen!r}")
    if kind == "binary":
        effect_name, effect = "cliffs_delta", cliffs_delta(b_means, c_means)
    else:
        effect_name, effect = "cohens_dz", cohens_dz(b_means, c_means)
    power = power_report(b_means, c_means, alpha=alpha, binary=kind == "binary")
    delta = float(c_means.mean() - b_means.mean())
    cmp = empty.model_copy(
        update={
            "baseline": b_ci,
            "candidate": c_ci,
            "delta": delta,
            "delta_low": diff_ci.low,
            "delta_high": diff_ci.high,
            "relative_delta": relative_delta(float(b_means.mean()), float(c_means.mean())),
            "test": result.test,
            "p_value": result.p_value,
            "p_adjusted": result.p_value,
            "effect_size_name": effect_name,
            "effect_size": effect if np.isfinite(effect) else None,
            "effect_magnitude": magnitude_label(effect) if effect_name == "cohens_dz" else None,
            "mde": power.mde,
            "warning": power.warning,
        }
    )
    return cmp.model_copy(update={"verdict": _verdict(cmp)})


def apply_correction(
    comparisons: list[MetricComparison], method: str = "holm"
) -> list[MetricComparison]:
    """Adjust p-values across a family of comparisons and recompute verdicts."""
    indexed = [(i, c) for i, c in enumerate(comparisons) if c.p_value is not None]
    adjusted = adjust([c.p_value for _, c in indexed if c.p_value is not None], method)
    out = list(comparisons)
    for (i, c), p in zip(indexed, adjusted, strict=True):
        updated = c.model_copy(update={"p_adjusted": p})
        out[i] = updated.model_copy(update={"verdict": _verdict(updated)})
    return out
