"""Bootstrap confidence intervals: percentile, BCa, and a two-stage cluster bootstrap.

Trials of the same sample are correlated (same input, same difficulty), so
treating every trial as independent understates uncertainty. The cluster
bootstrap resamples *samples* first and then *trials within each sample*.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel
from scipy import stats as sps

Statistic = Callable[[NDArray[np.float64]], float]


class ConfidenceInterval(BaseModel):
    """A point estimate with a two-sided confidence interval."""

    estimate: float
    low: float
    high: float
    level: float
    method: str
    n_samples: int
    n_trials: int

    @property
    def width(self) -> float:
        """``high - low``."""
        return self.high - self.low

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the interval (inclusive)."""
        return self.low <= value <= self.high


def _mean(x: NDArray[np.float64]) -> float:
    return float(np.mean(x))


def _as_array(values: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("need a non-empty 1-D sequence of values")
    return arr


def percentile_ci(
    values: ArrayLike,
    statistic: Statistic = _mean,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap CI of ``statistic`` over i.i.d. ``values``."""
    x = _as_array(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    boots = (
        np.array([statistic(x[row]) for row in idx]) if statistic is not _mean else x[idx].mean(1)
    )
    alpha = (1 - level) / 2
    low, high = np.quantile(boots, [alpha, 1 - alpha])
    return ConfidenceInterval(
        estimate=statistic(x),
        low=float(low),
        high=float(high),
        level=level,
        method="percentile",
        n_samples=int(x.size),
        n_trials=int(x.size),
    )


def bca_ci(
    values: ArrayLike,
    statistic: Statistic = _mean,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bias-corrected and accelerated (BCa) bootstrap CI (Efron 1987).

    Bias correction ``z0`` comes from the share of bootstrap replicates below
    the estimate; acceleration ``a`` from the jackknife skewness. Falls back to
    the percentile interval when the replicates are degenerate.
    """
    x = _as_array(values)
    theta = statistic(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    boots = (
        np.array([statistic(x[row]) for row in idx]) if statistic is not _mean else x[idx].mean(1)
    )
    if np.all(boots == boots[0]) or x.size < 3:
        return ConfidenceInterval(
            estimate=theta,
            low=float(np.min(boots)),
            high=float(np.max(boots)),
            level=level,
            method="bca",
            n_samples=int(x.size),
            n_trials=int(x.size),
        )
    prop = np.clip(np.mean(boots < theta) + 0.5 * np.mean(boots == theta), 1e-6, 1 - 1e-6)
    z0 = sps.norm.ppf(prop)
    jack = np.array([statistic(np.delete(x, i)) for i in range(x.size)])
    diffs = jack.mean() - jack
    denom = 6.0 * (np.sum(diffs**2) ** 1.5)
    accel = float(np.sum(diffs**3) / denom) if denom > 0 else 0.0
    alpha = (1 - level) / 2
    out = []
    for z_alpha in (sps.norm.ppf(alpha), sps.norm.ppf(1 - alpha)):
        adj = sps.norm.cdf(z0 + (z0 + z_alpha) / (1 - accel * (z0 + z_alpha)))
        out.append(float(np.quantile(boots, np.clip(adj, 0, 1))))
    return ConfidenceInterval(
        estimate=theta,
        low=out[0],
        high=out[1],
        level=level,
        method="bca",
        n_samples=int(x.size),
        n_trials=int(x.size),
    )


def _padded(clusters: Sequence[Sequence[float]]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    counts = np.array([len(c) for c in clusters], dtype=np.int64)
    if counts.size == 0 or np.any(counts == 0):
        raise ValueError("every cluster needs at least one value")
    matrix = np.full((counts.size, int(np.max(counts))), np.nan)
    for i, c in enumerate(clusters):
        matrix[i, : len(c)] = c
    return matrix, counts


def cluster_bootstrap_replicates(
    clusters: Sequence[Sequence[float]], *, n_resamples: int = 2000, seed: int = 0
) -> NDArray[np.float64]:
    """Two-stage bootstrap replicates of the mean of cluster means.

    Stage 1 resamples clusters (samples) with replacement; stage 2 resamples
    trials within each selected cluster with replacement.
    """
    matrix, counts = _padded(clusters)
    rng = np.random.default_rng(seed)
    n_clusters, width = matrix.shape
    chosen = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    inner = np.floor(rng.random((n_resamples, n_clusters, width)) * counts[chosen][..., None])
    inner = inner.astype(np.int64)
    picked = matrix[chosen[..., None], inner]
    valid = np.arange(width)[None, None, :] < counts[chosen][..., None]
    cluster_means = np.where(valid, picked, 0.0).sum(-1) / counts[chosen]
    out: NDArray[np.float64] = cluster_means.mean(axis=1)
    return out


def cluster_bootstrap_ci(
    clusters: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile CI for the mean of per-sample means using the two-stage cluster bootstrap."""
    groups = list(clusters.values()) if isinstance(clusters, Mapping) else list(clusters)
    boots = cluster_bootstrap_replicates(groups, n_resamples=n_resamples, seed=seed)
    alpha = (1 - level) / 2
    low, high = np.quantile(boots, [alpha, 1 - alpha])
    estimate = float(np.mean([np.mean(g) for g in groups]))
    return ConfidenceInterval(
        estimate=estimate,
        low=float(low),
        high=float(high),
        level=level,
        method="cluster-bootstrap",
        n_samples=len(groups),
        n_trials=int(sum(len(g) for g in groups)),
    )
