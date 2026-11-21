"""Paired hypothesis tests for comparing two configurations on the same samples."""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel
from scipy import stats as sps

from llmbench.stats.bootstrap import ConfidenceInterval

#: Up to this many non-zero differences the permutation test is exact (2**n sign flips).
EXACT_PERMUTATION_MAX_N = 16


class TestResult(BaseModel):
    """Outcome of a two-sided hypothesis test."""

    test: str
    statistic: float
    p_value: float


def _diffs(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("paired inputs must be non-empty 1-D sequences of equal length")
    out: NDArray[np.float64] = y - x
    return out


def paired_permutation_test(
    a: ArrayLike, b: ArrayLike, *, n_resamples: int = 10_000, seed: int = 0
) -> TestResult:
    """Sign-flip permutation test of mean(b - a) = 0.

    Under H0 each paired difference is symmetric around zero, so its sign is
    exchangeable. Exact enumeration is used for small n, Monte Carlo otherwise
    (with the +1 correction so p is never 0).
    """
    d = _diffs(a, b)
    observed = abs(d.mean())
    nonzero = d[d != 0]
    if nonzero.size == 0:
        return TestResult(test="paired-permutation", statistic=0.0, p_value=1.0)
    if nonzero.size <= EXACT_PERMUTATION_MAX_N:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=nonzero.size)))
        stats = np.abs((signs * nonzero).sum(axis=1) / d.size)
        p = float(np.mean(stats >= observed - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(n_resamples, d.size))
        stats = np.abs((signs * d).mean(axis=1))
        p = float((np.sum(stats >= observed - 1e-12) + 1) / (n_resamples + 1))
    return TestResult(test="paired-permutation", statistic=float(d.mean()), p_value=min(1.0, p))


def paired_bootstrap(
    a: ArrayLike,
    b: ArrayLike,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[ConfidenceInterval, TestResult]:
    """Bootstrap CI for mean(b - a) and a bootstrap p-value (shifted-null method)."""
    d = _diffs(a, b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    boots = d[idx].mean(axis=1)
    alpha = (1 - level) / 2
    low, high = np.quantile(boots, [alpha, 1 - alpha])
    centred = boots - d.mean()
    p = float((np.sum(np.abs(centred) >= abs(d.mean()) - 1e-12) + 1) / (n_resamples + 1))
    ci = ConfidenceInterval(
        estimate=float(d.mean()),
        low=float(low),
        high=float(high),
        level=level,
        method="paired-bootstrap",
        n_samples=int(d.size),
        n_trials=int(d.size),
    )
    return ci, TestResult(test="paired-bootstrap", statistic=float(d.mean()), p_value=min(1.0, p))


def mcnemar_test(a: ArrayLike, b: ArrayLike) -> TestResult:
    """McNemar's test for paired binary outcomes.

    Uses the exact binomial test on discordant pairs when there are fewer than
    25 of them, otherwise the chi-square statistic with continuity correction.
    """
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("paired inputs must be non-empty and of equal length")
    n01 = int(np.sum(x & ~y))  # baseline pass, candidate fail
    n10 = int(np.sum(~x & y))  # baseline fail, candidate pass
    discordant = n01 + n10
    if discordant == 0:
        return TestResult(test="mcnemar-exact", statistic=0.0, p_value=1.0)
    if discordant < 25:
        p = sps.binomtest(min(n01, n10), discordant, 0.5).pvalue
        return TestResult(test="mcnemar-exact", statistic=float(n10 - n01), p_value=float(p))
    chi2 = (abs(n01 - n10) - 1) ** 2 / discordant
    return TestResult(
        test="mcnemar-chi2", statistic=float(chi2), p_value=float(sps.chi2.sf(chi2, 1))
    )


def wilcoxon_test(a: ArrayLike, b: ArrayLike) -> TestResult:
    """Wilcoxon signed-rank test (zero differences split; p = 1 if all differences are zero)."""
    d = _diffs(a, b)
    if np.all(d == 0):
        return TestResult(test="wilcoxon", statistic=0.0, p_value=1.0)
    res = sps.wilcoxon(d, zero_method="zsplit")
    return TestResult(test="wilcoxon", statistic=float(res.statistic), p_value=float(res.pvalue))
