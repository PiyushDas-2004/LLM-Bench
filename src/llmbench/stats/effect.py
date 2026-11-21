"""Effect sizes, so "significant" is never reported without "how big"."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def cohens_dz(a: ArrayLike, b: ArrayLike) -> float:
    """Paired Cohen's d_z = mean(b - a) / sd(b - a); 0 when there is no variation."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if d.size < 2:
        return 0.0
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return 0.0 if float(np.mean(d)) == 0 else float(np.sign(np.mean(d)) * np.inf)
    return float(np.mean(d) / sd)


def cohens_d(a: ArrayLike, b: ArrayLike) -> float:
    """Independent-samples Cohen's d with pooled standard deviation."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    pooled = np.sqrt(
        ((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1)) / (x.size + y.size - 2)
    )
    if pooled == 0:
        return 0.0
    return float((y.mean() - x.mean()) / pooled)


def cliffs_delta(a: ArrayLike, b: ArrayLike) -> float:
    """Cliff's delta = P(b > a) - P(b < a) over all pairs; in [-1, 1]."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return 0.0
    diff = y[:, None] - x[None, :]
    wins, losses = int(np.sum(diff > 0)), int(np.sum(diff < 0))
    return (wins - losses) / diff.size


def relative_delta(baseline: float, candidate: float) -> float | None:
    """(candidate - baseline) / |baseline|; ``None`` when the baseline is zero."""
    if baseline == 0:
        return None
    return (candidate - baseline) / abs(baseline)


def magnitude_label(d: float) -> str:
    """Conventional label for a standardised effect size (Cohen 1988 thresholds)."""
    size = abs(d)
    if size < 0.2:
        return "negligible"
    if size < 0.5:
        return "small"
    if size < 0.8:
        return "medium"
    return "large"
