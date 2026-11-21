"""Power analysis: the smallest difference a benchmark of this size can reliably detect."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel
from scipy import stats as sps

#: Below this many samples, comparisons are flagged as unreliable regardless of variance.
MIN_RECOMMENDED_SAMPLES = 30


class PowerReport(BaseModel):
    """Minimum detectable effect for a paired comparison."""

    n_samples: int
    sd_diff: float
    alpha: float
    power: float
    mde: float
    warning: str | None = None


def minimum_detectable_effect(
    sd_diff: float, n: int, *, alpha: float = 0.05, power: float = 0.8
) -> float:
    """MDE of a two-sided paired test: (z_{1-a/2} + z_power) * sd / sqrt(n) (normal approx)."""
    if n < 1:
        return math.inf
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return float(z * sd_diff / math.sqrt(n))


def required_samples(
    sd_diff: float, effect: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Samples needed to detect ``effect`` with the given power (normal approximation)."""
    if effect <= 0:
        raise ValueError("effect must be positive")
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return max(2, math.ceil(float((z * sd_diff / effect) ** 2)))


def power_report(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    binary: bool = False,
) -> PowerReport:
    """Estimate the MDE from observed paired per-sample values.

    For binary metrics with no observed discordance the SD is floored at a
    conservative 0.5 so a zero-variance pilot run does not claim it could
    detect arbitrarily small differences.
    """
    d = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    n = int(d.size)
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if binary and sd == 0:
        sd = 0.5
    mde = minimum_detectable_effect(sd, n, alpha=alpha, power=power)
    warning = None
    if n < MIN_RECOMMENDED_SAMPLES:
        warning = (
            f"only {n} paired samples (< {MIN_RECOMMENDED_SAMPLES}); differences smaller than "
            f"~{mde:.3f} cannot be detected reliably"
        )
    return PowerReport(n_samples=n, sd_diff=sd, alpha=alpha, power=power, mde=mde, warning=warning)
