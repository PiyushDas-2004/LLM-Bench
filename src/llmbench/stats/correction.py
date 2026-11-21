"""Multiple-comparison corrections for families of p-values."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (controls family-wise error)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values (controls the false discovery rate)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running = 1.0
    for i, idx in enumerate(order):
        rank = m - i
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = running
    return [float(v) for v in np.clip(adjusted, 0, 1)]


def adjust(p_values: Sequence[float], method: str = "holm") -> list[float]:
    """Dispatch to ``holm``, ``bh`` (Benjamini-Hochberg) or ``none``."""
    if method == "holm":
        return holm(p_values)
    if method in {"bh", "benjamini-hochberg", "fdr"}:
        return benjamini_hochberg(p_values)
    if method == "none":
        return [float(p) for p in p_values]
    raise ValueError(f"unknown correction method {method!r}")
