"""Pareto analysis over (quality, cost, latency)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Objective:
    """One optimisation axis: a value per point and whether larger is better."""

    name: str
    maximize: bool


def dominates(a: Sequence[float], b: Sequence[float], maximize: Sequence[bool]) -> bool:
    """True if ``a`` is at least as good as ``b`` on every axis and strictly better on one."""
    at_least = all((x >= y) if m else (x <= y) for x, y, m in zip(a, b, maximize, strict=True))
    strictly = any((x > y) if m else (x < y) for x, y, m in zip(a, b, maximize, strict=True))
    return at_least and strictly


def pareto_front(
    points: Mapping[str, Sequence[float]], objectives: Sequence[Objective]
) -> list[str]:
    """Keys of points not dominated by any other point (input order preserved)."""
    maximize = [o.maximize for o in objectives]
    return [
        key
        for key, value in points.items()
        if not any(dominates(other, value, maximize) for k, other in points.items() if k != key)
    ]
