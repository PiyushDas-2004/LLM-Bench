"""Drift detection over time for one (model, prompt, dataset, metric) series.

Two complementary checks (see METHODOLOGY.md for assumptions and limits):

* **Rolling robust z-score** of the newest run against the median and MAD of
  the previous ``window`` runs. Catches a sudden shift such as a silent
  provider-side model update.
* **Two-sided CUSUM** over the whole series, standardised by the first
  ``window`` points, to locate a sustained change point.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from pydantic import BaseModel

from llmbench.metrics.base import MetricResult
from llmbench.tracking.store import RunStore

MAD_TO_SD = 1.4826


class DriftPoint(BaseModel):
    """Aggregate metric value of one run."""

    run_id: str
    created_at: dt.datetime
    value: float


class DriftReport(BaseModel):
    """Result of drift detection on one series."""

    metric: str
    model_id: str
    prompt_id: str
    dataset_hash: str | None
    points: list[DriftPoint]
    window: int
    z_threshold: float
    latest_z: float | None
    latest_flagged: bool
    change_point_index: int | None
    message: str


def robust_z(history: list[float], value: float, min_scale: float = 1e-3) -> float:
    """(value - median) / (1.4826 * MAD), with a floor on the scale."""
    arr = np.asarray(history, dtype=np.float64)
    median = float(np.median(arr))
    scale = max(
        MAD_TO_SD * float(np.median(np.abs(arr - median))), float(np.std(arr)) * 0.5, min_scale
    )
    return (value - median) / scale


def cusum_change_point(
    values: list[float], window: int, k: float = 0.5, h: float = 4.0
) -> int | None:
    """Index where a two-sided CUSUM first exceeds ``h`` standard deviations, else ``None``."""
    if len(values) <= window or window < 2:
        return None
    ref = np.asarray(values[:window], dtype=np.float64)
    mu = float(ref.mean())
    sd = max(float(ref.std(ddof=1)), 1e-3)
    hi = lo = 0.0
    for i, v in enumerate(values[window:], start=window):
        z = (v - mu) / sd
        hi = max(0.0, hi + z - k)
        lo = max(0.0, lo - z - k)
        if hi > h or lo > h:
            return i
    return None


def detect_drift(
    points: list[DriftPoint],
    *,
    metric: str,
    model_id: str,
    prompt_id: str,
    dataset_hash: str | None = None,
    window: int = 5,
    z_threshold: float = 3.0,
) -> DriftReport:
    """Run both drift checks on a time-ordered series."""
    values = [p.value for p in points]
    latest_z: float | None = None
    flagged = False
    if len(values) > window:
        latest_z = robust_z(values[-window - 1 : -1], values[-1])
        flagged = abs(latest_z) > z_threshold
    change = cusum_change_point(values, window)
    if len(values) <= window:
        message = f"need more than {window} runs for drift detection (have {len(values)})"
    elif flagged:
        message = f"latest run deviates from the rolling baseline (robust z = {latest_z:+.2f})"
    elif change is not None:
        message = (
            f"sustained shift detected starting at run #{change + 1} ({points[change].run_id})"
        )
    else:
        message = "no drift detected"
    return DriftReport(
        metric=metric,
        model_id=model_id,
        prompt_id=prompt_id,
        dataset_hash=dataset_hash,
        points=points,
        window=window,
        z_threshold=z_threshold,
        latest_z=latest_z,
        latest_flagged=flagged,
        change_point_index=change,
        message=message,
    )


def series_from_store(
    store: RunStore,
    *,
    metric: str,
    model_id: str,
    prompt_id: str,
    name: str | None = None,
    dataset_hash: str | None = None,
    limit: int = 200,
) -> list[DriftPoint]:
    """Collect aggregate metric values per run, oldest first, for one configuration."""
    points = []
    for run in reversed(store.list_runs(limit=limit, name=name)):
        if dataset_hash and run.dataset_hash != dataset_hash:
            continue
        values = [
            v
            for v in store.metric_values(run.id, metric)
            if v.model_id == model_id and v.prompt_id == prompt_id
        ]
        if not values:
            continue
        result = MetricResult(
            metric=metric, higher_is_better=True, kind="continuous", level="trial", values=values
        )
        agg = result.aggregate(model_id, prompt_id)
        if agg is not None:
            points.append(DriftPoint(run_id=run.id, created_at=run.created_at, value=agg))
    return points
