# ruff: noqa: E501 - SVG markup reads best as one element per line
"""Dependency-free inline SVG charts for the self-contained HTML report.

Charts reference CSS custom properties (``--series-1``, ``--muted``, ...) so
the report's light and dark themes apply without regenerating SVG. Each mark
carries an SVG ``<title>`` for a native hover tooltip; every chart is also
backed by a table in the report.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class BarRow:
    """One bar with an optional confidence interval."""

    label: str
    value: float
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class ScatterPoint:
    """One point in the Pareto chart."""

    label: str
    x: float
    y: float
    on_front: bool


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """ "Nice" tick values (1, 2, 2.5, 5 x 10^k steps) covering [lo, hi]."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    ticks = []
    value = start
    while value <= hi + 1e-9 * step:
        ticks.append(round(value, 12))
        value += step
    return ticks


def bar_chart(
    rows: Sequence[BarRow], *, title: str, x_max: float | None = None, unit: str = ""
) -> str:
    """Horizontal bars (one hue) with CI whiskers and right-aligned value labels."""
    if not rows:
        return ""
    label_w, plot_w, row_h, top = 190, 420, 30, 28
    height = top + row_h * len(rows) + 30
    vmax = x_max or max(max(r.high if r.high is not None else r.value for r in rows), 1e-9)

    def sx(v: float) -> float:
        return label_w + plot_w * max(0.0, min(v, vmax)) / vmax

    parts = [
        f'<svg class="chart" viewBox="0 0 {label_w + plot_w + 90} {height}" role="img" '
        f'aria-label="{escape(title)}">'
    ]
    for tick in _ticks(0, vmax):
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{height - 26}" class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 10}" class="axis" text-anchor="middle">{_fmt(tick)}{unit}</text>'
        )
    for i, row in enumerate(rows):
        y = top + i * row_h
        width = max(sx(row.value) - label_w, 1.0)
        tip = f"{row.label}: {_fmt(row.value)}{unit}"
        if row.low is not None and row.high is not None:
            tip += f" (95% CI {_fmt(row.low)}-{_fmt(row.high)})"
        parts.append(
            f'<text x="{label_w - 10}" y="{y + 15}" class="label" text-anchor="end">{escape(row.label)}</text>'
        )
        parts.append(
            f'<g class="mark"><title>{escape(tip)}</title>'
            f'<rect x="{label_w}" y="{y + 4}" width="{width:.1f}" height="16" rx="4" class="bar"/>'
        )
        if row.low is not None and row.high is not None:
            x1, x2 = sx(row.low), sx(row.high)
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y + 12}" x2="{x2:.1f}" y2="{y + 12}" class="whisker"/>'
                f'<line x1="{x1:.1f}" y1="{y + 7}" x2="{x1:.1f}" y2="{y + 17}" class="whisker"/>'
                f'<line x1="{x2:.1f}" y1="{y + 7}" x2="{x2:.1f}" y2="{y + 17}" class="whisker"/>'
            )
        parts.append("</g>")
        parts.append(
            f'<text x="{label_w + plot_w + 8}" y="{y + 15}" class="value">{_fmt(row.value)}{unit}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def pareto_chart(points: Sequence[ScatterPoint], *, x_label: str, y_label: str, title: str) -> str:
    """Quality (y) vs cost or latency (x); frontier points highlighted and connected."""
    if not points:
        return ""
    width, height, left, right, top, bottom = 640, 360, 64, 24, 20, 48
    xs, ys = [p.x for p in points], [p.y for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_pad = (x_hi - x_lo) * 0.12 or abs(x_hi) * 0.1 or 1.0
    y_pad = (y_hi - y_lo) * 0.15 or 0.05
    x_lo, x_hi, y_lo, y_hi = (
        max(0.0, x_lo - x_pad),
        x_hi + x_pad,
        max(0.0, y_lo - y_pad),
        min(1.0, y_hi + y_pad),
    )
    if y_hi <= y_lo:
        y_hi = y_lo + 0.1

    def sx(v: float) -> float:
        return left + (width - left - right) * (v - x_lo) / (x_hi - x_lo)

    def sy(v: float) -> float:
        return height - bottom - (height - top - bottom) * (v - y_lo) / (y_hi - y_lo)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
    ]
    for tick in _ticks(y_lo, y_hi):
        y = sy(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" class="axis" text-anchor="end">{_fmt(tick)}</text>'
        )
    for tick in _ticks(x_lo, x_hi):
        x = sx(tick)
        parts.append(
            f'<text x="{x:.1f}" y="{height - bottom + 18}" class="axis" text-anchor="middle">{_fmt(tick)}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="baseline"/>'
        f'<text x="{(left + width - right) / 2}" y="{height - 8}" class="axis-title" text-anchor="middle">{escape(x_label)}</text>'
        f'<text transform="translate(16 {(top + height - bottom) / 2}) rotate(-90)" class="axis-title" '
        f'text-anchor="middle">{escape(y_label)}</text>'
    )
    front = sorted((p for p in points if p.on_front), key=lambda p: p.x)
    if len(front) > 1:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(p.x):.1f},{sy(p.y):.1f}" for i, p in enumerate(front)
        )
        parts.append(f'<path d="{path}" class="frontier"/>')
    for p in sorted(points, key=lambda p: p.on_front):
        cls = "dot front" if p.on_front else "dot dominated"
        tip = f"{p.label}: {y_label} {_fmt(p.y)}, {x_label} {_fmt(p.x)}" + (
            " (Pareto-optimal)" if p.on_front else ""
        )
        x, y = sx(p.x), sy(p.y)
        anchor = "end" if x > width * 0.7 else "start"
        dx = -10 if anchor == "end" else 10
        parts.append(
            f'<g class="mark"><title>{escape(tip)}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="6" class="{cls}"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" class="hit"/></g>'
            f'<text x="{x + dx:.1f}" y="{y - 9:.1f}" class="label" text-anchor="{anchor}">{escape(p.label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def histogram(
    counts: Sequence[int], edges: Sequence[float], *, title: str, markers: dict[str, float]
) -> str:
    """Compact latency histogram with labelled percentile markers."""
    if not counts:
        return ""
    width, height, left, bottom, top = 300, 120, 8, 22, 18
    plot_w = width - 2 * left
    peak = max(max(counts), 1)
    lo, hi = edges[0], edges[-1]
    span = (hi - lo) or 1.0
    bar_w = plot_w / len(counts)
    parts = [
        f'<svg class="chart small" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
    ]
    for i, count in enumerate(counts):
        h = (height - top - bottom) * count / peak
        x = left + i * bar_w
        tip = f"{_fmt(edges[i])}-{_fmt(edges[i + 1])} ms: {count} calls"
        parts.append(
            f'<g class="mark"><title>{escape(tip)}</title><rect x="{x + 1:.1f}" y="{height - bottom - h:.1f}" '
            f'width="{max(bar_w - 2, 1):.1f}" height="{max(h, 0.5):.1f}" rx="2" class="bar"/></g>'
        )
    parts.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - left}" y2="{height - bottom}" class="baseline"/>'
    )
    for name, value in markers.items():
        x = left + plot_w * (value - lo) / span
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{height - bottom}" class="marker"/>'
            f'<text x="{x:.1f}" y="{top - 6}" class="axis" text-anchor="middle">{escape(name)}</text>'
        )
    parts.append(
        f'<text x="{left}" y="{height - 6}" class="axis">{_fmt(lo)} ms</text>'
        f'<text x="{width - left}" y="{height - 6}" class="axis" text-anchor="end">{_fmt(hi)} ms</text></svg>'
    )
    return "".join(parts)
