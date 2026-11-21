"""Render a :class:`RunSummary` as a self-contained HTML report (no external requests)."""

from __future__ import annotations

import datetime as dt
import json
from importlib import resources
from typing import Any

from jinja2 import Environment, select_autoescape

from llmbench import __version__
from llmbench.reporting.charts import BarRow, ScatterPoint, bar_chart, histogram, pareto_chart
from llmbench.reporting.pareto import Objective, pareto_front
from llmbench.reporting.summary import CONSISTENCY_METRICS, ConfigSummary, MetricSummary, RunSummary

HEADLINE_CANDIDATES = (
    "pass",
    "exact_match",
    "f1",
    "llm_judge_correctness",
    "json_schema",
    "field_accuracy",
    "injection_resistance",
    "over_refusal",
    "abstention_correct",
)


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _big(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.1f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f} ms"


def _money(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.2g}"
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def _ci_cell(summary: MetricSummary | None) -> str:
    if summary is None or summary.ci is None:
        return "n/a"
    ci = summary.ci
    return f"{ci.estimate:.3f} [{ci.low:.3f}, {ci.high:.3f}]"


def _tag_cell(cfg: ConfigSummary, tag: str) -> str:
    for name, n, value in cfg.tags:
        if name == tag:
            return f"{value:.3f} (n={n})"
    return "n/a"


def _environment() -> Environment:
    env = Environment(
        autoescape=select_autoescape(default=True), trim_blocks=True, lstrip_blocks=True
    )
    env.globals.update(
        num=_num, big=_big, pct=_pct, ms=_ms, money=_money, ci_cell=_ci_cell, tag_cell=_tag_cell
    )
    return env


def _scatter(xy: dict[str, tuple[float, float]]) -> list[ScatterPoint]:
    """Points for one 2-D projection, flagging that projection's own frontier (low x, high y)."""
    objectives = [Objective("x", maximize=False), Objective("y", maximize=True)]
    front = set(pareto_front({k: [x, y] for k, (x, y) in xy.items()}, objectives))
    return [ScatterPoint(k, x, y, k in front) for k, (x, y) in xy.items()]


def chart_context(summary: RunSummary) -> dict[str, Any]:
    """SVG charts and derived values used by the template."""
    quality = summary.quality_metric
    rows = []
    for cfg in summary.configs:
        metric = cfg.metrics.get(quality)
        if metric and metric.ci:
            rows.append(BarRow(cfg.key, metric.ci.estimate, metric.ci.low, metric.ci.high))
    rows.sort(key=lambda r: -r.value)
    cost_points = _scatter(
        {
            c.key: (float(c.cost.per_1k_requests_usd), float(c.value(quality) or 0))
            for c in summary.configs
            if c.cost.per_1k_requests_usd is not None and c.value(quality) is not None
        }
    )
    latency_points = _scatter(
        {
            c.key: (float(c.latency.p50), float(c.value(quality) or 0))
            for c in summary.configs
            if c.latency.p50 is not None and c.value(quality) is not None
        }
    )
    pareto_svg = (
        pareto_chart(
            cost_points,
            x_label="cost per 1k requests (USD)",
            y_label=quality,
            title="Pareto: quality vs cost",
        )
        if cost_points
        else pareto_chart(
            latency_points,
            x_label="p50 latency (ms)",
            y_label=quality,
            title="Pareto: quality vs latency",
        )
    )
    latency_svgs = {
        c.key: histogram(
            c.latency.histogram,
            c.latency.bin_edges,
            title=f"Latency distribution {c.key}",
            markers={
                k: v for k, v in (("p50", c.latency.p50), ("p95", c.latency.p95)) if v is not None
            },
        )
        for c in summary.configs
    }
    present = {m for c in summary.configs for m in c.metrics}
    return {
        "leaderboard_svg": bar_chart(rows, title=f"{quality} by configuration", x_max=1.0),
        "pareto_svg": pareto_svg,
        "pareto_latency_svg": (
            pareto_chart(
                latency_points,
                x_label="p50 latency (ms)",
                y_label=quality,
                title="Quality vs latency",
            )
            if cost_points
            else ""
        ),
        "latency_svgs": latency_svgs,
        "headline": [m for m in HEADLINE_CANDIDATES if m in present][:5],
        "consistency": [m for m in CONSISTENCY_METRICS if m in present],
        "tags": sorted({t[0] for c in summary.configs for t in c.tags}),
        "best_label": rows[0].label if rows else "n/a",
    }


def render_html(summary: RunSummary) -> str:
    """Render the full report."""
    source = (
        resources.files("llmbench.reporting")
        .joinpath("templates/report.html.j2")
        .read_text("utf-8")
    )
    template = _environment().from_string(source)
    manifest = summary.manifest
    return template.render(
        s=summary,
        mock=any(m.get("provider") == "mock" for m in manifest.get("models", [])),
        pricing_as_of=(manifest.get("pricing") or {}).get("as_of"),
        manifest_json=json.dumps(manifest, indent=2, sort_keys=True, default=str),
        version=__version__,
        generated=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        **chart_context(summary),
    )
