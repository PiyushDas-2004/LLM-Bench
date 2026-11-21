"""Rich console tables for the CLI."""

from __future__ import annotations

from collections.abc import Sequence

from rich.table import Table

from llmbench.reporting.html import HEADLINE_CANDIDATES
from llmbench.reporting.summary import RunSummary
from llmbench.runner.plan import PlanEstimate
from llmbench.stats.compare import Verdict
from llmbench.tracking.compare import RunComparison
from llmbench.tracking.db import RunRow
from llmbench.tracking.gates import GateReport

_STYLE = {
    Verdict.IMPROVED: "green",
    Verdict.REGRESSED: "bold red",
    Verdict.NO_CHANGE: "dim",
    Verdict.INSUFFICIENT: "yellow",
}


def runs_table(runs: Sequence[RunRow]) -> Table:
    """Recent runs."""
    table = Table(title="Runs", header_style="bold")
    for col in ("run id", "name", "status", "task", "trials", "cost (USD)", "created (UTC)"):
        table.add_column(col, justify="right" if col in {"trials", "cost (USD)"} else "left")
    for r in runs:
        cost = "unknown" if r.total_cost_usd is None else f"{r.total_cost_usd:.4f}"
        table.add_row(
            r.id,
            r.name,
            r.status,
            r.task,
            f"{r.n_done}/{r.n_expected}",
            cost,
            r.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    return table


def summary_table(summary: RunSummary) -> Table:
    """Compact leaderboard for one run (up to three headline metrics)."""
    present = {m for c in summary.configs for m in c.metrics}
    metrics = [
        m
        for m in HEADLINE_CANDIDATES
        if m in present and not (m == "exact_match" and "pass" in present)
    ][:3]
    table = Table(
        title=f"{summary.name} ({summary.run_id})",
        header_style="bold",
        caption="estimate [95% CI]; * = Pareto-optimal (quality, cost, latency)",
    )
    table.add_column("configuration", no_wrap=True)
    table.add_column("n", justify="right", no_wrap=True)
    for m in metrics:
        table.add_column(m.replace("llm_judge_correctness", "judge"), justify="right", no_wrap=True)
    for col in ("agreement", "p50 ms", "fail %", "$/1k req"):
        table.add_column(col, justify="right", no_wrap=True)
    for c in summary.configs:
        cells = []
        for m in metrics:
            s = c.metrics.get(m)
            cells.append(
                f"{s.ci.estimate:.3f} [{s.ci.low:.2f}, {s.ci.high:.2f}]" if s and s.ci else "n/a"
            )
        agreement = c.value("exact_agreement")
        table.add_row(
            c.key + (" *" if c.key in summary.pareto else ""),
            f"{c.n_samples}x{c.n_trials // max(c.n_samples, 1)}",
            *cells,
            "n/a" if agreement is None else f"{agreement:.3f}",
            "n/a" if c.latency.p50 is None else f"{c.latency.p50:,.0f}",
            f"{100 * c.failure_rate:.1f}",
            "unknown"
            if c.cost.per_1k_requests_usd is None
            else f"{c.cost.per_1k_requests_usd:.4f}",
        )
    return table


def _num(value: float, sign: bool = False) -> str:
    fmt = "+" if sign else ""
    if value != 0 and abs(value) < 0.01:
        return f"{value:{fmt}.3g}"
    return f"{value:{fmt}.3f}"


def comparison_tables(cmp: RunComparison) -> list[Table]:
    """One compact table per configuration pair."""
    tables = []
    for group in cmp.groups:
        table = Table(
            title=f"{group.baseline_config} -> {group.candidate_config}",
            header_style="bold",
            caption=f"alpha={cmp.alpha}, {cmp.correction} correction; delta = candidate - baseline",
        )
        table.add_column("metric", no_wrap=True)
        for col in ("baseline", "candidate", "delta [95% CI]", "test", "p adj.", "effect", "n"):
            table.add_column(col, justify="left" if col == "test" else "right", no_wrap=True)
        table.add_column("verdict", no_wrap=True, min_width=21)
        for m in group.metrics:
            d = (
                f"{_num(m.delta, True)} [{_num(m.delta_low, True)}, {_num(m.delta_high, True)}]"
                if m.delta is not None and m.delta_low is not None and m.delta_high is not None
                else "n/a"
            )
            p = (
                "n/a"
                if m.p_adjusted is None
                else ("<0.001" if m.p_adjusted < 0.001 else f"{m.p_adjusted:.3f}")
            )
            effect = "n/a" if m.effect_size is None else f"{m.effect_size:+.2f}"
            table.add_row(
                m.metric,
                _num(m.baseline.estimate) if m.baseline else "n/a",
                _num(m.candidate.estimate) if m.candidate else "n/a",
                d,
                m.test.replace("paired-", ""),
                p,
                effect,
                f"{m.n_samples}x{m.n_trials_candidate // max(m.n_samples, 1)}",
                f"[{_STYLE[m.verdict]}]{m.verdict.value}[/]",
            )
        tables.append(table)
    return tables


def plan_table(plan: PlanEstimate) -> Table:
    """Grid and cost estimate."""
    table = Table(
        title=f"Plan: {plan.n_models} models x {plan.n_prompts} prompts x "
        f"{plan.n_samples} samples x {plan.trials} trials = {plan.total_requests} requests",
        header_style="bold",
        caption="output tokens and cost are upper bounds (max_tokens per call)"
        + (f"; prices as of {plan.pricing_as_of}" if plan.pricing_as_of else "; no pricing table"),
    )
    for col in (
        "model",
        "provider/model",
        "requests",
        "input tokens",
        "output tokens (max)",
        "cost (max, USD)",
    ):
        table.add_column(col, justify="left" if col in {"model", "provider/model"} else "right")
    for m in plan.models:
        cost = "unknown" if m.cost_usd_upper is None else f"{m.cost_usd_upper:.4f}"
        table.add_row(
            m.model_id,
            f"{m.provider}/{m.model}",
            str(m.requests),
            f"{m.input_tokens:,}",
            f"{m.output_tokens_upper:,}",
            cost,
        )
    return table


def gate_table(report: GateReport) -> Table:
    """Gate results."""
    table = Table(
        title="Eval gate: " + ("PASSED" if report.passed else "FAILED"), header_style="bold"
    )
    for col in ("gate", "configuration", "result", "reason"):
        table.add_column(col)
    for r in report.results:
        result = "[green]pass[/]" if r.passed else "[bold red]FAIL[/]"
        table.add_row(r.metric, r.config, result, r.reason)
    return table
