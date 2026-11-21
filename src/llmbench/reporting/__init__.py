"""Reporting: run summaries, Pareto analysis, HTML / Markdown reports and exports."""

from llmbench.reporting.export import export_run, trial_rows
from llmbench.reporting.html import render_html
from llmbench.reporting.markdown import comparison_markdown, gate_markdown, run_markdown
from llmbench.reporting.pareto import Objective, dominates, pareto_front
from llmbench.reporting.summary import RunSummary, summarize_run

__all__ = [
    "Objective",
    "RunSummary",
    "comparison_markdown",
    "dominates",
    "export_run",
    "gate_markdown",
    "pareto_front",
    "render_html",
    "run_markdown",
    "summarize_run",
    "trial_rows",
]
