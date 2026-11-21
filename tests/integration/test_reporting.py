from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from llmbench.config import load_experiment
from llmbench.metrics.engine import evaluate_run
from llmbench.reporting import (
    Objective,
    comparison_markdown,
    dominates,
    gate_markdown,
    pareto_front,
    render_html,
    run_markdown,
    summarize_run,
    trial_rows,
)
from llmbench.reporting.charts import (
    BarRow,
    ScatterPoint,
    _ticks,
    bar_chart,
    histogram,
    pareto_chart,
)
from llmbench.runner import FakeClock, Runner
from llmbench.tracking import RunStore
from llmbench.tracking.compare import compare_runs
from llmbench.tracking.gates import GateConfig, check

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
RUN_ID = re.compile(r"[a-z0-9-]+-\d{8}T\d{6}-[0-9a-f]{8}")


def _golden(name: str, text: str) -> None:
    normalised = RUN_ID.sub("<run-id>", text)
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN") == "1" or not path.exists():
        path.write_text(normalised, encoding="utf-8")
    assert normalised == path.read_text(encoding="utf-8")


@pytest.fixture
async def runs(tmp_path: Path, repo_root: Path) -> tuple[RunStore, str, str]:
    store = RunStore(f"sqlite:///{tmp_path / 'rep.db'}")
    ids = []
    for name in ("baseline", "regression"):
        cfg = load_experiment(repo_root / "examples" / "gate" / f"{name}.yaml")
        outcome = await Runner(store, clock=FakeClock(), evaluator=evaluate_run).run(cfg)
        ids.append(outcome.run_id)
    return store, ids[0], ids[1]


async def test_markdown_reports_match_golden_files(runs: tuple[RunStore, str, str]) -> None:
    store, base, cand = runs
    _golden("run_report.md", run_markdown(summarize_run(store, base)))
    _golden(
        "comparison_report.md",
        comparison_markdown(compare_runs(store, base, cand, metrics=["pass", "f1"])),
    )
    gate = check(store, base, cand, GateConfig.default())
    assert not gate.passed
    _golden("gate_report.md", gate_markdown(gate))


async def test_html_report_is_self_contained(runs: tuple[RunStore, str, str]) -> None:
    store, base, _ = runs
    summary = summarize_run(store, base)
    html = render_html(summary)
    assert html.startswith("<!doctype html>")
    for needle in ("Pareto", "Consistency across trials", "Run manifest", "Mock provider", "<svg"):
        assert needle in html
    assert not re.search(r"""(src|href)=["']https?://""", html)
    assert summary.configs[0].latency.p95 is not None
    assert summary.config("assistant/v1").n_trials == 120
    with pytest.raises(KeyError):
        summary.config("nope/v1")
    rows = trial_rows(store, base)
    assert len(rows) == 120
    assert "metric.exact_match" in rows[0]
    assert "sample_metric.exact_agreement" in rows[0]


def test_pareto() -> None:
    objectives = [Objective("q", True), Objective("cost", False)]
    points = {"a": [0.9, 10.0], "b": [0.8, 5.0], "c": [0.7, 6.0], "d": [0.9, 12.0]}
    assert pareto_front(points, objectives) == ["a", "b"]
    assert dominates([1, 1], [0, 2], [True, False])
    assert not dominates([1, 1], [1, 1], [True, False])


def test_charts() -> None:
    assert bar_chart([], title="x") == ""
    assert pareto_chart([], x_label="x", y_label="y", title="t") == ""
    assert histogram([], [], title="t", markers={}) == ""
    svg = bar_chart([BarRow("a", 0.5, 0.4, 0.6), BarRow("b<script>", 0.2)], title="t", x_max=1.0)
    assert "<script>" not in svg
    assert "95% CI" in svg
    single = pareto_chart(
        [ScatterPoint("only", 1.0, 0.5, True)], x_label="x", y_label="y", title="t"
    )
    assert "only" in single
    assert _ticks(0, 1) == [0, 0.25, 0.5, 0.75, 1.0]
    assert _ticks(5, 5) == [5]
    assert histogram([1, 3], [0.0, 1.0, 2.0], title="h", markers={"p50": 1.0}).count("<rect") == 2
