"""Markdown reports suitable for pull-request comments."""

from __future__ import annotations

from llmbench.reporting.html import HEADLINE_CANDIDATES
from llmbench.reporting.summary import RunSummary
from llmbench.stats.compare import MetricComparison, Verdict
from llmbench.tracking.compare import RunComparison
from llmbench.tracking.gates import GateReport

_ICON = {
    Verdict.IMPROVED: "improved",
    Verdict.REGRESSED: "**regressed**",
    Verdict.NO_CHANGE: "no significant change",
    Verdict.INSUFFICIENT: "insufficient data",
}


def _f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def run_markdown(summary: RunSummary) -> str:
    """Leaderboard-style Markdown summary of one run."""
    mock = any(m.get("provider") == "mock" for m in summary.manifest.get("models", []))
    present = {m for c in summary.configs for m in c.metrics}
    headline = [m for m in HEADLINE_CANDIDATES if m in present][:4]
    lines = [
        f"## LLMBench: {summary.name}",
        "",
        f"Run `{summary.run_id}` | task `{summary.task}` | dataset `{summary.dataset_name}` "
        f"(`{summary.dataset_hash[:12]}`) | {summary.n_done}/{summary.n_expected} trials | "
        f"status `{summary.status}`",
        "",
    ]
    if mock:
        lines += [
            "> Mock provider: numbers are illustrative, not measurements of a real model; "
            "mock costs use fictional prices and nothing was billed.",
            "",
        ]
    lines.append("| configuration | " + " | ".join(headline) + " | p50 latency | cost / 1k req |")
    lines.append("|---|" + "---:|" * (len(headline) + 2))
    for cfg in summary.configs:
        cells = []
        for m in headline:
            s = cfg.metrics.get(m)
            cells.append(
                f"{s.ci.estimate:.3f} [{s.ci.low:.3f}, {s.ci.high:.3f}]" if s and s.ci else "n/a"
            )
        star = " (Pareto)" if cfg.key in summary.pareto else ""
        latency = f"{cfg.latency.p50:,.0f} ms" if cfg.latency.p50 is not None else "n/a"
        cost = (
            f"${cfg.cost.per_1k_requests_usd:.4f}"
            if cfg.cost.per_1k_requests_usd is not None
            else "unknown"
        )
        lines.append(f"| `{cfg.key}`{star} | " + " | ".join(cells) + f" | {latency} | {cost} |")
    lines += ["", "Values: estimate [95% cluster-bootstrap CI]."]
    return "\n".join(lines) + "\n"


def _v(value: float, sign: bool = False) -> str:
    """3 decimals, or 3 significant digits for very small magnitudes (e.g. per-call cost)."""
    fmt = "+" if sign else ""
    if value != 0 and abs(value) < 0.01:
        return f"{value:{fmt}.3g}"
    return f"{value:{fmt}.3f}"


def _metric_row(m: MetricComparison) -> str:
    base = _v(m.baseline.estimate) if m.baseline else "n/a"
    cand = _v(m.candidate.estimate) if m.candidate else "n/a"
    delta = (
        f"{_v(m.delta, True)} [{_v(m.delta_low, True)}, {_v(m.delta_high, True)}]"
        if m.delta is not None and m.delta_low is not None and m.delta_high is not None
        else "n/a"
    )
    effect = f"{m.effect_size_name}={_f(m.effect_size, 2)}"
    return (
        f"| {m.metric} | {base} | {cand} | {delta} | {m.test} | {_p(m.p_adjusted)} | {effect} | "
        f"{m.n_samples} x {m.n_trials_candidate // max(m.n_samples, 1)} | {_ICON[m.verdict]} |"
    )


def comparison_markdown(cmp: RunComparison, max_examples: int = 5) -> str:
    """Baseline vs candidate comparison with CIs, tests, effect sizes and verdicts."""
    lines = [
        f"## LLMBench comparison: `{cmp.baseline_run}` -> `{cmp.candidate_run}`",
        "",
        f"alpha = {cmp.alpha}, correction = {cmp.correction}, dataset match = {cmp.dataset_match}",
        "",
    ]
    lines += [f"> {note}" for note in cmp.notes]
    for group in cmp.groups:
        lines += [
            f"### `{group.baseline_config}` -> `{group.candidate_config}`",
            "",
            "| metric | baseline | candidate | delta [95% CI] | test | p (adj.) | effect size "
            "| n | verdict |",
            "|---|---:|---:|---:|---|---:|---:|---:|---|",
        ]
        lines += [_metric_row(m) for m in group.metrics]
        warnings = sorted({m.warning for m in group.metrics if m.warning})
        lines += [""] + [f"> {w}" for w in warnings]
        if group.regressions:
            lines += [
                "",
                "<details><summary>Per-sample regressions</summary>",
                "",
                "| sample | baseline | candidate | baseline output | candidate output |",
                "|---|---:|---:|---|---|",
            ]
            for d in group.regressions[:max_examples]:
                lines.append(
                    f"| {d.sample_id} | {d.baseline_score:.2f} | {d.candidate_score:.2f} | "
                    f"{_cell(d.baseline_output)} | {_cell(d.candidate_output)} |"
                )
            lines += ["", "</details>"]
        if group.tags:
            lines += ["", "| tag | n | baseline | candidate | delta |", "|---|---:|---:|---:|---:|"]
            lines += [
                f"| {t.tag} | {t.n_samples} | {t.baseline:.3f} | {t.candidate:.3f} "
                f"| {t.delta:+.3f} |"
                for t in group.tags
            ]
        lines.append("")
    return "\n".join(lines) + "\n"


def _cell(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "..."


def gate_markdown(report: GateReport) -> str:
    """CI gate summary."""
    status = "PASSED" if report.passed else "FAILED"
    lines = [
        f"## LLMBench eval gate: {status}",
        "",
        f"Baseline `{report.baseline_run}` vs candidate `{report.candidate_run}`",
        "",
        *(
            [
                "> Mock provider: numbers are illustrative, not measurements of a real model; "
                "mock costs use fictional prices and nothing was billed.",
                "",
            ]
            if report.uses_mock
            else []
        ),
        "| gate | configuration | result | reason |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r.metric} | `{r.config}` | {'pass' if r.passed else '**FAIL**'} | {r.reason} |"
        for r in report.results
    ]
    return "\n".join(lines) + "\n\n" + comparison_markdown(report.comparison)
