"""The ``llmbench`` command-line interface."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from llmbench import __version__
from llmbench.config.loader import load_experiment
from llmbench.config.models import ExperimentConfig, JudgeSpec, ModelSpec, PromptSpec
from llmbench.config.settings import Settings, get_settings
from llmbench.errors import LLMBenchError
from llmbench.metrics.base import METRICS, create_metric
from llmbench.metrics.engine import config_from_run, evaluate_run, resolve_metric_specs
from llmbench.observability import configure_logging
from llmbench.plugins.registry import load_plugin_modules
from llmbench.providers.registry import PROVIDERS
from llmbench.reporting import (
    comparison_markdown,
    export_run,
    gate_markdown,
    render_html,
    run_markdown,
    summarize_run,
)
from llmbench.reporting.tables import (
    comparison_tables,
    gate_table,
    plan_table,
    runs_table,
    summary_table,
)
from llmbench.runner.executor import Runner, RunOutcome, RunProgress
from llmbench.runner.plan import estimate
from llmbench.runner.prepare import prepare
from llmbench.tasks.base import TASKS
from llmbench.tracking.bundle import import_bundle, read_bundle
from llmbench.tracking.compare import compare_runs
from llmbench.tracking.drift import detect_drift, series_from_store
from llmbench.tracking.gates import GateConfig, check
from llmbench.tracking.store import RunStore

app = typer.Typer(
    name="llmbench",
    help="Reproducible evaluation, benchmarking and regression testing for LLM and RAG systems.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    add_completion=False,
)
baseline_app = typer.Typer(help="Manage named baselines.", no_args_is_help=True)
judge_app = typer.Typer(help="LLM-as-a-judge utilities.", no_args_is_help=True)
rag_app = typer.Typer(help="RAG evaluation.", no_args_is_help=True)
cache_app = typer.Typer(help="Response cache.", no_args_is_help=True)
app.add_typer(baseline_app, name="baseline")
app.add_typer(judge_app, name="judge")
app.add_typer(rag_app, name="rag")
app.add_typer(cache_app, name="cache")

console = Console()
err = Console(stderr=True)

ConfigArg = Annotated[
    Path, typer.Argument(help="Experiment YAML file.", exists=True, dir_okay=False)
]
RunArg = Annotated[str, typer.Argument(help="Run id, unique id prefix, baseline name or 'latest'.")]
OutputOpt = Annotated[Path | None, typer.Option("--output", "-o", help="Write to this file.")]


class _State:
    db_url: str | None = None


def _settings() -> Settings:
    return get_settings()


def _store() -> RunStore:
    return RunStore(_State.db_url or _settings().db_url)


def _fail(message: str, code: int = 1) -> typer.Exit:
    err.print(f"[bold red]error:[/] {message}")
    return typer.Exit(code)


def _write_or_print(text: str, output: Path | None) -> None:
    if output is None:
        console.print(text, markup=False, highlight=False, soft_wrap=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    err.print(f"wrote {output}")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"llmbench {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    db: Annotated[str | None, typer.Option("--db", help="SQLAlchemy URL of the run store.")] = None,
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="DEBUG, INFO, WARNING...")
    ] = None,
    log_format: Annotated[str | None, typer.Option("--log-format", help="console or json.")] = None,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Global options."""
    settings = _settings()
    _State.db_url = db
    configure_logging(
        level=log_level or settings.log_level,
        fmt=log_format or settings.log_format,
        redact_prompts=settings.redact_prompts,
        otel=settings.otel_enabled,
    )
    try:
        load_plugin_modules(settings.plugins)
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc


# --------------------------------------------------------------------------- init
def _copy_tree(src: Traversable, dest: Path) -> list[Path]:
    written: list[Path] = []
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            written += _copy_tree(item, target)
        elif not item.name.endswith((".py", ".pyc")):
            target.write_bytes(item.read_bytes())
            written.append(target)
    return written


@app.command()
def init(
    directory: Annotated[Path, typer.Argument(help="Where to create the project.")] = Path(),
    force: Annotated[bool, typer.Option(help="Overwrite existing files.")] = False,
) -> None:
    """Scaffold an example project (dataset, prompts, rubric, pricing, gates, experiment)."""
    if (directory / "experiment.yaml").exists() and not force:
        raise _fail(f"{directory / 'experiment.yaml'} exists; use --force to overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    written = _copy_tree(resources.files("llmbench").joinpath("scaffold"), directory)
    console.print(f"Created {len(written)} files in [bold]{directory}[/]. Next:")
    console.print(f"  cd {directory} && llmbench run experiment.yaml")


# ----------------------------------------------------------------------- validate
@app.command()
def validate(config: ConfigArg) -> None:
    """Validate a config, its dataset, prompts, metrics, judge and pricing."""
    try:
        cfg = load_experiment(config)
        prepared = prepare(cfg, _settings())
        for name, params in resolve_metric_specs(cfg):
            create_metric(name, **params)
        if cfg.judge is not None and not cfg.resolve_path(cfg.judge.rubric).is_file():
            raise LLMBenchError(f"judge rubric not found: {cfg.judge.rubric}")
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    table = Table(title=f"{cfg.name}: valid", header_style="bold", show_header=False)
    table.add_row(
        "dataset",
        f"{prepared.dataset.name} ({len(prepared.samples)} samples, "
        f"hash {prepared.evaluated_hash[:12]})",
    )
    table.add_row("task", prepared.task.name)
    table.add_row(
        "prompts", ", ".join(f"{k} ({v.content_hash[:8]})" for k, v in prepared.prompts.items())
    )
    table.add_row("models", ", ".join(m.id for m in prepared.models))
    table.add_row("metrics", ", ".join(n for n, _ in resolve_metric_specs(cfg)))
    table.add_row(
        "pricing", f"as of {prepared.pricing.as_of}" if prepared.pricing.models else "none"
    )
    console.print(table)
    for model_id, reason in prepared.skipped_models.items():
        err.print(f"[yellow]skipped[/] {model_id}: {reason}")


# --------------------------------------------------------------------------- plan
@app.command()
def plan(config: ConfigArg) -> None:
    """Print the grid size and an upper-bound token and cost estimate."""
    try:
        prepared = prepare(load_experiment(config), _settings())
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    estimate_ = estimate(prepared)
    console.print(plan_table(estimate_))
    total = estimate_.total_cost_upper
    console.print(
        f"Estimated cost (upper bound): {'unknown' if total is None else f'${total:.4f}'}"
    )
    for model_id, reason in estimate_.skipped_models.items():
        err.print(f"[yellow]skipped[/] {model_id}: {reason}")


# ---------------------------------------------------------------------------- run
def _parse_tags(tags: list[str]) -> dict[str, str]:
    out = {}
    for tag in tags:
        key, sep, value = tag.partition("=")
        if not sep or not key:
            raise _fail(f"--tag must look like key=value, got {tag!r}")
        out[key] = value
    return out


async def _execute(runner: Runner, coro: Any) -> RunOutcome:
    loop = asyncio.get_running_loop()
    # add_signal_handler is unavailable on Windows and outside the main thread.
    with contextlib.suppress(NotImplementedError, RuntimeError):
        loop.add_signal_handler(signal.SIGINT, runner.request_stop)
    try:
        result: RunOutcome = await coro
    finally:
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.remove_signal_handler(signal.SIGINT)
    return result


def _progress_runner(store: RunStore, total_hint: int) -> tuple[Runner, Progress]:
    progress = Progress(
        TextColumn("[bold]running"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(
            "failed {task.fields[failed]} | cached {task.fields[cached]} | "
            "${task.fields[spent]:.4f}"
        ),
        TimeElapsedColumn(),
        console=err,
        transient=True,
    )
    task_id = progress.add_task("run", total=total_hint, failed=0, cached=0, spent=0.0)

    def update(p: RunProgress) -> None:
        progress.update(
            task_id,
            completed=p.done,
            total=p.total,
            failed=p.failed,
            cached=p.cache_hits,
            spent=p.spent_usd,
        )

    runner = Runner(store, _settings(), evaluator=evaluate_run, progress=update)
    return runner, progress


def _finish(store: RunStore, outcome: RunOutcome, report: Path | None) -> None:
    style = "green" if outcome.status == "completed" else "yellow"
    manifest = store.get_run(outcome.run_id).manifest
    all_mock = all(m.get("provider") == "mock" for m in manifest.get("models", []))
    spent = (
        f"simulated cost ${outcome.spent_usd:.4f} (fictional mock prices; nothing billed)"
        if all_mock
        else f"spent ${outcome.spent_usd:.4f}"
    )
    console.print(
        f"[{style}]{outcome.status}[/] run [bold]{outcome.run_id}[/]: {outcome.n_done}/"
        f"{outcome.n_expected} trials, {outcome.n_failed} failed, {outcome.cache_hits} cache hits, "
        f"{spent}"
    )
    if outcome.stop_reason:
        console.print(f"stop reason: {outcome.stop_reason}")
    if outcome.n_done:
        summary = summarize_run(store, outcome.run_id)
        console.print(summary_table(summary))
        if report is not None:
            _write_or_print(render_html(summary), report)
    if outcome.status == "interrupted":
        console.print(f"resume with: llmbench resume {outcome.run_id}")


@app.command()
def run(
    config: ConfigArg,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the cost confirmation.")] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Ignore the response cache.")
    ] = False,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Run tag key=value (repeatable).")
    ] = None,
    strict_models: Annotated[
        bool, typer.Option("--strict-models", help="Fail instead of skipping unavailable models.")
    ] = False,
    report: Annotated[
        Path | None, typer.Option("--report", help="Also write an HTML report.")
    ] = None,
) -> None:
    """Run an experiment (results are checkpointed after every trial)."""
    try:
        cfg = load_experiment(config)
        prepared = prepare(cfg, _settings(), strict_models=strict_models)
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    estimate_ = estimate(prepared)
    total = estimate_.total_cost_upper or 0.0
    if total > cfg.cost_confirm_threshold_usd and not yes:
        console.print(plan_table(estimate_))
        raise _fail(
            f"estimated cost up to ${total:.2f} exceeds cost_confirm_threshold_usd="
            f"{cfg.cost_confirm_threshold_usd}; re-run with --yes to proceed",
            code=2,
        )
    for model_id, reason in prepared.skipped_models.items():
        err.print(f"[yellow]skipped[/] {model_id}: {reason}")
    store = _store()
    runner, progress = _progress_runner(store, prepared.grid_size)
    try:
        with progress:
            outcome = asyncio.run(
                _execute(
                    runner,
                    runner.run(
                        cfg,
                        use_cache=not no_cache,
                        tags=_parse_tags(tag or []),
                        strict_models=strict_models,
                    ),
                )
            )
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    _finish(store, outcome, report)
    if outcome.status not in {"completed", "interrupted"}:
        raise typer.Exit(3)


@app.command()
def resume(
    run_id: RunArg,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    report: Annotated[Path | None, typer.Option("--report")] = None,
) -> None:
    """Continue an interrupted run, executing only the missing trials."""
    store = _store()
    try:
        resolved = store.resolve_run(run_id).id
        runner, progress = _progress_runner(store, store.get_run(resolved).n_expected)
        with progress:
            outcome = asyncio.run(_execute(runner, runner.resume(resolved, use_cache=not no_cache)))
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    _finish(store, outcome, report)


@app.command()
def rerun(
    run_id: RunArg,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
) -> None:
    """Start a new run re-created from a stored run's manifest."""
    store = _store()
    try:
        cfg = config_from_run(store.resolve_run(run_id).manifest)
        prepared = prepare(cfg, _settings())
        total = estimate(prepared).total_cost_upper or 0.0
        if total > cfg.cost_confirm_threshold_usd and not yes:
            raise _fail(f"estimated cost up to ${total:.2f}; re-run with --yes", code=2)
        runner, progress = _progress_runner(store, prepared.grid_size)
        with progress:
            outcome = asyncio.run(_execute(runner, runner.run(cfg, use_cache=not no_cache)))
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    _finish(store, outcome, None)


# ------------------------------------------------------------------- inspection
@app.command("list")
def list_runs(
    limit: Annotated[int, typer.Option(help="Maximum runs to show.")] = 20,
    name: Annotated[str | None, typer.Option(help="Only runs of this experiment name.")] = None,
) -> None:
    """List recent runs."""
    console.print(runs_table(_store().list_runs(limit=limit, name=name)))


@app.command()
def show(run_id: RunArg) -> None:
    """Show a run's leaderboard with confidence intervals."""
    store = _store()
    try:
        summary = summarize_run(store, run_id)
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    console.print(summary_table(summary))
    if summary.stop_reason:
        console.print(f"stop reason: {summary.stop_reason}")


def _pairs(pair: list[str] | None) -> list[tuple[str, str]] | None:
    if not pair:
        return None
    out = []
    for item in pair:
        left, sep, right = item.partition("=")
        if not sep:
            raise _fail(f"--pair must look like base_model/prompt=cand_model/prompt, got {item!r}")
        out.append((left, right))
    return out


@app.command()
def compare(
    baseline: RunArg,
    candidate: RunArg,
    metric: Annotated[
        list[str] | None, typer.Option("--metric", "-m", help="Metrics to compare.")
    ] = None,
    pair: Annotated[
        list[str] | None, typer.Option("--pair", help="base/prompt=cand/prompt.")
    ] = None,
    allow_dataset_drift: Annotated[bool, typer.Option("--allow-dataset-drift")] = False,
    alpha: Annotated[float, typer.Option(help="Significance level.")] = 0.05,
    correction: Annotated[str, typer.Option(help="holm, bh or none.")] = "holm",
    fmt: Annotated[str, typer.Option("--format", help="table, md or json.")] = "table",
    output: OutputOpt = None,
) -> None:
    """Statistically compare two runs on aligned samples."""
    try:
        cmp = compare_runs(
            _store(),
            baseline,
            candidate,
            metrics=metric,
            pairs=_pairs(pair),
            allow_dataset_drift=allow_dataset_drift,
            alpha=alpha,
            correction=correction,
        )
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    if fmt == "json":
        _write_or_print(cmp.model_dump_json(indent=2), output)
    elif fmt == "md":
        _write_or_print(comparison_markdown(cmp), output)
    else:
        for note in cmp.notes:
            err.print(f"[yellow]note[/] {note}")
        for table in comparison_tables(cmp):
            console.print(table)
        for group in cmp.groups:
            if group.regressions:
                console.print(f"[bold]{group.candidate_config}: per-sample regressions[/]")
                for d in group.regressions[:5]:
                    console.print(
                        f"  {d.sample_id}: {d.baseline_score:.2f} -> {d.candidate_score:.2f} | "
                        f"baseline: {d.baseline_output[:60]!r} | "
                        f"candidate: {d.candidate_output[:60]!r}"
                    )


@app.command()
def export(
    run_id: RunArg,
    fmt: Annotated[str, typer.Option("--format", help="json (bundle), csv or parquet.")] = "json",
    output: OutputOpt = None,
) -> None:
    """Export a run (JSON bundle, or flat CSV / Parquet trial table)."""
    if fmt not in {"json", "csv", "parquet"}:
        raise _fail("--format must be json, csv or parquet")
    store = _store()
    try:
        run_row = store.resolve_run(run_id)
        path = output or Path("reports") / f"{run_row.id}.{fmt}"
        written = export_run(store, run_row.id, path, fmt)  # type: ignore[arg-type]
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    console.print(f"wrote {written}")


@app.command("import")
def import_(
    bundle: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Run bundle JSON.")],
    base_dir: Annotated[
        Path, typer.Option(help="Directory relative paths resolve against.")
    ] = Path(),
    replace: Annotated[
        bool, typer.Option(help="Replace an existing run with the same id.")
    ] = False,
    baseline_name: Annotated[
        str | None, typer.Option("--as-baseline", help="Also set as baseline.")
    ] = None,
) -> None:
    """Import a run bundle (e.g. a committed CI baseline)."""
    store = _store()
    try:
        run_id = import_bundle(store, read_bundle(bundle), base_dir=base_dir, replace=replace)
        if baseline_name:
            store.set_baseline(baseline_name, run_id)
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    console.print(
        f"imported {run_id}" + (f" as baseline '{baseline_name}'" if baseline_name else "")
    )


@app.command()
def report(
    run_id: RunArg,
    fmt: Annotated[str, typer.Option("--format", help="html, md or json.")] = "html",
    output: OutputOpt = None,
) -> None:
    """Render a self-contained HTML report, a Markdown summary, or the summary as JSON."""
    store = _store()
    try:
        summary = summarize_run(store, run_id)
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    if fmt == "md":
        _write_or_print(run_markdown(summary), output)
        return
    if fmt == "json":
        _write_or_print(summary.model_dump_json(indent=1), output)
        return
    if fmt != "html":
        raise _fail("--format must be html, md or json")
    html = render_html(summary)
    path = output or Path("reports") / f"{summary.run_id}.html"
    _write_or_print(html, path)


# --------------------------------------------------------------------- baselines
@baseline_app.command("set")
def baseline_set(
    run_id: RunArg,
    name: Annotated[str, typer.Option("--name", "-n", help="Baseline name.")] = "main",
    note: Annotated[str | None, typer.Option(help="Free-text note.")] = None,
) -> None:
    """Point a named baseline at a run."""
    store = _store()
    try:
        resolved = store.resolve_run(run_id).id
        store.set_baseline(name, resolved, note)
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    console.print(f"baseline '{name}' -> {resolved}")


@baseline_app.command("show")
def baseline_show(
    name: Annotated[str | None, typer.Argument(help="Baseline name.")] = None,
) -> None:
    """Show one baseline or list all of them."""
    store = _store()
    table = Table(title="Baselines", header_style="bold")
    for col in ("name", "run id", "note", "set at (UTC)"):
        table.add_column(col)
    try:
        rows = [store.get_baseline(name)] if name else store.list_baselines()
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    for b in rows:
        table.add_row(b.name, b.run_id, b.note or "", b.created_at.strftime("%Y-%m-%d %H:%M"))
    console.print(table)


@app.command("check")
def check_cmd(
    baseline: Annotated[
        str, typer.Option("--baseline", "-b", help="Baseline name or run id.")
    ] = "main",
    candidate: Annotated[str, typer.Option("--candidate", "-c", help="Candidate run.")] = "latest",
    gates: Annotated[Path | None, typer.Option("--gates", "-g", help="gates.yaml.")] = None,
    pair: Annotated[list[str] | None, typer.Option("--pair")] = None,
    allow_dataset_drift: Annotated[bool, typer.Option("--allow-dataset-drift")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write a Markdown report.")
    ] = None,
    json_output: Annotated[
        Path | None, typer.Option("--json", help="Also write the full gate report as JSON.")
    ] = None,
) -> None:
    """CI eval gate: exit 1 if any gated metric regresses beyond tolerance with significance."""
    try:
        gate_cfg = GateConfig.from_yaml(gates) if gates else GateConfig.default()
        result = check(
            _store(),
            baseline,
            candidate,
            gate_cfg,
            pairs=_pairs(pair),
            allow_dataset_drift=allow_dataset_drift or None,
        )
    except LLMBenchError as exc:
        raise _fail(str(exc), code=2) from exc
    console.print(gate_table(result))
    if json_output is not None:
        _write_or_print(result.model_dump_json(indent=1), json_output)
    if output is not None:
        _write_or_print(gate_markdown(result), output)
    raise typer.Exit(result.exit_code)


@app.command()
def drift(
    model: Annotated[str, typer.Option("--model", help="Model id.")],
    prompt: Annotated[str, typer.Option("--prompt", help="Prompt id.")],
    metric: Annotated[str, typer.Option("--metric", help="Metric to track.")] = "pass",
    name: Annotated[str | None, typer.Option("--name", help="Experiment name filter.")] = None,
    window: Annotated[int, typer.Option(help="Rolling baseline size.")] = 5,
    z: Annotated[float, typer.Option(help="Robust z-score threshold.")] = 3.0,
) -> None:
    """Detect drift in a metric over time for one (model, prompt) configuration."""
    store = _store()
    points = series_from_store(store, metric=metric, model_id=model, prompt_id=prompt, name=name)
    result = detect_drift(
        points, metric=metric, model_id=model, prompt_id=prompt, window=window, z_threshold=z
    )
    table = Table(title=f"{metric} for {model}/{prompt}", header_style="bold")
    for col in ("#", "run id", "created (UTC)", "value"):
        table.add_column(col)
    for i, p in enumerate(points, start=1):
        table.add_row(str(i), p.run_id, p.created_at.strftime("%Y-%m-%d %H:%M"), f"{p.value:.4f}")
    console.print(table)
    console.print(("[bold red]DRIFT[/] " if result.latest_flagged else "") + result.message)
    raise typer.Exit(1 if result.latest_flagged else 0)


# ------------------------------------------------------------------------- judge
@judge_app.command("calibrate")
def judge_calibrate(
    labeled: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Labelled JSONL.")],
    config: Annotated[
        Path | None, typer.Option("--config", help="Take the judge from this experiment.")
    ] = None,
    provider: Annotated[str, typer.Option(help="Judge provider (without --config).")] = "mock",
    model: Annotated[str | None, typer.Option(help="Judge model (without --config).")] = None,
    rubric: Annotated[Path, typer.Option(help="Rubric YAML (without --config).")] = Path(
        "rubrics/correctness.yaml"
    ),
    samples: Annotated[int, typer.Option(help="Judgments per item (self-consistency).")] = 1,
    attach: Annotated[str | None, typer.Option(help="Attach the report to this run.")] = None,
    output: OutputOpt = None,
) -> None:
    """Measure judge agreement with human labels (accuracy, Cohen's kappa, confusion matrix)."""
    from llmbench.judges import build_judge, calibrate, load_labeled

    store = _store()
    try:
        if config is not None:
            cfg = load_experiment(config)
        else:
            cfg = ExperimentConfig(
                name="calibration",
                dataset=str(labeled),
                models=[ModelSpec(id="placeholder", provider="mock")],
                prompts=[PromptSpec(id="p", text="{{ input }}")],
                judge=JudgeSpec(
                    provider=provider, model=model, rubric=str(rubric.resolve()), samples=samples
                ),
            )
        judge = build_judge(cfg, _settings())
        if judge is None:
            raise LLMBenchError(
                "judge is not configured or cannot run here (check keys / model name)"
            )
        report_ = asyncio.run(calibrate(judge.bind(store, None), load_labeled(labeled)))
        if attach:
            store.add_artifact(
                store.resolve_run(attach).id,
                "calibration",
                "calibration.json",
                report_.model_dump_json(exclude={"items"}),
            )
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    table = Table(
        title=f"Judge calibration: {report_.judge}", header_style="bold", show_header=False
    )
    table.add_row("examples scored", f"{report_.n_scored}/{report_.n}")
    for label, value in (
        ("accuracy", report_.accuracy),
        ("Cohen's kappa", report_.cohen_kappa),
        ("weighted kappa (quadratic)", report_.weighted_kappa_quadratic),
        ("Spearman (scores)", report_.spearman_scores),
        ("judge self-consistency", report_.mean_self_consistency),
    ):
        table.add_row(label, "n/a" if value is None else f"{value:.3f}")
    console.print(table)
    confusion = Table(title="confusion (rows = human, cols = judge)", header_style="bold")
    confusion.add_column("")
    confusion.add_column("pass", justify="right")
    confusion.add_column("fail", justify="right")
    for human, row in report_.confusion.items():
        confusion.add_row(human, str(row["pass"]), str(row["fail"]))
    console.print(confusion)
    if output is not None:
        _write_or_print(report_.model_dump_json(indent=2), output)


# --------------------------------------------------------------------------- rag
@rag_app.command("sweep")
def rag_sweep(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="RAG sweep YAML.")],
    output: OutputOpt = None,
    json_output: Annotated[
        Path | None, typer.Option("--json", help="Also write full results JSON.")
    ] = None,
    top: Annotated[int, typer.Option(help="Settings shown in the leaderboard.")] = 10,
) -> None:
    """Evaluate a grid of RAG settings and report which setting moves which metric."""
    from llmbench.rag import RAGSweepConfig, render_sweep_markdown, run_sweep

    try:
        report_ = asyncio.run(run_sweep(RAGSweepConfig.from_yaml(config), _settings()))
    except LLMBenchError as exc:
        raise _fail(str(exc)) from exc
    _write_or_print(render_sweep_markdown(report_, top=top), output)
    if json_output is not None:
        _write_or_print(report_.model_dump_json(indent=1), json_output)


# ------------------------------------------------------------------------- cache
@cache_app.command("clear")
def cache_clear() -> None:
    """Delete every cached response."""
    console.print(f"removed {_store().cache_clear()} cache entries")


# --------------------------------------------------------------------------- api
@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(envvar="PORT", help="Port (or $PORT).")] = 8000,
    insecure_no_auth: Annotated[
        bool,
        typer.Option(
            "--insecure-no-auth", help="Allow running without LLMBENCH_API_KEY (dev only)."
        ),
    ] = False,
) -> None:
    """Start the REST API (requires LLMBENCH_API_KEY unless --insecure-no-auth)."""
    import uvicorn

    from llmbench.api.app import create_app

    settings = _settings()
    if settings.api_key is None and not insecure_no_auth:
        raise _fail("set LLMBENCH_API_KEY (or pass --insecure-no-auth for local development)")
    application = create_app(settings, db_url=_State.db_url, require_auth=not insecure_no_auth)
    uvicorn.run(application, host=host, port=port, log_level="info")


# ----------------------------------------------------------------------- plugins
def _registry_table(title: str, rows: list[tuple[str, str]]) -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column("name")
    table.add_column("description")
    for name, desc in rows:
        table.add_row(name, desc)
    return table


@app.command("metrics")
def metrics_cmd() -> None:
    """List registered metrics (built-in and plugins)."""
    rows = []
    for name in METRICS.names():
        factory: Any = METRICS.get(name)
        direction = "higher" if getattr(factory, "higher_is_better", True) else "lower"
        rows.append((name, f"{getattr(factory, 'description', '')} ({direction} is better)"))
    console.print(_registry_table("Metrics", rows))


@app.command("tasks")
def tasks_cmd() -> None:
    """List registered task types."""
    console.print(_registry_table("Tasks", [(n, TASKS.get(n).description) for n in TASKS.names()]))


@app.command("providers")
def providers_cmd() -> None:
    """List registered providers and whether each can run here."""
    settings = _settings()
    status = {
        "mock": "always available (offline)",
        "openai": "ready" if settings.openai_api_key else "needs OPENAI_API_KEY",
        "gemini": "ready" if settings.gemini_api_key else "needs GEMINI_API_KEY",
        "openai_compat": "needs base_url (e.g. Ollama at http://localhost:11434/v1)",
    }
    console.print(
        _registry_table("Providers", [(n, status.get(n, "plugin")) for n in PROVIDERS.names()])
    )


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        err.print("interrupted")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
