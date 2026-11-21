from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmbench.cli import app

runner = CliRunner()


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("LLMBENCH_DB_URL", url)
    monkeypatch.setenv("COLUMNS", "200")
    return url


def invoke(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def _small_config(tmp_path: Path, repo_root: Path, accuracy: float = 0.85, seed: int = 0) -> Path:
    path = tmp_path / f"exp-{accuracy}-{seed}.yaml"
    path.write_text(
        f"""
name: cli-test
dataset: {repo_root}/datasets/qa_basic.jsonl
task: qa
trials: 3
seed: 1
pricing: {repo_root}/examples/pricing.yaml
sampling: {{n: 20, seed: 3}}
models:
  - id: bot
    provider: mock
    model: mock-strong
    params: {{accuracy: {accuracy}, temperature: 0.7, mock_seed: {seed}, stream: true}}
prompts:
  - {{id: v1, template: {repo_root}/prompts/qa_v1.j2}}
metrics: [exact_match, f1, consistency, latency, cost]
"""
    )
    return path


def test_version_and_plugin_listings(db: str) -> None:
    code, out = invoke("--version")
    assert code == 0
    assert "llmbench" in out
    for command, expected in (
        ("metrics", "exact_match"),
        ("tasks", "extraction"),
        ("providers", "openai_compat"),
    ):
        code, out = invoke(command)
        assert code == 0
        assert expected in out


def test_init_scaffolds_a_runnable_project(
    db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "proj"
    code, out = invoke("init", str(target))
    assert code == 0
    assert (target / "experiment.yaml").is_file()
    assert (target / "datasets" / "qa_sample.jsonl").is_file()
    code, _ = invoke("init", str(target))
    assert code == 1  # refuses to overwrite
    monkeypatch.chdir(target)
    code, out = invoke("run", "experiment.yaml")
    assert code == 0, out
    assert "completed" in out


def test_validate_plan_run_show_report_export(db: str, tmp_path: Path, repo_root: Path) -> None:
    cfg = _small_config(tmp_path, repo_root)
    code, out = invoke("validate", str(cfg))
    assert code == 0, out
    assert "valid" in out
    code, out = invoke("plan", str(cfg))
    assert code == 0
    assert "60 requests" in out
    code, out = invoke("run", str(cfg), "--tag", "team=eval", "--report", str(tmp_path / "r.html"))
    assert code == 0, out
    assert "completed" in out
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")
    code, out = invoke("list")
    assert code == 0
    assert "cli-test" in out
    code, out = invoke("show", "latest")
    assert code == 0
    assert "bot/v1" in out
    code, out = invoke("report", "latest", "--format", "md")
    assert code == 0
    assert "## LLMBench: cli-test" in out
    code, _ = invoke("report", "latest", "--format", "pdf")
    assert code == 1
    for fmt in ("json", "csv", "parquet"):
        out_path = tmp_path / f"export.{fmt}"
        code, out = invoke("export", "latest", "--format", fmt, "--output", str(out_path))
        assert code == 0, out
        assert out_path.stat().st_size > 0
    code, _ = invoke("export", "latest", "--format", "xml")
    assert code == 1
    code, out = invoke("rerun", "latest", "--yes")
    assert code == 0, out


def test_gate_flow_with_baseline_and_bundle(db: str, tmp_path: Path, repo_root: Path) -> None:
    invoke("run", str(_small_config(tmp_path, repo_root, 0.9, 0)))
    code, out = invoke("baseline", "set", "latest", "--name", "main", "--note", "first")
    assert code == 0
    assert "baseline 'main'" in out
    code, out = invoke("baseline", "show")
    assert "main" in out
    code, _ = invoke("baseline", "show", "missing")
    assert code == 1
    bundle = tmp_path / "baseline.json"
    assert invoke("export", "main", "--output", str(bundle))[0] == 0
    gates = tmp_path / "gates.yaml"
    gates.write_text("gates:\n  - {metric: pass, max_drop: 0.02}\n")
    invoke("run", str(_small_config(tmp_path, repo_root, 0.9, 1)))
    code, out = invoke(
        "check", "--baseline", "main", "--gates", str(gates), "--output", str(tmp_path / "gate.md")
    )
    assert code == 0, out
    assert "PASSED" in out
    invoke("run", str(_small_config(tmp_path, repo_root, 0.35, 2)))
    code, out = invoke("check", "--baseline", "main", "--gates", str(gates))
    assert code == 1, out
    assert "FAIL" in out
    code, out = invoke("compare", "main", "latest")
    assert code == 0
    assert "regressed" in out
    code, out = invoke("compare", "main", "latest", "--format", "md", "--metric", "pass")
    assert "| pass |" in out
    code, out = invoke("compare", "main", "latest", "--format", "json")
    assert json.loads(out)["groups"][0]["metrics"]
    code, _ = invoke("compare", "main", "latest", "--pair", "nonsense")
    assert code == 1
    code, _ = invoke("check", "--baseline", "nope")
    assert code == 2
    other = f"sqlite:///{tmp_path / 'other.db'}"
    code, out = invoke("--db", other, "import", str(bundle), "--as-baseline", "main")
    assert code == 0, out
    code, _ = invoke("--db", other, "import", str(bundle))
    assert code == 1  # already exists


def test_resume_and_cost_confirmation(db: str, tmp_path: Path, repo_root: Path) -> None:
    cfg = _small_config(tmp_path, repo_root)
    text = cfg.read_text().replace("trials: 3", "trials: 3\ncost_confirm_threshold_usd: 0.0")
    cfg.write_text(text)
    code, out = invoke("run", str(cfg))
    assert code == 2
    assert "--yes" in out
    code, out = invoke("run", str(cfg), "--yes", "--no-cache")
    assert code == 0
    code, out = invoke("resume", "latest")
    assert code == 0, out
    assert "60/60" in out


def test_drift_command(db: str, tmp_path: Path, repo_root: Path) -> None:
    for seed in range(2):
        invoke("run", str(_small_config(tmp_path, repo_root, seed=seed)))
    code, out = invoke("drift", "--model", "bot", "--prompt", "v1", "--name", "cli-test")
    assert code == 0
    assert "need more" in out


def test_judge_calibrate_and_rag_sweep(db: str, tmp_path: Path, repo_root: Path) -> None:
    labeled = repo_root / "datasets" / "judge_calibration.jsonl"
    rubric = repo_root / "rubrics" / "correctness.yaml"
    code, out = invoke(
        "judge",
        "calibrate",
        str(labeled),
        "--rubric",
        str(rubric),
        "--output",
        str(tmp_path / "cal.json"),
    )
    assert code == 0, out
    assert "Cohen's kappa" in out
    assert json.loads((tmp_path / "cal.json").read_text())["cohen_kappa"] is not None
    code, out = invoke(
        "judge", "calibrate", str(labeled), "--provider", "openai", "--rubric", str(rubric)
    )
    assert code == 1
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        f"name: s\ncorpus: {repo_root}/datasets/rag_mini/corpus\n"
        f"questions: {repo_root}/datasets/rag_mini/questions.jsonl\n"
        f"prompt: {repo_root}/prompts/rag_v1.j2\n"
        "generator: {id: g, provider: mock}\nn_resamples: 100\n"
        "sweep: {top_k: [1, 3]}\n"
    )
    code, out = invoke("rag", "sweep", str(sweep), "--json", str(tmp_path / "sweep.json"))
    assert code == 0, out
    assert "Main effects" in out
    assert (tmp_path / "sweep.json").is_file()


def test_errors_are_reported_cleanly(db: str, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    for command in ("validate", "plan", "run"):
        code, out = invoke(command, str(bad))
        assert code == 1
        assert "error" in out
    code, out = invoke("show", "missing-run")
    assert code == 1
    code, _ = invoke("run", str(bad), "--tag", "novalue")
    assert code == 1
    code, out = invoke("cache", "clear")
    assert code == 0
    assert "removed" in out


def test_serve_requires_api_key(db: str) -> None:
    code, out = invoke("serve")
    assert code == 1
    assert "LLMBENCH_API_KEY" in out
