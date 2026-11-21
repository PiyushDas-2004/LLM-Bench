"""Every shipped example config validates and runs offline end to end."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.config import load_experiment
from llmbench.metrics.base import MetricResult
from llmbench.metrics.engine import evaluate_run
from llmbench.runner import FakeClock, Runner, estimate, prepare
from llmbench.tracking import RunStore


def _aggregate(
    store: RunStore, run_id: str, metric: str, model: str, prompt: str = "v1"
) -> float | None:
    values = store.metric_values(run_id, metric)
    result = MetricResult(
        metric=metric, higher_is_better=True, kind="continuous", level="trial", values=values
    )
    return result.aggregate(model, prompt)


async def _run(repo_root: Path, tmp_path: Path, name: str) -> tuple[RunStore, str]:
    store = RunStore(f"sqlite:///{tmp_path / 'ex.db'}")
    cfg = load_experiment(repo_root / "examples" / name)
    outcome = await Runner(store, clock=FakeClock(), evaluator=evaluate_run).run(cfg)
    assert outcome.status == "completed", outcome.stop_reason
    return store, outcome.run_id


async def test_safety_suite_separates_hardened_from_naive(repo_root: Path, tmp_path: Path) -> None:
    store, run_id = await _run(repo_root, tmp_path, "safety.yaml")
    hardened = _aggregate(store, run_id, "injection_resistance", "hardened")
    naive = _aggregate(store, run_id, "injection_resistance", "naive")
    assert hardened is not None
    assert naive is not None
    assert hardened > naive
    over_h = _aggregate(store, run_id, "over_refusal", "hardened")
    over_n = _aggregate(store, run_id, "over_refusal", "naive")
    assert over_h is not None
    assert over_n is not None
    assert over_n >= over_h
    leak = _aggregate(store, run_id, "canary_leakage", "naive")
    assert leak is not None
    assert leak > 0  # system-prompt-leak probes elicit SYS-CANARY-5E1F from the naive model
    trials = store.trials(run_id)
    assert all("SYS-CANARY-5E1F" in (t.system_prompt or "") for t in trials)


async def test_classification_temperature_affects_consistency(
    repo_root: Path, tmp_path: Path
) -> None:
    store, run_id = await _run(repo_root, tmp_path, "classification.yaml")
    greedy = _aggregate(store, run_id, "exact_agreement", "greedy")
    sampled = _aggregate(store, run_id, "exact_agreement", "sampled")
    assert greedy == 1.0
    assert sampled is not None
    assert sampled < 1.0
    assert _aggregate(store, run_id, "flip_rate", "greedy") == 0.0


async def test_extraction_schema_metrics(repo_root: Path, tmp_path: Path) -> None:
    store, run_id = await _run(repo_root, tmp_path, "extraction.yaml")
    schema = _aggregate(store, run_id, "json_schema", "mock-strong")
    assert schema is not None
    assert 0.5 < schema <= 1.0
    malformed = _aggregate(store, run_id, "malformed", "mock-strong")
    assert malformed is not None
    assert malformed < 0.5


async def test_quickstart_runs_with_judge(repo_root: Path, tmp_path: Path) -> None:
    store, run_id = await _run(repo_root, tmp_path, "quickstart.yaml")
    assert len(store.trials(run_id)) == 2 * 2 * 40 * 5
    judged = _aggregate(store, run_id, "llm_judge_correctness", "mock-strong")
    assert judged is not None
    assert store.judge_calls(run_id)
    ttft = _aggregate(store, run_id, "ttft_ms", "mock-strong")
    assert ttft is not None


@pytest.mark.parametrize(
    "name",
    [
        "experiment.yaml",
        "quickstart.yaml",
        "extraction.yaml",
        "classification.yaml",
        "safety.yaml",
        "gate/baseline.yaml",
        "gate/noise.yaml",
        "gate/regression.yaml",
    ],
)
def test_examples_validate_and_plan(repo_root: Path, name: str) -> None:
    plan = estimate(prepare(load_experiment(repo_root / "examples" / name)))
    assert plan.total_requests > 0


def test_multi_provider_example_skips_unavailable_models(repo_root: Path) -> None:
    prepared = prepare(load_experiment(repo_root / "examples" / "experiment.yaml"))
    assert [m.id for m in prepared.models] == ["mock"]
    assert set(prepared.skipped_models) == {"gpt-main", "gemini-main", "local-small"}
