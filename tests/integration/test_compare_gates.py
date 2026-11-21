from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from llmbench.errors import DatasetDriftError, LLMBenchError
from llmbench.metrics.engine import evaluate_run
from llmbench.runner import FakeClock, Runner
from llmbench.stats import Verdict
from llmbench.tracking import RunStore
from llmbench.tracking.bundle import export_bundle, import_bundle, read_bundle, write_bundle
from llmbench.tracking.compare import compare_runs, summarize_verdicts
from llmbench.tracking.drift import (
    DriftPoint,
    cusum_change_point,
    detect_drift,
    robust_z,
    series_from_store,
)
from llmbench.tracking.gates import Gate, GateConfig, check, evaluate_gate
from tests.helpers import make_config


def _model(accuracy: float, seed: int) -> list[dict[str, Any]]:
    params = {"accuracy": accuracy, "temperature": 0.7, "mock_seed": seed, "sample_spread": 1.0}
    return [{"id": "bot", "provider": "mock", "model": "bot", "params": params}]


async def _run(store: RunStore, tmp_path: Path, accuracy: float, seed: int, **kw: Any) -> str:
    cfg = make_config(
        tmp_path,
        n_samples=kw.pop("n_samples", 40),
        trials=3,
        models=_model(accuracy, seed),
        metrics=["exact_match", "f1", "consistency"],
        **kw,
    )
    outcome = await Runner(store, clock=FakeClock(), evaluator=evaluate_run).run(
        cfg, use_cache=False
    )
    return outcome.run_id


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(f"sqlite:///{tmp_path / 'g.db'}")


async def test_gate_fails_on_seeded_regression(tmp_path: Path, store: RunStore) -> None:
    base = await _run(store, tmp_path, 0.85, seed=0)
    bad = await _run(store, tmp_path, 0.55, seed=1)
    report = check(store, base, bad, GateConfig.default())
    assert not report.passed
    assert report.exit_code == 1
    (result,) = report.results
    assert "regressed" in result.reason
    cmp = result.comparison
    assert cmp is not None
    assert cmp.verdict is Verdict.REGRESSED
    assert cmp.baseline is not None
    assert cmp.n_samples == 40
    assert cmp.n_trials_baseline == 120


@pytest.mark.slow
async def test_gate_passes_on_noise_only_reruns(tmp_path: Path, store: RunStore) -> None:
    """Same model re-run with different sampling noise: the gate should almost never fire."""
    base = await _run(store, tmp_path, 0.8, seed=0)
    failures = 0
    for seed in range(1, 11):
        candidate = await _run(store, tmp_path, 0.8, seed=seed)
        failures += not check(store, base, candidate, GateConfig.default()).passed
    assert failures <= 1  # documented false-positive rate ~alpha (0.05)


async def test_compare_runs_details(tmp_path: Path, store: RunStore) -> None:
    base = await _run(store, tmp_path, 0.85, seed=0, n_samples=20)
    cand = await _run(store, tmp_path, 0.5, seed=3, n_samples=20)
    comparison = compare_runs(store, base, cand, max_examples=3)
    (group,) = comparison.groups
    assert group.baseline_config == "bot/v1"
    names = {m.metric for m in group.metrics}
    assert {"pass", "exact_match", "f1", "pass_at_k", "latency_ms"} <= names
    assert "queue_wait_ms" not in names
    assert len(group.regressions) <= 3
    assert group.regressions
    assert group.regressions[0].baseline_output
    assert {t.tag for t in group.tags} == {"math", "arith"}
    assert group.difficulty[0].tag == "medium"
    assert all(m.p_adjusted is not None for m in group.metrics if m.p_value is not None)
    assert sum(summarize_verdicts(comparison).values()) == len(group.metrics)
    assert group.metric("nope") is None
    explicit = compare_runs(
        store, base, cand, pairs=[("bot/v1", "bot/v1")], metrics=["pass", "ghost"]
    )
    assert any("ghost" in n for n in explicit.notes)
    with pytest.raises(LLMBenchError, match="not found"):
        compare_runs(store, base, cand, pairs=[("x/y", "bot/v1")])
    with pytest.raises(LLMBenchError, match="model_id/prompt_id"):
        compare_runs(store, base, cand, pairs=[("bad", "bot/v1")])


async def test_dataset_drift_is_refused_unless_allowed(tmp_path: Path, store: RunStore) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    base = await _run(store, tmp_path / "a", 0.8, seed=0, n_samples=10)
    cand = await _run(store, tmp_path / "b", 0.8, seed=0, n_samples=12)
    with pytest.raises(DatasetDriftError):
        compare_runs(store, base, cand)
    allowed = compare_runs(store, base, cand, allow_dataset_drift=True)
    assert not allowed.dataset_match
    assert allowed.notes


async def test_unmatched_configs_pairing(tmp_path: Path, store: RunStore) -> None:
    a = await _run(store, tmp_path, 0.8, seed=0, n_samples=8)
    cfg = make_config(tmp_path, n_samples=8, trials=3, metrics=["exact_match"])
    two = (await Runner(store, clock=FakeClock(), evaluator=evaluate_run).run(cfg)).run_id
    with pytest.raises(LLMBenchError, match="share no"):
        compare_runs(store, a, two)
    single = make_config(
        tmp_path,
        n_samples=8,
        trials=3,
        metrics=["exact_match"],
        models=[{"id": "other", "provider": "mock", "params": {"temperature": 0.7}}],
    )
    other = (await Runner(store, clock=FakeClock(), evaluator=evaluate_run).run(single)).run_id
    paired = compare_runs(store, a, other)
    assert paired.groups[0].candidate_config == "other/v1"


def test_gate_config_and_edge_cases(tmp_path: Path) -> None:
    f = tmp_path / "gates.yaml"
    f.write_text(
        "alpha: 0.01\ngates:\n  - {metric: f1, max_drop: 0.05}\n"
        "  - {metric: latency_ms, direction: lower_is_better, max_drop: 50}\n"
    )
    cfg = GateConfig.from_yaml(f)
    assert cfg.alpha == 0.01
    assert cfg.gates[1].direction == "lower_is_better"
    with pytest.raises(Exception, match="not found"):
        GateConfig.from_yaml(tmp_path / "missing.yaml")
    missing = evaluate_gate(Gate(metric="x"), None, 0.05, "c")
    assert not missing.passed


async def test_gate_tolerance_and_relative_drop(tmp_path: Path, store: RunStore) -> None:
    base = await _run(store, tmp_path, 0.85, seed=0)
    bad = await _run(store, tmp_path, 0.55, seed=1)
    lenient = GateConfig(gates=[Gate(metric="pass", max_drop=0.9)])
    assert check(store, base, bad, lenient).passed
    relative = GateConfig(gates=[Gate(metric="pass", max_drop=0.0, max_relative_drop=0.99)])
    assert check(store, base, bad, relative).passed
    latency = GateConfig(gates=[Gate(metric="latency_ms", max_drop=10_000)])
    assert check(store, base, bad, latency).passed


async def test_bundle_round_trip(tmp_path: Path, store: RunStore) -> None:
    run_id = await _run(store, tmp_path, 0.8, seed=0, n_samples=6)
    path = write_bundle(store, run_id, tmp_path / "out" / "bundle.json")
    data = read_bundle(path)
    assert data["run"]["manifest"]["base_dir"] == "."
    other = RunStore(f"sqlite:///{tmp_path / 'other.db'}")
    imported = import_bundle(other, data, base_dir=tmp_path)
    assert imported == run_id
    assert len(other.trials(run_id)) == len(store.trials(run_id))
    assert len(other.metric_values(run_id)) == len(store.metric_values(run_id))
    assert check(other, run_id, run_id, GateConfig.default()).passed
    with pytest.raises(LLMBenchError, match="already exists"):
        import_bundle(other, data)
    assert import_bundle(other, data, replace=True) == run_id
    with pytest.raises(LLMBenchError, match="not an LLMBench"):
        import_bundle(other, {"format": "x"})
    bad = tmp_path / "bad.json"
    bad.write_text("[1]")
    with pytest.raises(LLMBenchError, match="not a run bundle"):
        read_bundle(bad)
    assert export_bundle(store, run_id)["trials"]


def test_drift_detection_on_synthetic_series() -> None:
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    stable = [0.80, 0.81, 0.79, 0.80, 0.82, 0.80, 0.81]
    shifted = [*stable, 0.62]

    def pts(vals: list[float]) -> list[DriftPoint]:
        return [
            DriftPoint(run_id=f"r{i}", created_at=t0 + dt.timedelta(days=i), value=v)
            for i, v in enumerate(vals)
        ]

    ok = detect_drift(pts(stable), metric="pass", model_id="m", prompt_id="p")
    assert not ok.latest_flagged
    assert ok.message == "no drift detected"
    bad = detect_drift(pts(shifted), metric="pass", model_id="m", prompt_id="p")
    assert bad.latest_flagged
    assert bad.latest_z is not None
    assert bad.latest_z < -3
    short = detect_drift(pts(stable[:3]), metric="pass", model_id="m", prompt_id="p")
    assert "need more" in short.message
    sustained = [0.8, 0.81, 0.79, 0.8, 0.8, 0.74, 0.73, 0.74, 0.72, 0.73, 0.73]
    assert cusum_change_point(sustained, window=5) is not None
    assert cusum_change_point(stable, window=5) is None
    assert cusum_change_point([1.0], window=5) is None
    gradual = detect_drift(
        pts(sustained), metric="pass", model_id="m", prompt_id="p", z_threshold=50
    )
    assert "sustained shift" in gradual.message
    assert robust_z([1, 1, 1], 1) == 0


async def test_series_from_store(tmp_path: Path, store: RunStore) -> None:
    for seed in range(3):
        await _run(store, tmp_path, 0.8, seed=seed, n_samples=6)
    points = series_from_store(
        store, metric="pass", model_id="bot", prompt_id="v1", name="unit-test"
    )
    assert len(points) == 3
    assert points[0].created_at <= points[-1].created_at
    assert (
        series_from_store(store, metric="pass", model_id="bot", prompt_id="v1", dataset_hash="zzz")
        == []
    )
    assert series_from_store(store, metric="pass", model_id="ghost", prompt_id="v1") == []
