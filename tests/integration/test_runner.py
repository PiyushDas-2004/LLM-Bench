from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect

from llmbench.config import ModelSpec, Settings
from llmbench.errors import ConfigError, DatasetDriftError, RunNotFoundError
from llmbench.providers import ErrorClass, MockProvider, Provider
from llmbench.runner import FakeClock, Runner, estimate, prepare
from llmbench.tracking import RunStore
from tests.helpers import make_config


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(f"sqlite:///{tmp_path / 'runs.db'}")


def _outputs(store: RunStore, run_id: str) -> list[tuple[str, str, bool | None]]:
    return [(t.trial_key, t.output_text, t.passed) for t in store.trials(run_id)]


async def test_full_run_records_every_trial(tmp_path: Path, store: RunStore) -> None:
    cfg = make_config(tmp_path)
    outcome = await Runner(store, clock=FakeClock()).run(cfg)
    assert outcome.status == "completed"
    assert outcome.n_done == outcome.n_expected == 2 * 1 * 6 * 3
    trials = store.trials(outcome.run_id)
    assert len(trials) == 36
    assert all(t.latency_ms and t.latency_ms > 0 for t in trials)
    assert all(t.cost_usd is not None and t.cost_usd > 0 for t in trials)
    run = store.get_run(outcome.run_id)
    assert run.n_done == 36
    assert run.total_cost_usd is not None
    assert run.manifest["dataset"]["n_evaluated"] == 6
    assert run.manifest["prompts"]["v1"]
    acc = {m: sum(bool(t.passed) for t in trials if t.model_id == m) / 18 for m in ("m1", "m2")}
    assert acc["m1"] > acc["m2"]


async def test_interrupt_and_resume_matches_uninterrupted_run(
    tmp_path: Path, store: RunStore
) -> None:
    cfg = make_config(tmp_path, concurrency=2)
    reference = await Runner(store, clock=FakeClock()).run(cfg, use_cache=False)

    runner: Runner

    def stop_after_ten(progress: object) -> None:
        if getattr(progress, "done", 0) >= 10:
            runner.request_stop()

    runner = Runner(store, clock=FakeClock(), progress=stop_after_ten)
    partial = await runner.run(cfg, use_cache=False)
    assert partial.status == "interrupted"
    assert 10 <= partial.n_done < partial.n_expected
    resumed = await Runner(store, clock=FakeClock()).resume(partial.run_id, use_cache=False)
    assert resumed.status == "completed"
    assert resumed.n_done == reference.n_done
    assert _outputs(store, partial.run_id) == [
        (k, o, p) for k, o, p in _outputs(store, reference.run_id)
    ]
    keys = [t.trial_key for t in store.trials(partial.run_id)]
    assert len(keys) == len(set(keys))


async def test_cache_makes_rerun_free(tmp_path: Path, store: RunStore) -> None:
    cfg = make_config(tmp_path)
    first = await Runner(store, clock=FakeClock()).run(cfg)
    assert first.cache_hits == 0
    second = await Runner(store, clock=FakeClock()).run(cfg)
    assert second.cache_hits == second.n_done
    assert second.spent_usd == 0
    assert _outputs(store, first.run_id) == _outputs(store, second.run_id)
    assert all(t.cached for t in store.trials(second.run_id))
    third = await Runner(store, clock=FakeClock()).run(cfg, use_cache=False)
    assert third.cache_hits == 0


async def test_budget_abort_keeps_partial_results(tmp_path: Path, store: RunStore) -> None:
    cfg = make_config(tmp_path, budget={"max_requests": 7}, concurrency=1)
    outcome = await Runner(store, clock=FakeClock()).run(cfg, use_cache=False)
    assert outcome.status == "budget_exceeded"
    assert outcome.n_done == 7
    assert outcome.stop_reason
    assert "max_requests" in outcome.stop_reason
    assert store.get_run(outcome.run_id).status == "budget_exceeded"
    usd = make_config(tmp_path, budget={"max_usd": 0.0001}, concurrency=1)
    capped = await Runner(store, clock=FakeClock()).run(usd, use_cache=False)
    assert capped.status == "budget_exceeded"
    assert capped.n_done < capped.n_expected


async def test_fault_injection_taxonomy(tmp_path: Path, store: RunStore) -> None:
    params = {
        "failure_rate": 0.5,
        "error_mix": {"TIMEOUT": 1, "RATE_LIMITED": 1, "SERVER_ERROR": 1},
        "retry_after_s": 2.0,
        "temperature": 0.7,
    }
    cfg = make_config(
        tmp_path,
        models=[{"id": "flaky", "provider": "mock", "params": params}],
        circuit_breaker={"failure_threshold": 1000, "reset_timeout_s": 1},
    )
    clock = FakeClock()
    outcome = await Runner(store, clock=clock).run(cfg, use_cache=False)
    trials = store.trials(outcome.run_id)
    retried = [t for t in trials if t.attempts > 1]
    assert retried, "some calls must have been retried"
    failed = [t for t in trials if not t.ok]
    # p(fail 3 times) = 0.125: some trials exhaust retries; classes come from the mix
    assert {t.error_class for t in failed} <= {
        ErrorClass.TIMEOUT,
        ErrorClass.RATE_LIMITED,
        ErrorClass.SERVER_ERROR,
    }
    assert all(t.attempts == 3 for t in failed)
    assert 2.0 in clock.sleeps  # Retry-After honoured
    assert outcome.n_failed == len(failed)


class _AuthFailProvider(MockProvider):
    async def generate(self, request):  # type: ignore[no-untyped-def]
        from llmbench.providers import ProviderError

        raise ProviderError(ErrorClass.AUTH, "bad key")


async def test_non_retryable_errors_and_breaker_abandon(tmp_path: Path, store: RunStore) -> None:
    def factory(spec: ModelSpec, settings: Settings) -> Provider:
        return _AuthFailProvider()

    cfg = make_config(
        tmp_path,
        models=[{"id": "dead", "provider": "mock"}],
        circuit_breaker={"failure_threshold": 2, "reset_timeout_s": 1},
        concurrency=1,
    )
    outcome = await Runner(store, provider_factory=factory, clock=FakeClock()).run(cfg)
    trials = store.trials(outcome.run_id)
    assert all(t.error_class is ErrorClass.AUTH and t.attempts == 1 for t in trials)
    assert outcome.status == "partial"
    assert outcome.stop_reason
    assert "abandoned" in outcome.stop_reason
    assert len(trials) < outcome.n_expected


class _HangingProvider(MockProvider):
    async def generate(self, request):  # type: ignore[no-untyped-def]
        import asyncio

        await asyncio.sleep(10)
        raise AssertionError("unreachable")


async def test_timeouts_are_classified(tmp_path: Path, store: RunStore) -> None:
    cfg = make_config(
        tmp_path,
        n_samples=1,
        trials=1,
        models=[{"id": "slow", "provider": "mock"}],
        retry={"max_attempts": 1, "timeout_s": 0.01},
    )
    outcome = await Runner(store, provider_factory=lambda s, st: _HangingProvider()).run(cfg)
    (trial,) = store.trials(outcome.run_id)
    assert trial.error_class is ErrorClass.TIMEOUT


async def test_unavailable_models_are_skipped(tmp_path: Path, store: RunStore) -> None:
    cfg = make_config(
        tmp_path,
        models=[
            {"id": "real", "provider": "openai", "model": "<set me>"},
            {"id": "keyless", "provider": "openai", "model": "x"},
            {"id": "gem", "provider": "gemini", "model": "x"},
            {"id": "local", "provider": "openai_compat", "model": "x"},
            {"id": "weird", "provider": "nope"},
            {"id": "m", "provider": "mock"},
        ],
    )
    prepared = prepare(cfg)
    assert [m.id for m in prepared.models] == ["m"]
    assert set(prepared.skipped_models) == {"real", "keyless", "gem", "local", "weird"}
    with pytest.raises(ConfigError, match="cannot run"):
        prepare(cfg, strict_models=True)
    only_bad = make_config(tmp_path, models=[{"id": "real", "provider": "openai"}])
    with pytest.raises(ConfigError, match="no runnable models"):
        prepare(only_bad)


async def test_plan_estimate(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, sampling={"n": 4, "seed": 1})
    plan = estimate(prepare(cfg))
    assert plan.n_samples == 4
    assert plan.total_requests == 2 * 4 * 3
    assert plan.total_cost_upper is not None
    assert plan.total_cost_upper > 0
    assert plan.unpriced_models == []
    assert plan.pricing_as_of == "2026-01-01"
    no_price = make_config(tmp_path, pricing=None)
    assert estimate(prepare(no_price)).total_cost_upper is None
    missing = make_config(tmp_path, pricing="missing.yaml")
    with pytest.raises(ConfigError, match="pricing file"):
        prepare(missing)


def _rewrite(dataset: Path) -> None:
    dataset.write_text(dataset.read_text().replace("plus", "add"))


async def test_resume_refuses_dataset_drift(tmp_path: Path, store: RunStore) -> None:
    cfg = make_config(tmp_path, concurrency=1)
    runner: Runner

    def stop(progress: object) -> None:
        runner.request_stop()

    runner = Runner(store, clock=FakeClock(), progress=stop)
    outcome = await runner.run(cfg)
    _rewrite(Path(cfg.dataset))
    with pytest.raises(DatasetDriftError):
        await Runner(store, clock=FakeClock()).resume(outcome.run_id)


def test_store_run_resolution_and_baselines(tmp_path: Path, store: RunStore) -> None:
    with pytest.raises(RunNotFoundError):
        store.resolve_run("latest")
    for rid in ("abc-1", "abc-2"):
        store.create_run(
            run_id=rid,
            name="n",
            task="qa",
            dataset_name="d",
            dataset_hash="h",
            config={},
            manifest={},
            tags={},
            n_expected=1,
        )
    assert store.resolve_run("latest").id in {"abc-1", "abc-2"}
    assert store.resolve_run("abc-1").id == "abc-1"
    with pytest.raises(RunNotFoundError, match="ambiguous"):
        store.resolve_run("abc")
    with pytest.raises(RunNotFoundError):
        store.resolve_run("zzz")
    store.set_baseline("main", "abc-1", note="first")
    store.set_baseline("main", "abc-2")
    assert store.get_baseline("main").run_id == "abc-2"
    assert store.resolve_run("main").id == "abc-2"
    assert [b.name for b in store.list_baselines()] == ["main"]
    with pytest.raises(RunNotFoundError):
        store.get_baseline("nope")
    with pytest.raises(RunNotFoundError):
        store.update_run("nope", status="x")
    store.add_artifact("abc-1", "html", "r.html", "<html/>")
    assert len(store.list_runs(name="n")) == 2
    store.cache_put("k", "p", "m", {"text": "x"})
    assert store.cache_get("k") == {"text": "x"}
    assert store.cache_clear() == 1
    store.delete_run("abc-1")
    with pytest.raises(RunNotFoundError):
        store.get_run("abc-1")
    store.close()


def test_migrations_match_models(tmp_path: Path) -> None:
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from llmbench.tracking.db import Base

    store = RunStore(f"sqlite:///{tmp_path / 'm.db'}")
    with store.engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        tables = set(inspect(conn).get_table_names())
    assert diff == []
    assert {"runs", "trials", "metric_values", "judge_calls", "baselines"} <= tables


async def test_resume_refuses_changed_prompt(tmp_path: Path, store: RunStore) -> None:
    template = tmp_path / "p.j2"
    template.write_text("Q: {{ input }}")
    cfg = make_config(tmp_path, concurrency=1, prompts=[{"id": "v1", "template": str(template)}])
    runner: Runner

    def stop(progress: object) -> None:
        runner.request_stop()

    runner = Runner(store, clock=FakeClock(), progress=stop)
    outcome = await runner.run(cfg)
    _edit(template)
    with pytest.raises(ConfigError, match="prompt templates changed"):
        await Runner(store, clock=FakeClock()).resume(outcome.run_id)


def _edit(path: Path) -> None:
    path.write_text("Question: {{ input }}")


async def test_models_on_one_endpoint_share_a_circuit_breaker(
    tmp_path: Path, store: RunStore
) -> None:
    cfg = make_config(
        tmp_path,
        models=[{"id": "a", "provider": "mock"}, {"id": "b", "provider": "mock"}],
        circuit_breaker={"failure_threshold": 2, "reset_timeout_s": 1},
        concurrency=1,
    )
    outcome = await Runner(
        store, provider_factory=lambda s, st: _AuthFailProvider(), clock=FakeClock()
    ).run(cfg)
    assert outcome.stop_reason is not None
    assert "['a', 'b']" in outcome.stop_reason
