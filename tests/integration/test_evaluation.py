from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.errors import DatasetDriftError
from llmbench.metrics.engine import evaluate_run, resolve_metric_specs
from llmbench.runner import FakeClock, Runner
from llmbench.tracking import RunStore
from tests.helpers import make_config


def test_resolve_metric_specs(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, metrics=["f1", {"name": "consistency", "params": {"k": 2}}])
    specs = dict(resolve_metric_specs(cfg))
    assert specs["pass_at_k"] == {"k": 2}
    assert "latency_ms" in specs
    assert "cost_usd" in specs
    assert "pass" in specs
    default = dict(resolve_metric_specs(make_config(tmp_path, metrics=[])))
    assert "rouge_l" in default  # qa task defaults


async def test_run_with_evaluator_and_judge(tmp_path: Path, repo_root: Path) -> None:
    store = RunStore(f"sqlite:///{tmp_path / 'r.db'}")
    cfg = make_config(
        tmp_path,
        metrics=[
            "exact_match",
            "f1",
            "semantic_similarity",
            "llm_judge_correctness",
            "consistency",
        ],
        judge={
            "provider": "mock",
            "model": "judge",
            "rubric": str(repo_root / "rubrics" / "correctness.yaml"),
        },
    )
    outcome = await Runner(store, clock=FakeClock(), evaluator=evaluate_run).run(cfg)
    values = store.metric_values(outcome.run_id)
    names = {v.metric for v in values}
    assert {"exact_match", "llm_judge_correctness", "pass_at_k", "latency_ms", "failed"} <= names
    judge_values = [v for v in values if v.metric == "llm_judge_correctness"]
    assert len(judge_values) == outcome.n_done
    assert all(v.value is not None for v in judge_values)
    assert len(store.judge_calls(outcome.run_id)) == outcome.n_done
    assert store.judge_calls(None) == []
    # recomputing replaces instead of duplicating
    await evaluate_run(store, outcome.run_id)
    assert len(store.metric_values(outcome.run_id, "exact_match")) == outcome.n_done


async def test_evaluate_refuses_changed_dataset(tmp_path: Path) -> None:
    store = RunStore(f"sqlite:///{tmp_path / 'r.db'}")
    cfg = make_config(tmp_path)
    outcome = await Runner(store, clock=FakeClock()).run(cfg)
    _mutate(Path(cfg.dataset))
    with pytest.raises(DatasetDriftError):
        await evaluate_run(store, outcome.run_id)


def _mutate(path: Path) -> None:
    path.write_text(path.read_text().replace("plus", "and"))
