from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.config import parse_experiment
from llmbench.config.settings import get_settings
from llmbench.errors import DatasetError
from llmbench.judges import LLMJudge, Rubric, build_judge, calibrate, load_labeled
from llmbench.judges.metrics import length_bias
from llmbench.providers import ErrorClass, MockProvider, ProviderError
from llmbench.stats.agreement import cohens_kappa, confusion_matrix


@pytest.fixture
def rubric(repo_root: Path) -> Rubric:
    return Rubric.from_yaml(repo_root / "rubrics" / "correctness.yaml")


def make_judge(rubric: Rubric, **params: object) -> LLMJudge:
    return LLMJudge(
        provider=MockProvider(dict(params)),
        provider_name="mock",
        model="judge",
        rubric=rubric,
        samples=int(params.pop("samples", 1)) if "samples" in params else 1,
    )


def test_rubric_validation(rubric: Rubric) -> None:
    assert rubric.normalise(5) == 1.0
    assert rubric.normalise(1) == 0.0
    _, user = rubric.pointwise_prompt("Q?", "A", "R")
    assert "JSON" in user
    assert "Reference answer" in user
    assert "impartial" in rubric.pairwise_prompt("Q?", "a", "b", None)[0]
    base = {"name": "r", "criteria": [{"name": "c", "description": "d"}]}
    with pytest.raises(ValueError, match="scale_min"):
        Rubric.model_validate({**base, "scale_min": 5, "scale_max": 1})
    with pytest.raises(ValueError, match="pass_threshold"):
        Rubric.model_validate({**base, "pass_threshold": 9})
    with pytest.raises(ValueError, match="anchor"):
        Rubric.model_validate({**base, "anchors": [{"score": 9, "description": "x"}]})


async def test_pointwise_and_self_consistency(rubric: Rubric) -> None:
    judge = make_judge(rubric)
    good = await judge.pointwise("Capital of France?", "Paris", "Paris")
    bad = await judge.pointwise("Capital of France?", "Bananas are yellow", "Paris")
    assert good.passed is True
    assert good.normalized is not None
    assert good.normalized >= 0.75
    assert bad.passed is False
    multi = LLMJudge(
        provider=MockProvider({"temperature": 0.9}),
        provider_name="mock",
        model="judge",
        rubric=rubric,
        samples=3,
        params={"temperature": 0.9},
    )
    verdict = await multi.pointwise("Q", "Paris", "Paris")
    assert len(verdict.sample_scores) == 3
    assert verdict.self_consistency is not None


async def test_malformed_judgments_are_repaired_or_reported(rubric: Rubric) -> None:
    broken = make_judge(rubric, malformed_rate=1.0)
    verdict = await broken.pointwise("Q", "Paris", "Paris")
    assert verdict.score is None
    assert verdict.error is not None
    assert "malformed" in verdict.error
    assert verdict.attempts == 3  # 1 call + 2 repairs
    flaky = make_judge(rubric, malformed_rate=0.5)
    results = [await flaky.pointwise("Q", f"Paris {i}", "Paris") for i in range(10)]
    assert any(r.score is not None and r.attempts > 1 for r in results)


class _Exploding(MockProvider):
    async def generate(self, request):  # type: ignore[no-untyped-def]
        raise ProviderError(ErrorClass.SERVER_ERROR, "boom")


async def test_provider_errors_are_reported(rubric: Rubric) -> None:
    judge = LLMJudge(provider=_Exploding(), provider_name="mock", model="j", rubric=rubric)
    verdict = await judge.pointwise("Q", "A", "R")
    assert verdict.error is not None
    assert "SERVER_ERROR" in verdict.error
    pair = await judge.pairwise("Q", "a", "b", "R")
    assert pair.winner is None


async def test_pairwise_position_swap(rubric: Rubric) -> None:
    fair = make_judge(rubric)
    verdict = await fair.pairwise("Capital?", "Paris", "Rome", "Paris")
    assert verdict.winner == "1"
    assert verdict.position_consistent is True
    reverse = await fair.pairwise("Capital?", "Rome", "Paris", "Paris")
    assert reverse.winner == "2"
    biased = make_judge(rubric, position_bias=1.0)
    flagged = await biased.pairwise("Capital?", "Rome", "Paris", "Paris")
    assert flagged.position_consistent is False
    assert flagged.winner == "tie"
    single = LLMJudge(
        provider=MockProvider(), provider_name="mock", model="j", rubric=rubric, swap_order=False
    )
    one = await single.pairwise("Capital?", "Paris", "Rome", "Paris")
    assert one.winner == "1"
    assert one.swapped_order is None


async def test_calibration_on_shipped_set(repo_root: Path, rubric: Rubric) -> None:
    examples = load_labeled(repo_root / "datasets" / "judge_calibration.jsonl")
    assert len(examples) >= 30
    report = await calibrate(make_judge(rubric), examples)
    assert report.n == len(examples)
    assert report.n_scored == report.n
    assert report.cohen_kappa is not None
    assert -1 <= report.cohen_kappa <= 1
    assert report.accuracy is not None
    assert sum(sum(row.values()) for row in report.confusion.values()) == report.n
    assert report.weighted_kappa_quadratic is not None


def test_load_labeled_errors(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text("\n")
    with pytest.raises(DatasetError, match="no labelled"):
        load_labeled(f)
    f.write_text('{"id": "1"}\n')
    with pytest.raises(DatasetError):
        load_labeled(f)


def test_kappa_known_values() -> None:
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert cohens_kappa(["a", "a"], ["a", "a"]) == 1.0
    # textbook example: 20/5/10/15 table -> kappa = 0.4
    truth = ["y"] * 25 + ["n"] * 25
    pred = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(truth, pred, ["y", "n"]) == pytest.approx(0.4)
    ordinal = cohens_kappa([1, 2, 3, 4], [1, 2, 3, 3], [1, 2, 3, 4], weights="quadratic")
    linear = cohens_kappa([1, 2, 3, 4], [1, 2, 3, 3], [1, 2, 3, 4], weights="linear")
    assert 0.8 < ordinal <= 1.0
    assert 0.5 < linear < 1.0
    with pytest.raises(ValueError, match="weights"):
        cohens_kappa([1], [1], weights="cubic")
    with pytest.raises(ValueError, match="no ratings"):
        cohens_kappa([], [], ["a"])
    with pytest.raises(ValueError, match="equal length"):
        confusion_matrix(["a"], [], ["a"])


def test_length_bias() -> None:
    assert length_bias([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert length_bias([1, 2], [0.1, 0.2]) is None


def test_build_judge(tmp_path: Path, repo_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = {
        "name": "x",
        "dataset": "d",
        "models": [{"id": "m", "provider": "mock", "model": "judge-model"}],
        "prompts": [{"id": "p", "text": "{{ input }}"}],
    }
    rubric_path = str(repo_root / "rubrics" / "correctness.yaml")
    settings = get_settings()
    assert build_judge(parse_experiment(base, tmp_path), settings) is None
    unavailable = {**base, "judge": {"provider": "openai", "model": "x", "rubric": rubric_path}}
    assert build_judge(parse_experiment(unavailable, tmp_path), settings) is None
    same = {**base, "judge": {"provider": "mock", "model": "judge-model", "rubric": rubric_path}}
    judge = build_judge(parse_experiment(same, tmp_path), settings)
    assert judge is not None
    assert judge.model == "judge-model"
