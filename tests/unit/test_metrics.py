from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import llmbench.metrics.engine  # noqa: F401  - registers judge/RAG metrics
from llmbench.datasets import Sample
from llmbench.domain import TrialRecord
from llmbench.errors import ConfigError, RegistryError
from llmbench.metrics import (
    METRICS,
    MetricContext,
    create_metric,
    expand_metric_names,
    pass_at_k,
    pass_hat_k,
    trial_metric,
)
from llmbench.metrics.lexical import detect_language
from llmbench.providers import ErrorClass


def trial(output: str = "Paris", sample: str = "s1", i: int = 0, **kw: Any) -> TrialRecord:
    base: dict[str, Any] = {
        "run_id": "r",
        "trial_key": f"m|p|{sample}|{i}",
        "model_id": "m",
        "prompt_id": "p",
        "sample_id": sample,
        "trial_index": i,
        "output_text": output,
        "latency_ms": 100.0,
    }
    base.update(kw)
    return TrialRecord(**base)


def ctx(*samples: Sample, **kw: Any) -> MetricContext:
    return MetricContext(samples={s.id: s for s in samples}, **kw)


QA = Sample(
    id="s1", input="Capital of France?", reference="Paris", metadata={"aliases": ["Paris, France"]}
)


async def score(
    name: str, trials: list[TrialRecord], c: MetricContext, **params: Any
) -> list[float | None]:
    result = await create_metric(name, **params).compute(trials, c)
    return [v.value for v in result.values]


def test_registry_and_bundles() -> None:
    names = set(METRICS.names())
    assert {
        "exact_match",
        "f1",
        "rouge_l",
        "json_schema",
        "pass_at_k",
        "latency_ms",
        "cost_usd",
        "injection_resistance",
        "llm_judge_correctness",
        "semantic_similarity",
    } <= names
    assert expand_metric_names(["f1", "consistency", "f1"])[:3] == [
        "f1",
        "exact_agreement",
        "semantic_consistency",
    ]
    with pytest.raises(RegistryError):
        create_metric("does_not_exist")


async def test_lexical_metrics() -> None:
    c = ctx(QA, Sample(id="s2", input="?"))
    trials = [
        trial("paris."),
        trial("Lyon"),
        trial("The answer is Paris", i=1),
        trial("x", sample="s2"),
    ]
    assert await score("exact_match", trials, c) == [1.0, 0.0, 0.0, None]
    f1 = await score("f1", trials, c)
    assert f1[0] == 1.0
    assert f1[1] == 0.0
    assert 0 < f1[2] < 1  # type: ignore[operator]
    assert await score("contains", trials, c) == [1.0, 0.0, 1.0, None]
    assert (await score("rouge_l", trials[:1], c)) == [1.0]
    failed = trial("", error_class=ErrorClass.TIMEOUT)
    assert await score("exact_match", [failed], c) == [0.0]
    assert await score("exact_match", [trial(sample="unknown")], c) == [None]


async def test_regex_length_language_pass() -> None:
    s = Sample(id="s1", input="?", metadata={"regex": r"\d{4}"})
    c = ctx(s)
    assert await score("regex", [trial("in 1969"), trial("never")], c) == [1.0, 0.0]
    assert await score("regex", [trial("abc")], c, pattern="b") == [1.0]
    assert await score("regex", [trial("abc")], ctx(Sample(id="s1", input="?"))) == [None]
    assert await score("length_ok", [trial("short"), trial("x" * 50)], c, max_chars=10) == [
        1.0,
        0.0,
    ]
    assert await score("length_ok", [trial("one two three")], c, max_words=2) == [0.0]
    en = "The cat is on the mat and it is happy with the sun"
    es = "El gato de la casa es muy bonito y los perros también"
    assert await score("language", [trial(en), trial(es), trial("ok")], c) == [1.0, 0.0, None]
    assert detect_language(es) == "es"
    assert detect_language("") is None
    assert detect_language("zzz qqq") is None
    assert await score("pass", [trial(passed=True), trial(passed=False)], c) == [1.0, 0.0]


async def test_structural_metrics(tmp_path: Path) -> None:
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema))
    ref = Sample(id="s1", input="?", reference={"name": "Ana", "email": None})
    c = ctx(ref, resolve_path=lambda p: tmp_path / p)
    good, bad, junk = trial('{"name": "ana", "email": null}'), trial('{"name": 3}'), trial("nope")
    assert await score("json_valid", [good, junk], c) == [1.0, 0.0]
    assert await score("json_schema", [good, bad, junk], c, schema="schema.json") == [1.0, 0.0, 0.0]
    assert await score("json_schema", [good], c, schema=schema) == [1.0]
    assert await score("json_schema", [good], c) == [None]
    with pytest.raises(ConfigError, match="not found"):
        await score("json_schema", [good], c, schema="missing.json")
    assert await score("field_accuracy", [good, bad, junk], c) == [1.0, 0.5, 0.0]
    parsed = trial("ignored", parsed_output={"name": "Ana", "email": "x"})
    assert await score("field_accuracy", [parsed], c) == [0.5]
    assert await score("field_accuracy", [good], ctx(QA)) == [None]


async def test_semantic_similarity() -> None:
    c = ctx(QA, Sample(id="s2", input="?"))
    values = await score(
        "semantic_similarity",
        [
            trial("Paris"),
            trial("bananas"),
            trial("x", sample="s2"),
            trial("", error_class=ErrorClass.AUTH),
        ],
        c,
    )
    assert values[0] == pytest.approx(1.0)
    assert values[1] < 0.5  # type: ignore[operator]
    assert values[2:] == [None, 0.0]
    with pytest.raises(ConfigError, match="bert-score"):
        await score("bertscore", [trial()], c)


def test_pass_at_k_known_values() -> None:
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 3) == 1.0
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)
    assert pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
    assert pass_hat_k(5, 5, 5) == 1.0
    with pytest.raises(ValueError, match="0 < k"):
        pass_at_k(3, 1, 4)
    with pytest.raises(ValueError, match="0 < k"):
        pass_hat_k(3, 1, 0)


@given(
    st.integers(1, 20).flatmap(
        lambda n: st.tuples(st.just(n), st.integers(0, n), st.integers(1, n))
    )
)
def test_pass_k_properties(args: tuple[int, int, int]) -> None:
    n, c, k = args
    at, hat = pass_at_k(n, c, k), pass_hat_k(n, c, k)
    eps = 1e-9
    assert 0.0 <= hat <= c / n + eps
    assert c / n <= at + eps
    assert at <= 1.0 + eps


async def test_consistency_metrics() -> None:
    c = ctx(QA)
    stable = [trial("Paris", i=i, passed=True) for i in range(4)]
    mixed = [
        trial(o, i=i, passed=o == "Paris") for i, o in enumerate(["Paris", "Lyon", "Paris", "Nice"])
    ]
    assert await score("exact_agreement", stable, c) == [1.0]
    assert await score("exact_agreement", mixed, c) == [0.5]
    assert await score("answer_entropy", stable, c) == [0.0]
    assert await score("answer_entropy", mixed, c) == [pytest.approx(1.5)]
    assert await score("flip_rate", stable, c) == [0.0]
    assert await score("flip_rate", mixed, c) == [1.0]
    assert await score("flip_rate", stable[:1], c) == [None]
    assert await score("pass_at_k", mixed, c, k=2) == [pytest.approx(1 - 1 / 6)]
    assert await score("pass_hat_k", mixed, c, k=2) == [pytest.approx(1 / 6)]
    assert await score("semantic_consistency", stable, c) == [pytest.approx(1.0)]
    assert await score("semantic_consistency", stable[:1], c) == [None]
    errors = [trial("", i=i, error_class=ErrorClass.TIMEOUT) for i in range(2)]
    assert await score("exact_agreement", errors, c) == [1.0]
    json_trials = [trial("x", i=i, parsed_output={"a": 1}) for i in range(2)]
    assert await score("exact_agreement", json_trials, c) == [1.0]
    assert await score("exact_agreement", [trial(sample="ghost")], c) == [None]


async def test_operational_metrics() -> None:
    c = ctx(QA)
    ok = trial(
        latency_ms=200.0,
        ttft_ms=50.0,
        output_tokens=30,
        cost_usd=0.01,
        attempts=3,
        queue_wait_ms=5.0,
    )
    bad = trial(error_class=ErrorClass.RATE_LIMITED, latency_ms=None, refused=False)
    weird = trial(malformed=True, refused=True)
    assert await score("failed", [ok, bad], c) == [0.0, 1.0]
    assert await score("malformed", [ok, bad, weird], c) == [0.0, None, 1.0]
    assert await score("refused", [ok, weird], c) == [0.0, 1.0]
    assert await score("retries", [ok, bad], c) == [2.0, 0.0]
    assert await score("latency_ms", [ok, bad], c) == [200.0, None]
    assert await score("ttft_ms", [ok], c) == [50.0]
    assert await score("tokens_per_second", [ok], c) == [pytest.approx(200.0)]
    assert await score("queue_wait_ms", [ok], c) == [5.0]
    assert await score("cost_usd", [ok], c) == [0.01]


async def test_safety_metrics() -> None:
    attack = Sample(
        id="a",
        input="?",
        metadata={
            "category": "instruction_override",
            "canary": "CANARY-1",
            "expected_behavior": "resist",
        },
    )
    benign = Sample(
        id="b", input="?", metadata={"category": "benign", "expected_behavior": "comply"}
    )
    other = Sample(id="o", input="?")
    c = ctx(attack, benign, other)
    leaked = trial("Sure. CANARY-1", sample="a")
    safe = trial("A summary.", sample="a")
    refused_attack = trial("I can't help", sample="a", refused=True)
    helped = trial("Use kill.", sample="b")
    refused = trial("I'm sorry", sample="b", refused=True)
    sys_leak = trial("token SYS-CANARY-5E1F", sample="b")
    assert await score("injection_resistance", [leaked, safe, helped], c) == [0.0, 1.0, None]
    assert await score("canary_leakage", [sys_leak, helped], c) == [1.0, 0.0]
    assert await score("over_refusal", [helped, refused, safe], c) == [0.0, 1.0, None]
    assert await score(
        "refusal_correctness", [leaked, refused_attack, helped, refused, trial(sample="o")], c
    ) == [0.0, 1.0, 1.0, 0.0, None]


async def test_trial_metric_decorator_and_result_helpers() -> None:
    @trial_metric("mentions_paris_test", kind="binary")
    def mentions(t: TrialRecord, s: Sample) -> float:
        """Mentions Paris."""
        return float("paris" in t.output_text.lower())

    try:
        metric = create_metric("mentions_paris_test")
        assert metric.description == "Mentions Paris."
        result = await metric.compute(
            [trial("Paris"), trial("Rome", i=1), trial("Paris", sample="s2")],
            ctx(QA, Sample(id="s2", input="?")),
        )
        assert result.groups() == [("m", "p")]
        assert result.per_sample("m", "p") == {"s1": [1.0, 0.0], "s2": [1.0]}
        assert result.aggregate("m", "p") == pytest.approx(0.75)
        assert result.aggregate("x", "y") is None
    finally:
        METRICS.unregister("mentions_paris_test")


def test_register_metric_requires_name() -> None:
    from llmbench.metrics import TrialMetric, register_metric

    class Nameless(TrialMetric):
        def score(self, trial, sample, ctx):  # type: ignore[no-untyped-def]
            return 1.0

    with pytest.raises(ValueError, match="name"):
        register_metric(Nameless)
