from __future__ import annotations

import json

import pytest

from llmbench.providers import (
    ErrorClass,
    GenerationParams,
    GenerationRequest,
    MockProvider,
    ProviderError,
)
from llmbench.providers.mock import ABSTAIN_TEXT, REFUSAL_TEXT


def _req(
    prompt: str = "Q?",
    trial: int = 0,
    temperature: float | None = 0.7,
    oracle: dict[str, object] | None = None,
    **meta: object,
) -> GenerationRequest:
    metadata: dict[str, object] = {"oracle": oracle or {"reference": "Paris", "sample_id": "s"}}
    metadata.update(meta)
    return GenerationRequest.from_prompt(
        "mock-model",
        prompt,
        system="sys",
        params=GenerationParams(temperature=temperature, stream=True),
        trial_index=trial,
        metadata=metadata,
    )


async def test_deterministic_across_instances() -> None:
    a = await MockProvider({"accuracy": 0.5}).generate(_req(trial=3))
    b = await MockProvider({"accuracy": 0.5}).generate(_req(trial=3))
    assert a.text == b.text
    assert a.latency_ms == b.latency_ms
    assert a.time_to_first_token_ms is not None
    assert a.input_tokens > 0
    assert a.output_tokens > 0
    assert a.tokens_per_second is not None
    assert a.raw_response_id is not None
    assert a.raw_response_id.startswith("mock-")


async def test_temperature_zero_is_identical_across_trials() -> None:
    mock = MockProvider({"accuracy": 0.5})
    texts = {(await mock.generate(_req(trial=t, temperature=0.0))).text for t in range(10)}
    assert len(texts) == 1


async def test_temperature_positive_varies_across_trials() -> None:
    mock = MockProvider({"accuracy": 0.5, "sample_spread": 0.0})
    texts = {(await mock.generate(_req(trial=t))).text for t in range(30)}
    assert len(texts) > 1


async def test_accuracy_controls_correctness() -> None:
    perfect = MockProvider({"accuracy": 1.0, "sample_spread": 0.0})
    hopeless = MockProvider({"accuracy": 0.0, "sample_spread": 0.0})
    assert (await perfect.generate(_req())).text == "Paris"
    assert (await hopeless.generate(_req())).text != "Paris"


async def test_failure_injection_and_retry_after() -> None:
    mock = MockProvider(
        {"failure_rate": 1.0, "error_mix": {"RATE_LIMITED": 1.0}, "retry_after_s": 2.5}
    )
    with pytest.raises(ProviderError) as info:
        await mock.generate(_req())
    assert info.value.error_class is ErrorClass.RATE_LIMITED
    assert info.value.retry_after_s == 2.5
    assert info.value.retryable


async def test_failures_differ_per_attempt() -> None:
    mock = MockProvider({"failure_rate": 0.5})
    outcomes = []
    for _ in range(20):
        try:
            await mock.generate(_req())
            outcomes.append(True)
        except ProviderError:
            outcomes.append(False)
    assert any(outcomes)
    assert not all(outcomes)


async def test_wrong_answers_for_labels_numbers_and_json() -> None:
    mock = MockProvider({"accuracy": 0.0, "sample_spread": 0.0})
    label = await mock.generate(_req(oracle={"reference": "a", "labels": ["a", "b"]}))
    assert label.text == "b"
    number = await mock.generate(_req(oracle={"reference": "42"}))
    assert number.text != "42"
    assert number.text.lstrip("-").isdigit()
    obj = await mock.generate(_req(oracle={"reference": {"k": "v"}}))
    assert json.loads(obj.text) != {"k": "v"}


async def test_correct_json_and_aliases() -> None:
    mock = MockProvider({"accuracy": 1.0, "sample_spread": 0.0})
    obj = await mock.generate(_req(oracle={"reference": {"k": "v"}}))
    assert json.loads(obj.text) == {"k": "v"}
    seen = {
        (await mock.generate(_req(trial=t, oracle={"reference": "7", "aliases": ["seven"]}))).text
        for t in range(40)
    }
    assert seen == {"7", "seven"}


async def test_malformed_and_refusal_rates() -> None:
    mock = MockProvider({"accuracy": 1.0, "malformed_rate": 1.0})
    out = await mock.generate(_req(oracle={"reference": {"k": "v"}}))
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.text)
    refuser = MockProvider({"refusal_rate": 1.0})
    assert (await refuser.generate(_req())).text == REFUSAL_TEXT


async def test_safety_behaviour() -> None:
    attack = {"category": "instruction_override", "canary": "CANARY-1", "sample_id": "x"}
    leaky = MockProvider({"injection_susceptibility": 1.0})
    safe = MockProvider({"injection_susceptibility": 0.0})
    assert "CANARY-1" in (await leaky.generate(_req(oracle=attack))).text
    assert "CANARY-1" not in (await safe.generate(_req(oracle=attack))).text
    leak = {**attack, "category": "system_prompt_leak"}
    assert "can't share" in (await safe.generate(_req(oracle=leak))).text
    benign = {"category": "benign", "reference": "ok", "sample_id": "y"}
    over = MockProvider({"refusal_rate": 1.0})
    assert (await over.generate(_req(oracle=benign))).text == REFUSAL_TEXT


async def test_rag_behaviour() -> None:
    ev = {"text": "The pump station opened in 1972."}
    prompt = "Context:\n[1] Unrelated text.\n[2] The pump station opened in 1972.\nQuestion: q"
    oracle = {"reference": "1972", "evidence": [ev], "answerable": True, "sample_id": "r"}
    good = MockProvider({"accuracy": 1.0, "sample_spread": 0.0})
    assert (await good.generate(_req(prompt=prompt, oracle=oracle))).text == "1972 [2]"
    missing = "Context:\n[1] Unrelated text.\nQuestion: q"
    assert (await good.generate(_req(prompt=missing, oracle=oracle))).text == ABSTAIN_TEXT
    unanswerable = {"reference": "UNANSWERABLE", "evidence": [], "answerable": False}
    assert (await good.generate(_req(oracle=unanswerable))).text == ABSTAIN_TEXT
    bad = MockProvider({"accuracy": 0.0, "sample_spread": 0.0})
    assert (await bad.generate(_req(oracle=unanswerable))).text.endswith("[1]")
    assert (await bad.generate(_req(prompt=missing, oracle=oracle))).text.endswith("[1]")


async def test_judge_mode() -> None:
    mock = MockProvider({})
    point = {"candidate": "Paris", "reference": "Paris", "scale_min": 1, "scale_max": 5}
    out = json.loads((await mock.generate(_req(mock_judge=point))).text)
    assert out["score"] >= 4
    pair = {"mode": "pairwise", "candidate_a": "Paris", "candidate_b": "Rome", "reference": "Paris"}
    assert json.loads((await mock.generate(_req(mock_judge=pair))).text)["winner"] == "A"
    broken = MockProvider({"malformed_rate": 1.0})
    with pytest.raises(json.JSONDecodeError):
        json.loads((await broken.generate(_req(mock_judge=point))).text)


async def test_embed_and_close() -> None:
    mock = MockProvider()
    vectors = await mock.embed(["a b", "a b"])
    assert vectors[0] == vectors[1]
    await mock.aclose()


async def test_time_scale_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr("llmbench.providers.mock.asyncio.sleep", fake_sleep)
    await MockProvider({"time_scale": 0.5, "latency_sigma": 0.0}).generate(_req())
    assert slept == [pytest.approx(0.2)]
