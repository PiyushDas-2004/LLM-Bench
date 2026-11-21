"""Provider contract tests against real APIs.

Deselected by default (``-m 'not live'``). Run with::

    OPENAI_API_KEY=... LLMBENCH_LIVE_OPENAI_MODEL=<model> \
    GEMINI_API_KEY=... LLMBENCH_LIVE_GEMINI_MODEL=<model> \
    pytest -m live -o addopts=""

Model names come from the environment because LLMBench never hardcodes them.
Each test skips cleanly when its key or model variable is missing.
"""

from __future__ import annotations

import os

import pytest

from llmbench.config import ModelSpec, Settings
from llmbench.providers import ErrorClass, GenerationParams, GenerationRequest, ProviderError
from llmbench.providers.registry import create_provider

pytestmark = pytest.mark.live

CASES = [
    ("openai", "OPENAI_API_KEY", "LLMBENCH_LIVE_OPENAI_MODEL"),
    ("gemini", "GEMINI_API_KEY", "LLMBENCH_LIVE_GEMINI_MODEL"),
]


def _spec(provider: str, key_var: str, model_var: str) -> ModelSpec:
    if not os.environ.get(key_var) or not os.environ.get(model_var):
        pytest.skip(f"set {key_var} and {model_var} to run live {provider} tests")
    return ModelSpec(id=provider, provider=provider, model=os.environ[model_var])


def _request(model: str, stream: bool = False) -> GenerationRequest:
    return GenerationRequest.from_prompt(
        model,
        "Reply with the single word: pong",
        params=GenerationParams(temperature=0.0, max_tokens=256, stream=stream),
    )


@pytest.mark.parametrize(("provider", "key_var", "model_var"), CASES)
@pytest.mark.parametrize("stream", [False, True])
async def test_generate_contract(provider: str, key_var: str, model_var: str, stream: bool) -> None:
    spec = _spec(provider, key_var, model_var)
    client = create_provider(spec, Settings())
    try:
        result = await client.generate(_request(spec.model_name, stream=stream))
    finally:
        await client.aclose()
    assert result.ok, result.error_message
    assert "pong" in result.text.lower()
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.latency_ms > 0
    if stream:
        assert result.time_to_first_token_ms is not None


@pytest.mark.parametrize(("provider", "key_var", "model_var"), CASES)
async def test_auth_errors_are_classified(provider: str, key_var: str, model_var: str) -> None:
    spec = _spec(provider, key_var, model_var)
    os.environ["LLMBENCH_LIVE_BAD_KEY"] = "invalid-key-for-contract-test"
    bad = spec.model_copy(update={"api_key_env": "LLMBENCH_LIVE_BAD_KEY"})
    client = create_provider(bad, Settings())
    try:
        with pytest.raises(ProviderError) as info:
            await client.generate(_request(spec.model_name))
    finally:
        await client.aclose()
    assert info.value.error_class in {ErrorClass.AUTH, ErrorClass.INVALID_REQUEST}
