"""Adapter tests against mocked HTTP transports (no network, no keys)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from llmbench.config import ModelSpec, Settings
from llmbench.providers import (
    ErrorClass,
    GenerationParams,
    GenerationRequest,
    ProviderError,
    check_availability,
    create_provider,
)
from llmbench.providers._http import classify_transport_error, parse_retry_after
from llmbench.providers.gemini_provider import GeminiProvider
from llmbench.providers.openai_provider import OpenAIProvider

Handler = Callable[[httpx2.Request], httpx2.Response]


def _client(handler: Handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def _req(stream: bool = False, json_mode: bool = False) -> GenerationRequest:
    return GenerationRequest.from_prompt(
        "test-model",
        "Capital of France?",
        system="Be brief.",
        params=GenerationParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=16,
            seed=1,
            stop=("\n",),
            json_mode=json_mode,
            stream=stream,
        ),
    )


def _completion(
    content: str | None = "Paris", finish: str = "stop", refusal: str | None = None
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model-2026",
        "system_fingerprint": "fp_1",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
    }


# ----------------------------------------------------------------------------- OpenAI
async def test_openai_success_and_request_mapping() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx2.Response(200, json=_completion())

    provider = OpenAIProvider(api_key="sk-test-not-real", http_client=_client(handler))
    result = await provider.generate(_req(json_mode=True))
    assert result.text == "Paris"
    assert result.ok
    assert (result.input_tokens, result.output_tokens) == (12, 2)
    assert result.model == "test-model-2026"
    assert result.raw_response_id == "chatcmpl-1"
    body = seen["body"]
    assert body["max_completion_tokens"] == 16
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body["response_format"] == {"type": "json_object"}
    assert body["seed"] == 1
    assert body["stop"] == ["\n"]
    assert seen["auth"] == "Bearer sk-test-not-real"
    await provider.aclose()


async def test_openai_compat_uses_max_tokens() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=_completion())

    provider = OpenAIProvider(
        api_key="x",
        base_url="http://localhost:11434/v1",
        max_tokens_param="max_tokens",
        http_client=_client(handler),
    )
    await provider.generate(_req())
    assert seen["url"].startswith("http://localhost:11434/v1/chat/completions")
    assert seen["body"]["max_tokens"] == 16


@pytest.mark.parametrize(
    ("status", "headers", "expected", "retry_after"),
    [
        (429, {"retry-after": "3"}, ErrorClass.RATE_LIMITED, 3.0),
        (429, {"retry-after-ms": "1500"}, ErrorClass.RATE_LIMITED, 1.5),
        (401, {}, ErrorClass.AUTH, None),
        (403, {}, ErrorClass.AUTH, None),
        (400, {}, ErrorClass.INVALID_REQUEST, None),
        (404, {}, ErrorClass.INVALID_REQUEST, None),
        (500, {}, ErrorClass.SERVER_ERROR, None),
        (503, {}, ErrorClass.SERVER_ERROR, None),
        (504, {}, ErrorClass.TIMEOUT, None),
    ],
)
async def test_openai_error_taxonomy(
    status: int, headers: dict[str, str], expected: ErrorClass, retry_after: float | None
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status, headers=headers, json={"error": {"message": "nope", "type": "x"}}
        )

    provider = OpenAIProvider(api_key="x", http_client=_client(handler))
    with pytest.raises(ProviderError) as info:
        await provider.generate(_req())
    assert info.value.error_class is expected
    assert info.value.status_code == status
    assert info.value.retry_after_s == retry_after


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx2.ReadTimeout("slow"), ErrorClass.TIMEOUT),
        (httpx2.ConnectError("down"), ErrorClass.NETWORK),
    ],
)
async def test_openai_transport_errors(exc: Exception, expected: ErrorClass) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exc

    provider = OpenAIProvider(api_key="x", http_client=_client(handler))
    with pytest.raises(ProviderError) as info:
        await provider.generate(_req())
    assert info.value.error_class is expected


async def test_openai_content_filter_and_refusal() -> None:
    def filtered(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_completion(content="", finish="content_filter"))

    def refused(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json=_completion(content=None, refusal="I can't help with that.")
        )

    r1 = await OpenAIProvider(api_key="x", http_client=_client(filtered)).generate(_req())
    assert r1.error is ErrorClass.CONTENT_FILTERED
    r2 = await OpenAIProvider(api_key="x", http_client=_client(refused)).generate(_req())
    assert r2.error is ErrorClass.REFUSAL
    assert r2.error_message == "I can't help with that."


def _sse(events: list[dict[str, Any]]) -> bytes:
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode()


async def test_openai_streaming_measures_ttft() -> None:
    chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m-s"}
    events = [
        {
            **chunk,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Par"},
                    "finish_reason": None,
                }
            ],
        },
        {**chunk, "choices": [{"index": 0, "delta": {"content": "is"}, "finish_reason": "stop"}]},
        {
            **chunk,
            "choices": [],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
        },
    ]
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200, content=_sse(events), headers={"content-type": "text/event-stream"}
        )

    result = await OpenAIProvider(api_key="x", http_client=_client(handler)).generate(
        _req(stream=True)
    )
    assert result.text == "Paris"
    assert result.time_to_first_token_ms is not None
    assert (result.input_tokens, result.output_tokens) == (9, 2)
    assert result.finish_reason == "stop"
    assert seen["body"]["stream_options"] == {"include_usage": True}


async def test_openai_embeddings() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "e",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    provider = OpenAIProvider(api_key="x", http_client=_client(handler), embedding_model="emb")
    assert await provider.embed(["hi"]) == [[0.1, 0.2]]
    with pytest.raises(ValueError, match="embedding model"):
        await OpenAIProvider(api_key="x", http_client=_client(handler)).embed(["hi"])


# ----------------------------------------------------------------------------- Gemini
def _gemini_body(text: str = "Paris", finish: str = "STOP", **extra: Any) -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}
        ],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 1, "totalTokenCount": 8},
        "modelVersion": "gemini-test-001",
        "responseId": "resp-1",
        **extra,
    }


async def test_gemini_success_and_request_mapping() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=_gemini_body())

    provider = GeminiProvider(api_key="test-key", http_client=_client(handler))
    result = await provider.generate(_req(json_mode=True))
    assert result.text == "Paris"
    assert (result.input_tokens, result.output_tokens) == (7, 1)
    assert result.raw_response_id == "resp-1"
    assert result.model == "gemini-test-001"
    assert ":generateContent" in seen["url"]
    body = seen["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "Be brief."
    assert body["contents"][0]["parts"][0]["text"] == "Capital of France?"
    assert body["generationConfig"]["maxOutputTokens"] == 16
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    await provider.aclose()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, ErrorClass.RATE_LIMITED),
        (401, ErrorClass.AUTH),
        (400, ErrorClass.INVALID_REQUEST),
        (500, ErrorClass.SERVER_ERROR),
        (503, ErrorClass.SERVER_ERROR),
    ],
)
async def test_gemini_error_taxonomy(status: int, expected: ErrorClass) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status, json={"error": {"code": status, "message": "nope", "status": "X"}}
        )

    with pytest.raises(ProviderError) as info:
        await GeminiProvider(api_key="k", http_client=_client(handler)).generate(_req())
    assert info.value.error_class is expected


async def test_gemini_transport_error_and_safety_block() -> None:
    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down")

    with pytest.raises(ProviderError) as info:
        await GeminiProvider(api_key="k", http_client=_client(down)).generate(_req())
    assert info.value.error_class is ErrorClass.NETWORK

    def blocked(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_gemini_body(text="", finish="SAFETY"))

    result = await GeminiProvider(api_key="k", http_client=_client(blocked)).generate(_req())
    assert result.error is ErrorClass.CONTENT_FILTERED


async def test_gemini_streaming() -> None:
    events = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "Par"}]}}]},
        _gemini_body(text="is"),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = "".join(f"data: {json.dumps(e)}\r\n\r\n" for e in events)
        return httpx2.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    result = await GeminiProvider(api_key="k", http_client=_client(handler)).generate(
        _req(stream=True)
    )
    assert result.text == "Paris"
    assert result.time_to_first_token_ms is not None
    assert result.output_tokens == 1


async def test_gemini_embeddings() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"embeddings": [{"values": [0.5, 0.5]}]})

    provider = GeminiProvider(api_key="k", http_client=_client(handler), embedding_model="e")
    assert await provider.embed(["hi"]) == [[0.5, 0.5]]
    with pytest.raises(ValueError, match="embedding model"):
        await GeminiProvider(api_key="k", http_client=_client(handler)).embed(["x"])


# ---------------------------------------------------------------------- factory/helpers
def test_factory_and_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, OPENAI_API_KEY="sk-x", GEMINI_API_KEY="g-x")  # type: ignore[call-arg]
    for spec in (
        ModelSpec(id="o", provider="openai", model="m"),
        ModelSpec(id="g", provider="gemini", model="m"),
        ModelSpec(id="l", provider="openai_compat", model="m", base_url="http://localhost:1/v1"),
    ):
        assert check_availability(spec, settings).ok
        assert create_provider(spec, settings).name == spec.provider
    monkeypatch.setenv("MY_KEY", "abc")
    custom = ModelSpec(id="c", provider="openai", model="m", api_key_env="MY_KEY")
    assert check_availability(custom, Settings(_env_file=None)).ok  # type: ignore[call-arg]
    empty = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_provider(ModelSpec(id="o", provider="openai", model="m"), empty)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_provider(ModelSpec(id="g", provider="gemini", model="m"), empty)
    with pytest.raises(ValueError, match="base_url"):
        create_provider(ModelSpec(id="l", provider="openai_compat", model="m"), empty)


def test_retry_after_parsing() -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after({"Retry-After": "2"}) == 2.0
    assert parse_retry_after({"retry-after-ms": "bad"}) is None
    assert (
        parse_retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:10 GMT"}, now=1445412480.0)
        == 10.0
    )
    assert parse_retry_after({"retry-after": "garbage"}) is None
    assert parse_retry_after({"x": "1"}) is None


def test_transport_classification() -> None:
    import httpx

    assert classify_transport_error(httpx.ReadTimeout("x")) is ErrorClass.TIMEOUT
    assert classify_transport_error(httpx2.ConnectError("x")) is ErrorClass.NETWORK
    assert classify_transport_error(ValueError("x")) is None
