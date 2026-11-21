"""OpenAI Chat Completions adapter, also used for OpenAI-compatible servers.

Uses the official ``openai`` SDK (install the ``openai`` extra). SDK retries are
disabled (``max_retries=0``) because the LLMBench runner owns retry policy, so
every attempt is visible in the trial record.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from llmbench.providers._http import classify_transport_error, parse_retry_after
from llmbench.providers.base import (
    ErrorClass,
    GenerationRequest,
    GenerationResult,
    ProviderError,
    classify_http_status,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def map_openai_error(exc: Exception) -> ProviderError:
    """Translate an ``openai`` SDK exception into a classified :class:`ProviderError`."""
    import openai

    if isinstance(exc, openai.APITimeoutError):
        return ProviderError(ErrorClass.TIMEOUT, str(exc))
    if isinstance(exc, openai.APIConnectionError):
        return ProviderError(ErrorClass.NETWORK, str(exc))
    if isinstance(exc, openai.ContentFilterFinishReasonError):
        return ProviderError(ErrorClass.CONTENT_FILTERED, str(exc))
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        retry_after = parse_retry_after(dict(exc.response.headers)) if status == 429 else None
        return ProviderError(
            classify_http_status(status),
            f"HTTP {status}: {exc.message}",
            retry_after_s=retry_after,
            status_code=status,
        )
    transport = classify_transport_error(exc)
    if transport is not None:
        return ProviderError(transport, f"{type(exc).__name__}: {exc}")
    return ProviderError(ErrorClass.SERVER_ERROR, f"{type(exc).__name__}: {exc}")


class OpenAIProvider:
    """Chat Completions adapter.

    Args:
        api_key: API key (read from the environment by the factory, never from config files).
        base_url: Override for OpenAI-compatible servers (Ollama, vLLM, LM Studio).
        name: Provider name recorded in results.
        timeout_s: Per-request timeout.
        max_tokens_param: ``max_completion_tokens`` for OpenAI, ``max_tokens`` for most
            OpenAI-compatible servers.
        http_client: Optional pre-built async HTTP client of the type the SDK expects
            (used by tests to inject a mock transport).
        embedding_model: Default embedding model for :meth:`embed`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        name: str = "openai",
        timeout_s: float = 60.0,
        max_tokens_param: str = "max_completion_tokens",
        http_client: Any = None,
        embedding_model: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("install the 'openai' extra: pip install 'llmbench[openai]'") from exc
        self.name = name
        self._max_tokens_param = max_tokens_param
        self._embedding_model = embedding_model
        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    def _kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        params = request.params
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.max_tokens is not None:
            kwargs[self._max_tokens_param] = params.max_tokens
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.stop:
            kwargs["stop"] = list(params.stop)
        if params.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    @staticmethod
    def _finish_error(finish_reason: str | None, refusal: str | None) -> ErrorClass | None:
        if refusal:
            return ErrorClass.REFUSAL
        if finish_reason == "content_filter":
            return ErrorClass.CONTENT_FILTERED
        return None

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Call ``chat.completions.create`` (streaming when ``params.stream``)."""
        kwargs = self._kwargs(request)
        start = time.perf_counter()
        try:
            if request.params.stream:
                return await self._generate_stream(request, kwargs, start)
            response = await self._client.chat.completions.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise map_openai_error(exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        finish = choice.finish_reason if choice else None
        refusal = getattr(message, "refusal", None) if message else None
        usage = response.usage
        error = self._finish_error(finish, refusal)
        return GenerationResult(
            text=(message.content or "") if message else "",
            finish_reason=finish,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            raw_response_id=response.id,
            error=error,
            error_message=refusal or (f"finish_reason={finish}" if error else None),
            provider=self.name,
            model=response.model or request.model,
            request_params=request.params.model_dump(exclude_none=True),
            provider_metadata={"system_fingerprint": response.system_fingerprint},
        )

    async def _generate_stream(
        self, request: GenerationRequest, kwargs: dict[str, Any], start: float
    ) -> GenerationResult:
        stream = await self._client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True}
        )
        parts: list[str] = []
        ttft_ms: float | None = None
        finish: str | None = None
        refusal: str | None = None
        input_tokens = output_tokens = 0
        response_id: str | None = None
        model = request.model
        async for chunk in stream:
            response_id = chunk.id
            model = chunk.model or model
            if chunk.usage is not None:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
            for choice in chunk.choices:
                delta = choice.delta
                if delta.content:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    parts.append(delta.content)
                if getattr(delta, "refusal", None):
                    refusal = (refusal or "") + str(delta.refusal)
                if choice.finish_reason:
                    finish = choice.finish_reason
        error = self._finish_error(finish, refusal)
        return GenerationResult(
            text="".join(parts),
            finish_reason=finish,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            time_to_first_token_ms=ttft_ms,
            raw_response_id=response_id,
            error=error,
            error_message=refusal or (f"finish_reason={finish}" if error else None),
            provider=self.name,
            model=model,
            request_params=request.params.model_dump(exclude_none=True),
        )

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed via ``embeddings.create``; the model must be configured explicitly."""
        name = model or self._embedding_model
        if not name:
            raise ValueError("no embedding model configured for provider " + self.name)
        try:
            response = await self._client.embeddings.create(model=name, input=texts)
        except Exception as exc:
            raise map_openai_error(exc) from exc
        return [list(item.embedding) for item in response.data]

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
