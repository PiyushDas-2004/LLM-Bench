"""Google Gemini adapter using the official ``google-genai`` SDK (``gemini`` extra)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from llmbench.providers._http import classify_transport_error
from llmbench.providers.base import (
    ErrorClass,
    GenerationRequest,
    GenerationResult,
    ProviderError,
    classify_http_status,
)

if TYPE_CHECKING:
    from google.genai import Client

_FILTER_REASONS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}
)


def map_gemini_error(exc: Exception) -> ProviderError:
    """Translate ``google-genai`` / transport exceptions into a :class:`ProviderError`."""
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        status = int(exc.code or 500)
        return ProviderError(
            classify_http_status(status),
            f"HTTP {status}: {exc.message or exc.status}",
            status_code=status,
        )
    transport = classify_transport_error(exc)
    if transport is not None:
        return ProviderError(transport, f"{type(exc).__name__}: {exc}")
    return ProviderError(ErrorClass.SERVER_ERROR, f"{type(exc).__name__}: {exc}")


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", value))


class GeminiProvider:
    """Gemini ``generate_content`` adapter.

    Args:
        api_key: Gemini API key (from ``GEMINI_API_KEY``).
        name: Provider name recorded in results.
        timeout_s: Per-request timeout.
        http_client: Optional async HTTP client of the type the SDK expects (used by
            tests to inject a mock transport).
        embedding_model: Default embedding model for :meth:`embed`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        name: str = "gemini",
        timeout_s: float = 60.0,
        http_client: Any = None,
        embedding_model: str | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("install the 'gemini' extra: pip install 'llmbench[gemini]'") from exc
        self.name = name
        self._embedding_model = embedding_model
        options = types.HttpOptions(timeout=int(timeout_s * 1000), httpx_async_client=http_client)
        self._client: Client = genai.Client(api_key=api_key, http_options=options)

    def _config(self, request: GenerationRequest) -> Any:
        from google.genai import types

        params = request.params
        system = "\n".join(m.content for m in request.messages if m.role == "system") or None
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=params.temperature,
            top_p=params.top_p,
            max_output_tokens=params.max_tokens,
            seed=params.seed,
            stop_sequences=list(params.stop) if params.stop else None,
            response_mime_type="application/json" if params.json_mode else None,
        )

    @staticmethod
    def _contents(request: GenerationRequest) -> list[dict[str, Any]]:
        contents = []
        for message in request.messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})
        return contents

    @staticmethod
    def _text_of(response: Any) -> str:
        candidates = getattr(response, "candidates", None) or []
        if not candidates or candidates[0].content is None:
            return ""
        parts = candidates[0].content.parts or []
        return "".join(p.text for p in parts if getattr(p, "text", None))

    def _result(
        self,
        request: GenerationRequest,
        text: str,
        last: Any,
        latency_ms: float,
        ttft_ms: float | None,
    ) -> GenerationResult:
        candidates = getattr(last, "candidates", None) or []
        finish = _enum_name(candidates[0].finish_reason) if candidates else None
        feedback = getattr(last, "prompt_feedback", None)
        block = _enum_name(getattr(feedback, "block_reason", None)) if feedback else None
        error: ErrorClass | None = None
        if block or (finish in _FILTER_REASONS):
            error = ErrorClass.CONTENT_FILTERED
        usage = getattr(last, "usage_metadata", None)
        return GenerationResult(
            text=text,
            finish_reason=finish,
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            raw_response_id=getattr(last, "response_id", None),
            error=error,
            error_message=f"blocked: {block or finish}" if error else None,
            provider=self.name,
            model=getattr(last, "model_version", None) or request.model,
            request_params=request.params.model_dump(exclude_none=True),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Call ``aio.models.generate_content`` (streaming when ``params.stream``)."""
        config = self._config(request)
        contents = self._contents(request)
        start = time.perf_counter()
        try:
            if request.params.stream:
                parts: list[str] = []
                ttft_ms: float | None = None
                last: Any = None
                stream = await self._client.aio.models.generate_content_stream(
                    model=request.model, contents=contents, config=config
                )
                async for chunk in stream:
                    piece = self._text_of(chunk)
                    if piece and ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    parts.append(piece)
                    last = chunk
                latency = (time.perf_counter() - start) * 1000
                return self._result(request, "".join(parts), last, latency, ttft_ms)
            response = await self._client.aio.models.generate_content(
                model=request.model, contents=contents, config=config
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise map_gemini_error(exc) from exc
        latency = (time.perf_counter() - start) * 1000
        return self._result(request, self._text_of(response), response, latency, None)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed via ``aio.models.embed_content``; the model must be configured explicitly."""
        name = model or self._embedding_model
        if not name:
            raise ValueError("no embedding model configured for provider " + self.name)
        try:
            response = await self._client.aio.models.embed_content(model=name, contents=texts)
        except Exception as exc:
            raise map_gemini_error(exc) from exc
        return [list(e.values or []) for e in (response.embeddings or [])]

    async def aclose(self) -> None:
        """Close the SDK's async client."""
        await self._client.aio.aclose()
