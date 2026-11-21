# ADR 0008: Chat Completions for OpenAI-style servers, google-genai for Gemini, SDK retries off

- Status: Accepted
- Date: 2026-11-21

## Context

LLMBench needs adapters for OpenAI, for OpenAI-compatible servers (Ollama, vLLM, LM Studio)
and for Gemini. Each adapter must report tokens, latency, time to first token (TTFT), finish
reason, refusals and content filtering, and must map failures onto LLMBench's `ErrorClass`
taxonomy. The runner (ADR 0002) owns retries, backoff, the circuit breaker and the budget, so
hidden retries inside an SDK would distort attempt counts, latency and cost accounting.

## Decision

- **OpenAI and compatible servers use Chat Completions** through the official `openai` SDK
  (`providers/openai_provider.py`, class `OpenAIProvider`). The same class serves `openai` and
  `openai_compat`, which differ by `base_url` and by `max_tokens_param`:
  `max_completion_tokens` for OpenAI, `max_tokens` for compatible servers
  (`providers/registry.py`). `json_mode` maps to `response_format={"type": "json_object"}`.
  A non-empty `message.refusal` maps to `REFUSAL`; `finish_reason == "content_filter"` to
  `CONTENT_FILTERED`.
- **Streaming for TTFT.** With `stream: true` the adapter streams with
  `stream_options={"include_usage": True}` so token usage still arrives in the final chunk,
  and records TTFT at the first content delta.
- **Gemini uses `google-genai`** (`providers/gemini_provider.py`): `aio.models.generate_content`
  or `generate_content_stream`, system messages folded into `system_instruction`,
  `response_mime_type="application/json"` for `json_mode`, prompt block reasons and safety-type
  finish reasons mapped to `CONTENT_FILTERED`.
- **SDK retries disabled.** `AsyncOpenAI(max_retries=0)`. The Gemini adapter does not set
  `HttpOptions.retry_options`; in google-genai 3.x an unset value means a single attempt on the
  httpx transport. The runner wraps every attempt in `asyncio.wait_for(timeout_s)` in addition
  to the SDK timeout.
- **Error mapping.** `map_openai_error` / `map_gemini_error` translate SDK exceptions;
  HTTP statuses go through `classify_http_status` (401/403 AUTH, 429 RATE_LIMITED, 408/504
  TIMEOUT, other 5xx SERVER_ERROR, other 4xx INVALID_REQUEST). For OpenAI 429s,
  `parse_retry_after` reads `retry-after-ms` or `retry-after` (seconds or HTTP date).
- **Transport.** The installed SDKs (openai 3.x, google-genai 3.x in `.venv`) use `httpx2`,
  a fork of httpx under a separate import name whose exception classes are not httpx
  subclasses. Adapters therefore accept a generic `http_client` (typed `Any`) that is passed to
  the SDK unchanged, and tests inject an `httpx2.AsyncClient` with a `MockTransport`.
  `providers/_http.py::classify_transport_error` checks both `httpx` and `httpx2`
  `TimeoutException` and `TransportError`.
- SDK imports stay inside these two modules (ruff TID251, see ARCHITECTURE.md) and are
  performed lazily, so the core installs and runs without either extra.

## Consequences

Positive:

- Chat Completions is the API that OpenAI-compatible servers implement, so one adapter covers
  OpenAI and local or self-hosted models with the same request shape and usage accounting.
- Every attempt is a runner attempt: `attempts`, `queue_wait_ms`, latency and cost are accurate,
  and Retry-After, breaker and budget logic see every failure.
- Accepting any client object keeps the adapters working whichever HTTP library the SDK uses.

Negative:

- Responses-API-only features (built-in tools, server-side conversation state, reasoning
  summaries) are not reachable through this adapter.
- Gemini relies on an SDK default for "no retries". If `aiohttp` is installed and no
  `http_client` is passed, google-genai switches to its aiohttp transport, which performs an
  implicit retry on connection errors, and aiohttp exceptions are not classified by
  `_http.py` (they map to SERVER_ERROR). Setting retry options explicitly would remove the
  dependence on defaults.
- The Gemini adapter does not extract Retry-After from 429 responses, so its backoff is purely
  exponential.
- `pyproject.toml` floors (`openai>=1.40`, `google-genai>=1.0`) predate the httpx2 transport;
  the offline adapter tests import `httpx2` and assume SDK versions that use it.
- Parameter dialects drift: some compatible servers reject `seed` or `response_format`, which
  surfaces as INVALID_REQUEST (not retried).

## Alternatives considered

- **OpenAI Responses API.** The newer OpenAI interface with richer features, but most
  OpenAI-compatible servers do not implement it, so it would require a second adapter for the
  same model families and different usage fields. Can be added later as a separate provider.
- **Raw HTTP with httpx for every provider.** Full control and one transport, but duplicates
  auth, streaming parsing and schema handling that the SDKs maintain.
- **Keep SDK retries on.** Fewer moving parts, but attempts become invisible, retries stack
  with the runner's (multiplying load during an outage), and the breaker sees fewer failures.
- **A multi-provider abstraction library.** Broad coverage, but an extra layer between
  LLMBench and the provider that obscures error classes, usage fields and retry behaviour.
