"""Provider-neutral request/response types, the error taxonomy and the Provider protocol.

Core modules depend only on this file; concrete adapters (OpenAI, Gemini, ...)
live in sibling modules and are the only place provider SDKs are imported.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ErrorClass(StrEnum):
    """Typed failure classes. Retry behaviour depends on the class."""

    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    AUTH = "AUTH"
    SERVER_ERROR = "SERVER_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    CONTENT_FILTERED = "CONTENT_FILTERED"
    REFUSAL = "REFUSAL"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    NETWORK = "NETWORK"


#: Error classes that are worth retrying (transient by nature).
RETRYABLE_ERRORS: frozenset[ErrorClass] = frozenset(
    {ErrorClass.TIMEOUT, ErrorClass.RATE_LIMITED, ErrorClass.SERVER_ERROR, ErrorClass.NETWORK}
)


class ProviderError(Exception):
    """A provider call failed with a classified error."""

    def __init__(
        self,
        error_class: ErrorClass,
        message: str,
        *,
        retry_after_s: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.message = message
        self.retry_after_s = retry_after_s
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        """Whether the runner should retry this error."""
        return self.error_class in RETRYABLE_ERRORS


def classify_http_status(status: int) -> ErrorClass:
    """Map an HTTP status code to an :class:`ErrorClass`."""
    if status in (401, 403):
        return ErrorClass.AUTH
    if status == 429:
        return ErrorClass.RATE_LIMITED
    if status in (408, 504):
        return ErrorClass.TIMEOUT
    if status >= 500:
        return ErrorClass.SERVER_ERROR
    return ErrorClass.INVALID_REQUEST


class GenerationParams(BaseModel):
    """Sampling parameters; ``None`` means "provider default"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    json_mode: bool = False
    stream: bool = False

    @classmethod
    def from_mapping(cls, params: dict[str, Any]) -> GenerationParams:
        """Pick the known generation keys out of a free-form params mapping."""
        keys = set(cls.model_fields)
        picked = {k: v for k, v in params.items() if k in keys}
        if isinstance(picked.get("stop"), list):
            picked["stop"] = tuple(picked["stop"])
        return cls.model_validate(picked)


class Message(BaseModel):
    """A chat message."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class GenerationRequest(BaseModel):
    """Everything a provider needs to produce one completion.

    ``metadata`` is never sent to a real provider. The mock provider reads its
    ``oracle`` entry to simulate realistic, controllable behaviour.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    messages: tuple[Message, ...]
    params: GenerationParams = Field(default_factory=GenerationParams)
    trial_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_prompt(
        cls,
        model: str,
        user: str,
        system: str | None = None,
        params: GenerationParams | None = None,
        trial_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationRequest:
        """Build a request from a system + user prompt pair."""
        messages = [Message(role="system", content=system)] if system else []
        messages.append(Message(role="user", content=user))
        return cls(
            model=model,
            messages=tuple(messages),
            params=params or GenerationParams(),
            trial_index=trial_index,
            metadata=metadata or {},
        )

    @property
    def prompt_text(self) -> str:
        """All message contents joined (used for token estimates)."""
        return "\n".join(m.content for m in self.messages)


class GenerationResult(BaseModel):
    """Outcome of one provider call, successful or not."""

    text: str = ""
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    raw_response_id: str | None = None
    error: ErrorClass | None = None
    error_message: str | None = None
    provider: str = ""
    model: str = ""
    request_params: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the call produced usable output."""
        return self.error is None

    @property
    def tokens_per_second(self) -> float | None:
        """Output tokens per second of generation time, when measurable."""
        if self.output_tokens <= 0 or self.latency_ms <= 0:
            return None
        gen_ms = self.latency_ms - (self.time_to_first_token_ms or 0.0)
        return self.output_tokens / (max(gen_ms, 1e-6) / 1000.0)


@runtime_checkable
class Provider(Protocol):
    """Interface every provider adapter implements."""

    name: str

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce a completion or raise :class:`ProviderError`."""
        ...

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed texts; adapters without embeddings raise ``NotImplementedError``."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


REFUSAL_PATTERNS = (
    "i can't help with",
    "i cannot help with",
    "i can't assist",
    "i cannot assist",
    "i'm not able to help",
    "i am not able to help",
    "i won't be able to",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i must decline",
    "i can't comply",
    "i cannot comply",
)


def looks_like_refusal(text: str) -> bool:
    """Heuristic refusal detector (documented as such in METHODOLOGY.md)."""
    lowered = text.strip().lower()[:300]
    return any(p in lowered for p in REFUSAL_PATTERNS)
