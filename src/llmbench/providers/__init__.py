"""Provider abstraction and adapters (mock, OpenAI, Gemini, OpenAI-compatible)."""

from llmbench.providers.base import (
    RETRYABLE_ERRORS,
    ErrorClass,
    GenerationParams,
    GenerationRequest,
    GenerationResult,
    Message,
    Provider,
    ProviderError,
    classify_http_status,
    looks_like_refusal,
)
from llmbench.providers.mock import MockConfig, MockProvider
from llmbench.providers.registry import (
    PROVIDERS,
    Availability,
    check_availability,
    create_provider,
)

__all__ = [
    "PROVIDERS",
    "RETRYABLE_ERRORS",
    "Availability",
    "ErrorClass",
    "GenerationParams",
    "GenerationRequest",
    "GenerationResult",
    "Message",
    "MockConfig",
    "MockProvider",
    "Provider",
    "ProviderError",
    "check_availability",
    "classify_http_status",
    "create_provider",
    "looks_like_refusal",
]
