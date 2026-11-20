"""Provider registry and factory.

A provider factory takes a :class:`~llmbench.config.models.ModelSpec` plus
:class:`~llmbench.config.settings.Settings` and returns a ready provider.
Third-party providers register under the ``llmbench.providers`` entry-point group.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from llmbench.config.models import ModelSpec
from llmbench.config.settings import Settings, get_settings
from llmbench.plugins.registry import Registry
from llmbench.providers.base import Provider
from llmbench.providers.mock import MockProvider

ProviderFactory = Callable[[ModelSpec, Settings], Provider]
PROVIDERS: Registry[ProviderFactory] = Registry("provider", "llmbench.providers")


@dataclass(frozen=True)
class Availability:
    """Whether a model spec can run in this environment, and why not."""

    ok: bool
    reason: str = ""


def _env_key(spec: ModelSpec, default: str | None) -> str | None:
    if spec.api_key_env:
        return os.environ.get(spec.api_key_env) or None
    return default


def _timeout(spec: ModelSpec) -> float:
    return float(spec.params.get("timeout_s", 60.0))


@PROVIDERS.decorator("mock")
def _mock_factory(spec: ModelSpec, settings: Settings) -> Provider:
    return MockProvider(spec.params, name="mock")


@PROVIDERS.decorator("openai")
def _openai_factory(spec: ModelSpec, settings: Settings) -> Provider:
    from llmbench.providers.openai_provider import OpenAIProvider

    default = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    key = _env_key(spec, default)
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAIProvider(
        api_key=key,
        base_url=spec.base_url,
        timeout_s=_timeout(spec),
        embedding_model=spec.params.get("embedding_model"),
    )


@PROVIDERS.decorator("openai_compat")
def _compat_factory(spec: ModelSpec, settings: Settings) -> Provider:
    from llmbench.providers.openai_provider import OpenAIProvider

    if not spec.base_url:
        raise ValueError("openai_compat requires base_url (e.g. http://localhost:11434/v1)")
    return OpenAIProvider(
        api_key=_env_key(spec, None) or "not-needed",
        base_url=spec.base_url,
        name="openai_compat",
        timeout_s=_timeout(spec),
        max_tokens_param="max_tokens",
        embedding_model=spec.params.get("embedding_model"),
    )


@PROVIDERS.decorator("gemini")
def _gemini_factory(spec: ModelSpec, settings: Settings) -> Provider:
    from llmbench.providers.gemini_provider import GeminiProvider

    default = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    key = _env_key(spec, default)
    if not key:
        raise ValueError("GEMINI_API_KEY is not set")
    return GeminiProvider(
        api_key=key, timeout_s=_timeout(spec), embedding_model=spec.params.get("embedding_model")
    )


def check_availability(spec: ModelSpec, settings: Settings | None = None) -> Availability:
    """Decide whether ``spec`` can run here (keys present, model name filled in)."""
    settings = settings or get_settings()
    if spec.provider not in PROVIDERS.names():
        return Availability(False, f"unknown provider {spec.provider!r}")
    if spec.is_placeholder:
        return Availability(False, "model name is a placeholder; set 'model' in the config")
    if spec.provider == "openai" and not _env_key(
        spec, settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    ):
        return Availability(False, "OPENAI_API_KEY is not set")
    if spec.provider == "gemini" and not _env_key(
        spec, settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    ):
        return Availability(False, "GEMINI_API_KEY is not set")
    if spec.provider == "openai_compat" and not spec.base_url:
        return Availability(False, "openai_compat requires base_url")
    return Availability(True)


def create_provider(spec: ModelSpec, settings: Settings | None = None) -> Provider:
    """Instantiate the provider for ``spec``."""
    factory = PROVIDERS.get(spec.provider)
    return factory(spec, settings or get_settings())
