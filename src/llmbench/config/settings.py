"""Process-level settings read from the environment (never from committed files)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    Values come from environment variables (``LLMBENCH_*`` plus the provider key
    variables) and an optional, git-ignored ``.env`` file. Secrets are wrapped in
    :class:`~pydantic.SecretStr` so they never appear in ``repr`` or logs.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLMBENCH_", env_file=".env", extra="ignore", populate_by_name=True
    )

    db_url: str = "sqlite:///.llmbench/llmbench.db"
    api_key: SecretStr | None = None
    log_format: Literal["console", "json"] = "console"
    log_level: str = "INFO"
    redact_prompts: bool = False
    otel_enabled: bool = False
    workspace: str = "."
    api_workers: int = Field(default=2, ge=1, le=64)

    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "LLMBENCH_OPENAI_API_KEY")
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "LLMBENCH_GEMINI_API_KEY"
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings."""
    return Settings()
