"""Core domain records shared by the runner, metrics, stats and reporting layers.

These are plain pydantic models, decoupled from the ORM so metrics and stats
never depend on storage.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field

from llmbench.providers.base import ErrorClass


def utcnow() -> dt.datetime:
    """Timezone-aware current UTC time."""
    return dt.datetime.now(dt.UTC)


class TrialRecord(BaseModel):
    """One (model, prompt, sample, trial) execution and its measurements."""

    run_id: str
    trial_key: str
    model_id: str
    prompt_id: str
    sample_id: str
    trial_index: int
    system_prompt: str | None = None
    rendered_prompt: str = ""
    output_text: str = ""
    parsed_output: Any = None
    passed: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float | None = None
    ttft_ms: float | None = None
    queue_wait_ms: float = 0.0
    attempts: int = 1
    error_class: ErrorClass | None = None
    error_message: str | None = None
    finish_reason: str | None = None
    refused: bool = False
    malformed: bool = False
    cost_usd: float | None = None
    cached: bool = False
    response_id: str | None = None
    created_at: dt.datetime = Field(default_factory=utcnow)

    @property
    def ok(self) -> bool:
        """True when the provider call succeeded (output may still be wrong)."""
        return self.error_class is None

    @property
    def group(self) -> tuple[str, str]:
        """The configuration this trial belongs to: ``(model_id, prompt_id)``."""
        return (self.model_id, self.prompt_id)

    @property
    def tokens_per_second(self) -> float | None:
        """Output tokens per second of generation time (excluding time to first token)."""
        if not self.ok or self.output_tokens <= 0 or not self.latency_ms:
            return None
        gen_ms = max(self.latency_ms - (self.ttft_ms or 0.0), 1e-6)
        return self.output_tokens / (gen_ms / 1000.0)


class MetricValue(BaseModel):
    """A metric value for one trial (``trial_index`` set) or one sample (``None``)."""

    metric: str
    model_id: str
    prompt_id: str
    sample_id: str
    trial_index: int | None = None
    value: float | None
    details: dict[str, Any] = Field(default_factory=dict)


def config_key(model_id: str, prompt_id: str) -> str:
    """Human-readable key of a configuration, e.g. ``gpt-main/v1``."""
    return f"{model_id}/{prompt_id}"
