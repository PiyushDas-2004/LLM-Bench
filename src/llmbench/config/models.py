"""Pydantic models describing an experiment (the YAML a user writes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

PLACEHOLDER_MODEL_NAMES = frozenset({"", "<set me>", "set-me", "changeme"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RateLimitConfig(_Strict):
    """Token-bucket limits applied per provider."""

    requests_per_minute: float | None = Field(default=None, gt=0)
    tokens_per_minute: float | None = Field(default=None, gt=0)


class BudgetConfig(_Strict):
    """Hard limits; the runner stops gracefully (keeping partial results) when hit."""

    max_usd: float | None = Field(default=None, ge=0)
    max_requests: int | None = Field(default=None, ge=1)


class RetryConfig(_Strict):
    """Retry and timeout policy for transient provider errors."""

    max_attempts: int = Field(default=4, ge=1, le=20)
    base_delay_s: float = Field(default=0.5, ge=0)
    max_delay_s: float = Field(default=30.0, ge=0)
    timeout_s: float = Field(default=60.0, gt=0)


class CircuitBreakerConfig(_Strict):
    """Per-provider circuit breaker thresholds."""

    failure_threshold: int = Field(default=8, ge=1)
    reset_timeout_s: float = Field(default=30.0, gt=0)


class ModelSpec(_Strict):
    """One model under test.

    ``params`` holds generation parameters (``temperature``, ``top_p``,
    ``max_tokens``, ``seed``, ...). For the ``mock`` provider it also holds the
    simulation knobs documented in :mod:`llmbench.providers.mock`.
    """

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    provider: str = Field(min_length=1)
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    rate_limit: RateLimitConfig | None = None

    @property
    def model_name(self) -> str:
        """The provider-side model name (falls back to the provider name for the mock)."""
        return self.model or self.provider

    @property
    def is_placeholder(self) -> bool:
        """True when the user has not filled in a real model name yet."""
        return self.provider != "mock" and (self.model or "").strip().lower() in (
            PLACEHOLDER_MODEL_NAMES
        )


class PromptSpec(_Strict):
    """A versioned prompt: a Jinja2 template file or inline text, plus an optional system prompt."""

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    template: str | None = None
    text: str | None = None
    system: str | None = None
    system_template: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> PromptSpec:
        if (self.template is None) == (self.text is None):
            raise ValueError(f"prompt {self.id!r}: set exactly one of 'template' or 'text'")
        if self.system is not None and self.system_template is not None:
            raise ValueError(f"prompt {self.id!r}: set at most one of 'system'/'system_template'")
        return self


class MetricSpec(_Strict):
    """A metric reference with optional parameters."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class JudgeSpec(_Strict):
    """Configuration of the LLM-as-a-judge used by judge metrics."""

    provider: str
    model: str | None = None
    base_url: str | None = None
    rubric: str
    swap_order: bool = True
    samples: int = Field(default=1, ge=1, le=10)
    max_repairs: int = Field(default=2, ge=0, le=5)
    params: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0})


class SamplingConfig(_Strict):
    """Optional reproducible subsampling of the dataset."""

    n: int | None = Field(default=None, ge=1)
    stratify_by_tag: bool = True
    seed: int = 0


class ExperimentConfig(_Strict):
    """A full experiment: dataset x models x prompts x trials, plus metrics and limits."""

    name: str = Field(min_length=1)
    description: str = ""
    dataset: str
    task: str = "qa"
    trials: int = Field(default=1, ge=1, le=1000)
    seed: int = 0
    concurrency: int = Field(default=8, ge=1, le=1024)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    models: list[ModelSpec] = Field(min_length=1)
    prompts: list[PromptSpec] = Field(min_length=1)
    metrics: list[MetricSpec] = Field(default_factory=list)
    judge: JudgeSpec | None = None
    pricing: str | None = None
    cost_confirm_threshold_usd: float = Field(default=1.0, ge=0)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    cache: bool = True
    tags: dict[str, str] = Field(default_factory=dict)

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @field_validator("metrics", mode="before")
    @classmethod
    def _coerce_metrics(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [{"name": item} if isinstance(item, str) else item for item in value]

    @model_validator(mode="after")
    def _unique_ids(self) -> ExperimentConfig:
        for label, ids in (
            ("model", [m.id for m in self.models]),
            ("prompt", [p.id for p in self.prompts]),
        ):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise ValueError(f"duplicate {label} ids: {dupes}")
        return self

    @property
    def base_dir(self) -> Path:
        """Directory the config was loaded from; relative paths resolve against it."""
        return self._base_dir

    def with_base_dir(self, base_dir: Path) -> ExperimentConfig:
        """Return a copy whose relative paths resolve against ``base_dir``."""
        clone = self.model_copy(deep=True)
        clone._base_dir = base_dir.resolve()
        return clone

    def resolve_path(self, value: str) -> Path:
        """Resolve a path from the config.

        Absolute paths are used as-is. Relative paths are tried against the
        config file's directory first, then the current working directory; the
        first that exists wins. If neither exists the config-relative path is
        returned so error messages point at the expected location.
        """
        path = Path(value)
        if path.is_absolute():
            return path
        for root in (self._base_dir, Path.cwd()):
            candidate = (root / path).resolve()
            if candidate.exists():
                return candidate
        return (self._base_dir / path).resolve()

    def model_by_id(self, model_id: str) -> ModelSpec:
        """Look up a model spec by id."""
        for spec in self.models:
            if spec.id == model_id:
                return spec
        raise KeyError(model_id)

    def prompt_by_id(self, prompt_id: str) -> PromptSpec:
        """Look up a prompt spec by id."""
        for spec in self.prompts:
            if spec.id == prompt_id:
                return spec
        raise KeyError(prompt_id)
