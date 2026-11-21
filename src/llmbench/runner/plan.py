"""Pre-flight estimates of grid size, tokens and cost (``llmbench plan``)."""

from __future__ import annotations

import math

from pydantic import BaseModel

from llmbench.runner.prepare import PreparedExperiment

#: Assumed output length when a model sets no ``max_tokens`` (upper-bound estimate).
DEFAULT_MAX_OUTPUT_TOKENS = 512


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token) used only for estimates and rate limits."""
    return max(1, math.ceil(len(text) / 4))


class ModelEstimate(BaseModel):
    """Estimate for one model across all prompts, samples and trials."""

    model_id: str
    provider: str
    model: str
    requests: int
    input_tokens: int
    output_tokens_upper: int
    cost_usd_upper: float | None


class PlanEstimate(BaseModel):
    """Grid size and upper-bound cost estimate for an experiment."""

    n_models: int
    n_prompts: int
    n_samples: int
    trials: int
    total_requests: int
    models: list[ModelEstimate]
    skipped_models: dict[str, str]
    pricing_as_of: str | None

    @property
    def total_cost_upper(self) -> float | None:
        """Sum of known model costs; ``None`` if every model's price is unknown."""
        known = [m.cost_usd_upper for m in self.models if m.cost_usd_upper is not None]
        return sum(known) if known else None

    @property
    def unpriced_models(self) -> list[str]:
        """Models with no pricing entry (their cost is not included in the total)."""
        return [m.model_id for m in self.models if m.cost_usd_upper is None]


def max_output_tokens(params: dict[str, object]) -> int:
    """The output-token ceiling a model config allows."""
    value = params.get("max_tokens")
    return int(value) if isinstance(value, int) and value > 0 else DEFAULT_MAX_OUTPUT_TOKENS


def estimate(prepared: PreparedExperiment) -> PlanEstimate:
    """Estimate requests, tokens and an upper bound on cost before spending money."""
    trials = prepared.config.trials
    prompt_tokens = sum(
        estimate_tokens(f"{r.system or ''}\n{r.user}")
        for pid in prepared.prompts
        for r in (prepared.render(pid, s) for s in prepared.samples)
    )
    per_model_requests = len(prepared.prompts) * len(prepared.samples) * trials
    models = []
    for spec in prepared.models:
        input_tokens = prompt_tokens * trials
        output_upper = max_output_tokens(spec.params) * per_model_requests
        cost = prepared.pricing.cost(spec.provider, spec.model_name, input_tokens, output_upper)
        models.append(
            ModelEstimate(
                model_id=spec.id,
                provider=spec.provider,
                model=spec.model_name,
                requests=per_model_requests,
                input_tokens=input_tokens,
                output_tokens_upper=output_upper,
                cost_usd_upper=cost,
            )
        )
    has_prices = bool(prepared.pricing.models)
    return PlanEstimate(
        n_models=len(prepared.models),
        n_prompts=len(prepared.prompts),
        n_samples=len(prepared.samples),
        trials=trials,
        total_requests=per_model_requests * len(prepared.models),
        models=models,
        skipped_models=prepared.skipped_models,
        pricing_as_of=prepared.pricing.as_of.isoformat() if has_prices else None,
    )
