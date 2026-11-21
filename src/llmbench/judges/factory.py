"""Build an :class:`LLMJudge` from an experiment's ``judge`` section."""

from __future__ import annotations

import structlog

from llmbench.config.models import ExperimentConfig, JudgeSpec, ModelSpec
from llmbench.config.settings import Settings
from llmbench.judges.judge import LLMJudge
from llmbench.judges.rubric import Rubric
from llmbench.providers.registry import check_availability, create_provider
from llmbench.runner.prepare import load_pricing

log = structlog.get_logger(__name__)


def judge_model_spec(spec: JudgeSpec) -> ModelSpec:
    """The judge expressed as a model spec (for availability checks and the factory)."""
    return ModelSpec(
        id="judge",
        provider=spec.provider,
        model=spec.model,
        base_url=spec.base_url,
        params=dict(spec.params),
    )


def build_judge(config: ExperimentConfig, settings: Settings) -> LLMJudge | None:
    """Create the configured judge, or ``None`` if none is configured or it cannot run here."""
    spec = config.judge
    if spec is None:
        return None
    model_spec = judge_model_spec(spec)
    availability = check_availability(model_spec, settings)
    if not availability.ok:
        log.warning("judge_unavailable", reason=availability.reason)
        return None
    for candidate in config.models:
        if candidate.provider == spec.provider and candidate.model_name == model_spec.model_name:
            log.warning(
                "judge_is_candidate",
                model=model_spec.model_name,
                hint="self-preference bias: use a different judge model",
            )
    return LLMJudge(
        provider=create_provider(model_spec, settings),
        provider_name=spec.provider,
        model=model_spec.model_name,
        rubric=Rubric.from_yaml(config.resolve_path(spec.rubric)),
        params=spec.params,
        samples=spec.samples,
        max_repairs=spec.max_repairs,
        swap_order=spec.swap_order,
        pricing=load_pricing(config),
    )
