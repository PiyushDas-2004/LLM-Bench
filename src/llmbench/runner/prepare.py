"""Turn an :class:`ExperimentConfig` into everything needed to execute or estimate it."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import structlog

from llmbench.config.models import ExperimentConfig, ModelSpec
from llmbench.config.pricing import PricingTable
from llmbench.config.settings import Settings, get_settings
from llmbench.datasets.loader import hash_samples, load_dataset
from llmbench.datasets.sampling import stratified_sample
from llmbench.datasets.schema import Dataset, Sample
from llmbench.errors import ConfigError
from llmbench.prompts.template import PromptTemplate, RenderedPrompt, load_prompt
from llmbench.providers.registry import check_availability
from llmbench.tasks.base import Task, get_task

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class WorkItem:
    """One unit of work: a (model, prompt, sample, trial) cell of the grid."""

    model: ModelSpec
    prompt_id: str
    sample: Sample
    trial_index: int

    @property
    def key(self) -> str:
        """Deterministic trial key, unique within a run."""
        return f"{self.model.id}|{self.prompt_id}|{self.sample.id}|{self.trial_index}"


@dataclass
class PreparedExperiment:
    """A validated, loaded experiment ready to run."""

    config: ExperimentConfig
    dataset: Dataset
    samples: list[Sample]
    task: Task
    prompts: dict[str, PromptTemplate]
    models: list[ModelSpec]
    skipped_models: dict[str, str]
    pricing: PricingTable
    _rendered: dict[tuple[str, str], RenderedPrompt] = field(default_factory=dict)

    @property
    def evaluated_hash(self) -> str:
        """Content hash of the samples actually evaluated (after sampling)."""
        return hash_samples(self.samples)

    @property
    def grid_size(self) -> int:
        """Number of trials the full grid contains."""
        return len(self.models) * len(self.prompts) * len(self.samples) * self.config.trials

    def iter_items(self) -> Iterator[WorkItem]:
        """Lazily expand the grid, trial-major so partial runs cover every config evenly."""
        for trial in range(self.config.trials):
            for sample in self.samples:
                for model in self.models:
                    for prompt_id in self.prompts:
                        yield WorkItem(model, prompt_id, sample, trial)

    def render(self, prompt_id: str, sample: Sample) -> RenderedPrompt:
        """Render (and memoise) a prompt for a sample."""
        key = (prompt_id, sample.id)
        if key not in self._rendered:
            self._rendered[key] = self.prompts[prompt_id].render(sample)
        return self._rendered[key]


def load_pricing(config: ExperimentConfig) -> PricingTable:
    """Load the configured pricing table or an empty one."""
    if not config.pricing:
        return PricingTable.empty()
    path = config.resolve_path(config.pricing)
    if not path.is_file():
        raise ConfigError(f"pricing file not found: {path}")
    return PricingTable.from_yaml(path)


def prepare(
    config: ExperimentConfig,
    settings: Settings | None = None,
    *,
    strict_models: bool = False,
) -> PreparedExperiment:
    """Load dataset, prompts, task and pricing, and decide which models can run.

    Models whose provider is unavailable here (missing API key, placeholder
    model name) are skipped with a recorded reason unless ``strict_models``.
    """
    settings = settings or get_settings()
    dataset = load_dataset(config.resolve_path(config.dataset))
    samples = dataset.samples
    if config.sampling.n is not None:
        samples = stratified_sample(
            samples,
            config.sampling.n,
            seed=config.sampling.seed,
            stratify=config.sampling.stratify_by_tag,
        )
    task = get_task(config.task)
    prompts = {spec.id: load_prompt(spec, config) for spec in config.prompts}
    models: list[ModelSpec] = []
    skipped: dict[str, str] = {}
    for spec in config.models:
        availability = check_availability(spec, settings)
        if availability.ok:
            models.append(spec)
        elif strict_models:
            raise ConfigError(f"model {spec.id!r} cannot run: {availability.reason}")
        else:
            skipped[spec.id] = availability.reason
            log.debug("model_skipped", model_id=spec.id, reason=availability.reason)
    if not models:
        reasons = "; ".join(f"{k}: {v}" for k, v in skipped.items())
        raise ConfigError(f"no runnable models ({reasons})")
    return PreparedExperiment(
        config=config,
        dataset=dataset,
        samples=samples,
        task=task,
        prompts=prompts,
        models=models,
        skipped_models=skipped,
        pricing=load_pricing(config),
    )
