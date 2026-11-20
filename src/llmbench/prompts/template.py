"""Versioned prompt templates rendered in a sandboxed Jinja2 environment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, TemplateError, meta
from jinja2.sandbox import SandboxedEnvironment

from llmbench.config.models import ExperimentConfig, PromptSpec
from llmbench.datasets.schema import Sample
from llmbench.errors import PromptError

#: Variables a template may reference. ``reference`` is intentionally absent so
#: a template can never leak the expected answer into the prompt.
ALLOWED_VARIABLES = frozenset({"input", "context", "documents", "tags", "metadata", "sample_id"})

_ENV = SandboxedEnvironment(
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True)
class RenderedPrompt:
    """The exact text sent to a provider for one sample."""

    user: str
    system: str | None

    @property
    def content_hash(self) -> str:
        """Hash of the rendered prompt; part of the response-cache key."""
        blob = f"{self.system or ''}\x00{self.user}".encode()
        return hashlib.sha256(blob).hexdigest()


class PromptTemplate:
    """A compiled prompt (user template + optional system template) with a stable hash."""

    def __init__(self, prompt_id: str, user_source: str, system_source: str | None = None) -> None:
        self.id = prompt_id
        self.user_source = user_source
        self.system_source = system_source
        self._validate_variables()
        try:
            self._user = _ENV.from_string(user_source)
            self._system = _ENV.from_string(system_source) if system_source is not None else None
        except TemplateError as exc:
            raise PromptError(f"prompt {prompt_id!r}: template syntax error: {exc}") from exc

    def _validate_variables(self) -> None:
        for source in (self.user_source, self.system_source):
            if source is None:
                continue
            try:
                names = meta.find_undeclared_variables(_ENV.parse(source))
            except TemplateError as exc:
                raise PromptError(f"prompt {self.id!r}: template syntax error: {exc}") from exc
            unknown = sorted(names - ALLOWED_VARIABLES)
            if unknown:
                raise PromptError(
                    f"prompt {self.id!r}: unknown template variables {unknown}; "
                    f"allowed: {sorted(ALLOWED_VARIABLES)}"
                )

    @property
    def content_hash(self) -> str:
        """Hash of the template sources; identifies the prompt version."""
        blob = f"{self.system_source or ''}\x00{self.user_source}".encode()
        return hashlib.sha256(blob).hexdigest()

    @staticmethod
    def variables_for(sample: Sample) -> dict[str, Any]:
        """Template variables exposed for a sample."""
        return {
            "input": sample.input,
            "context": sample.context or "",
            "documents": sample.documents or [],
            "tags": sample.tags,
            "metadata": sample.metadata,
            "sample_id": sample.id,
        }

    def render(self, sample: Sample, **extra: Any) -> RenderedPrompt:
        """Render for one sample. ``extra`` overrides variables (e.g. RAG context)."""
        variables = {**self.variables_for(sample), **extra}
        try:
            user = self._user.render(**variables).strip()
            system = self._system.render(**variables).strip() if self._system else None
        except TemplateError as exc:
            raise PromptError(f"prompt {self.id!r} failed on sample {sample.id!r}: {exc}") from exc
        return RenderedPrompt(user=user, system=system)


def _read(path: Path, prompt_id: str) -> str:
    if not path.is_file():
        raise PromptError(f"prompt {prompt_id!r}: template file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_prompt(spec: PromptSpec, config: ExperimentConfig | None = None) -> PromptTemplate:
    """Build a :class:`PromptTemplate` from a prompt spec, resolving file paths."""

    def resolve(value: str) -> Path:
        return config.resolve_path(value) if config is not None else Path(value)

    user = spec.text if spec.text is not None else _read(resolve(str(spec.template)), spec.id)
    if spec.system_template is not None:
        system: str | None = _read(resolve(spec.system_template), spec.id)
    else:
        system = spec.system
    return PromptTemplate(spec.id, user, system)
