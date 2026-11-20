"""Load experiment YAML files into validated :class:`ExperimentConfig` objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llmbench.config.models import ExperimentConfig
from llmbench.errors import ConfigError


def load_yaml(path: str | Path) -> Any:
    """Parse a YAML file with the safe loader, raising :class:`ConfigError` on failure."""
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"file not found: {file}")
    try:
        return yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {file}: {exc}") from exc


def parse_experiment(data: Any, base_dir: Path) -> ExperimentConfig:
    """Validate an already-parsed mapping as an experiment config."""
    if not isinstance(data, dict):
        raise ConfigError("experiment config must be a mapping")
    try:
        config = ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid experiment config:\n{exc}") from exc
    return config.with_base_dir(base_dir)


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    file = Path(path)
    return parse_experiment(load_yaml(file), file.parent)
