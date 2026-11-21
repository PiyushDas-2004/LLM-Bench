"""Structural metrics for JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from jsonschema import Draft202012Validator

from llmbench.datasets.schema import Sample
from llmbench.domain import TrialRecord
from llmbench.errors import ConfigError
from llmbench.metrics.base import MetricContext, MetricKind, TrialMetric, register_metric
from llmbench.text import extract_json


def _parsed(trial: TrialRecord) -> Any:
    if trial.parsed_output is not None and not isinstance(trial.parsed_output, str):
        return trial.parsed_output
    try:
        return extract_json(trial.output_text)
    except ValueError:
        return None


@register_metric
class JsonValid(TrialMetric):
    """Output contains a decodable JSON value (1/0)."""

    name = "json_valid"
    description = "Output parses as JSON (1/0)."
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 if JSON could be parsed."""
        return float(_parsed(trial) is not None)


@register_metric
class JsonSchemaValid(TrialMetric):
    """Parsed output validates against a JSON Schema (draft 2020-12).

    The schema comes from ``params.schema`` (a path, resolved like other config
    paths, or an inline mapping) or ``sample.metadata.schema``.
    """

    name = "json_schema"
    description = "Output conforms to the JSON Schema (1/0)."
    kind: ClassVar[MetricKind] = "binary"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._validators: dict[str, Draft202012Validator] = {}

    def _validator(self, spec: Any, ctx: MetricContext) -> Draft202012Validator:
        key = json.dumps(spec, sort_keys=True) if isinstance(spec, dict) else str(spec)
        if key not in self._validators:
            if isinstance(spec, dict):
                schema = spec
            else:
                path = ctx.resolve_path(str(spec)) if ctx.resolve_path else Path(str(spec))
                if not Path(path).is_file():
                    raise ConfigError(f"json_schema: schema file not found: {path}")
                schema = json.loads(Path(path).read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            self._validators[key] = Draft202012Validator(schema)
        return self._validators[key]

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 if valid, 0.0 if invalid or unparseable, ``None`` without a schema."""
        spec = self.params.get("schema") or sample.metadata.get("schema")
        if spec is None:
            return None
        parsed = _parsed(trial)
        if parsed is None:
            return 0.0
        return float(self._validator(spec, ctx).is_valid(parsed))


@register_metric
class FieldAccuracy(TrialMetric):
    """Fraction of reference fields reproduced exactly (strings compared case-insensitively)."""

    name = "field_accuracy"
    description = "Share of reference JSON fields extracted correctly."

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """Matching fields / reference fields."""
        reference = sample.reference
        if not isinstance(reference, dict) or not reference:
            return None
        parsed = _parsed(trial)
        if not isinstance(parsed, dict):
            return 0.0
        hits = 0
        for key, want in reference.items():
            got = parsed.get(key)
            if isinstance(got, str) and isinstance(want, str):
                hits += got.strip().lower() == want.strip().lower()
            else:
                hits += got == want
        return hits / len(reference)
