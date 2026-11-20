"""Load JSONL / YAML datasets with strict validation and content hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llmbench.datasets.schema import Dataset, Sample
from llmbench.errors import DatasetError


def canonical_json(value: Any) -> str:
    """Serialize ``value`` deterministically (sorted keys, compact, UTF-8 safe)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_samples(samples: Iterable[Sample]) -> str:
    """Content hash of a dataset.

    Samples are sorted by id and serialized canonically, so the hash is stable
    under key reordering and line reordering but changes with any content edit.
    """
    payload = [s.model_dump(mode="json") for s in sorted(samples, key=lambda s: s.id)]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise DatasetError(f"{path}:{lineno}: each line must be a JSON object")
        records.append(record)
    return records


def _records_from_yaml(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise DatasetError(f"{path}: YAML dataset must be a list of mappings (or have 'samples')")
    return data


def _resolve_file(path: Path) -> Path:
    if path.is_dir():
        for name in ("questions.jsonl", "samples.jsonl", "dataset.jsonl", "samples.yaml"):
            if (path / name).is_file():
                return path / name
        raise DatasetError(f"{path}: directory has no questions.jsonl/samples.jsonl/samples.yaml")
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    return path


def load_samples(path: str | Path) -> list[Sample]:
    """Load and validate samples, failing loudly on schema errors or duplicate ids."""
    file = _resolve_file(Path(path))
    if file.suffix == ".jsonl":
        records = _records_from_jsonl(file)
    elif file.suffix in {".yaml", ".yml"}:
        records = _records_from_yaml(file)
    else:
        raise DatasetError(f"{file}: unsupported dataset format {file.suffix!r}")
    samples: list[Sample] = []
    for index, record in enumerate(records):
        try:
            samples.append(Sample.model_validate(record))
        except ValidationError as exc:
            raise DatasetError(
                f"{file}: record {index} ({record.get('id')!r}) invalid:\n{exc}"
            ) from exc
    seen: set[str] = set()
    for sample in samples:
        if sample.id in seen:
            raise DatasetError(f"{file}: duplicate sample id {sample.id!r}")
        seen.add(sample.id)
    if not samples:
        raise DatasetError(f"{file}: dataset is empty")
    return samples


def load_dataset(path: str | Path, name: str | None = None) -> Dataset:
    """Load a dataset file (or dataset directory) and compute its content hash."""
    src = Path(path)
    samples = load_samples(src)
    return Dataset(
        name=name or src.stem,
        path=str(src),
        samples=samples,
        content_hash=hash_samples(samples),
    )
