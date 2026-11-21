"""Shared builders for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmbench.config import ExperimentConfig, parse_experiment

STRONG: dict[str, Any] = {"accuracy": 0.9, "temperature": 0.7, "sample_spread": 0.5}
WEAK: dict[str, Any] = {"accuracy": 0.3, "temperature": 0.7, "sample_spread": 0.5}


def write_qa_dataset(path: Path, n: int = 6) -> Path:
    rows = [
        {
            "id": f"q{i}",
            "input": f"What is {i} plus {i}?",
            "reference": str(2 * i),
            "tags": ["math" if i % 2 else "arith"],
            "difficulty": "medium",
        }
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def make_config(tmp_path: Path, **overrides: Any) -> ExperimentConfig:
    dataset = write_qa_dataset(tmp_path / "qa.jsonl", overrides.pop("n_samples", 6))
    (tmp_path / "pricing.yaml").write_text(
        "as_of: 2026-01-01\nmodels:\n  mock/m1: {input_per_1m: 1.0, output_per_1m: 2.0}\n"
        "  mock/m2: {input_per_1m: 1.0, output_per_1m: 2.0}\n"
    )
    data: dict[str, Any] = {
        "name": "unit-test",
        "dataset": str(dataset),
        "task": "qa",
        "trials": 3,
        "seed": 7,
        "concurrency": 4,
        "pricing": "pricing.yaml",
        "retry": {"max_attempts": 3, "base_delay_s": 0.0, "max_delay_s": 0.0, "timeout_s": 5},
        "models": [
            {
                "id": "m1",
                "provider": "mock",
                "model": "m1",
                "params": {**STRONG},
            },
            {
                "id": "m2",
                "provider": "mock",
                "model": "m2",
                "params": {**WEAK},
            },
        ],
        "prompts": [{"id": "v1", "text": "Q: {{ input }}"}],
        "metrics": ["exact_match", "f1"],
    }
    data.update(overrides)
    return parse_experiment(data, tmp_path)
