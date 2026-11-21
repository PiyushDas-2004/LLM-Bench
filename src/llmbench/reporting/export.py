"""Export run data to JSON, CSV or Parquet (``parquet`` extra)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from llmbench.errors import ConfigError
from llmbench.tracking.bundle import export_bundle
from llmbench.tracking.store import RunStore

ExportFormat = Literal["json", "csv", "parquet"]


def trial_rows(store: RunStore, run_id: str) -> list[dict[str, Any]]:
    """One flat row per trial, with every per-trial metric value as a column."""
    metrics: dict[tuple[str, int], dict[str, float | None]] = {}
    sample_level: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for v in store.metric_values(run_id):
        if v.trial_index is None:
            sample_level.setdefault((v.model_id, v.prompt_id, v.sample_id), {})[v.metric] = v.value
        else:
            key = (f"{v.model_id}|{v.prompt_id}|{v.sample_id}|{v.trial_index}", 0)
            metrics.setdefault(key, {})[f"metric.{v.metric}"] = v.value
    rows = []
    for t in store.trials(run_id):
        row = t.model_dump(mode="json", exclude={"parsed_output", "system_prompt"})
        row["parsed_output"] = json.dumps(t.parsed_output) if t.parsed_output is not None else None
        row.update(metrics.get((t.trial_key, 0), {}))
        for name, value in sample_level.get((t.model_id, t.prompt_id, t.sample_id), {}).items():
            row[f"sample_metric.{name}"] = value
        rows.append(row)
    return rows


def export_run(store: RunStore, run_ref: str, path: str | Path, fmt: ExportFormat) -> Path:
    """Write a run to ``path``; JSON is a full bundle, CSV/Parquet are flat trial tables."""
    run = store.resolve_run(run_ref)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out.write_text(
            json.dumps(export_bundle(store, run.id), indent=1, sort_keys=True), encoding="utf-8"
        )
        return out
    rows = trial_rows(store, run.id)
    columns = sorted({key for row in rows for key in row})
    if fmt == "csv":
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return out
    if fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ConfigError(
                "parquet export needs the 'parquet' extra: pip install 'llmbench[parquet]'"
            ) from exc
        table = pa.Table.from_pylist([{c: row.get(c) for c in columns} for row in rows])
        pq.write_table(table, out)
        return out
    raise ConfigError(f"unknown export format {fmt!r}")
