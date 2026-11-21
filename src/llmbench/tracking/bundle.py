"""Portable run bundles: export a run (manifest, trials, metrics) to JSON and import it elsewhere.

Bundles let CI compare against a *stored* baseline without sharing a database:
commit ``baselines/main.json``, then ``llmbench import`` it in the CI job.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from llmbench.domain import MetricValue, TrialRecord
from llmbench.errors import LLMBenchError
from llmbench.tracking.store import RunStore

BUNDLE_FORMAT = "llmbench-run/1"


def _relative(path: str, base: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return path


def export_bundle(store: RunStore, run_ref: str, root: str | Path | None = None) -> dict[str, Any]:
    """Serialise a run; paths under ``root`` (default: cwd) are stored relative to it."""
    run = store.resolve_run(run_ref)
    manifest = json.loads(json.dumps(run.manifest))
    base = str(root or Path.cwd())
    if "dataset" in manifest:
        manifest["dataset"]["path"] = _relative(manifest["dataset"]["path"], base)
    manifest["base_dir"] = _relative(manifest.get("base_dir", "."), base)
    return {
        "format": BUNDLE_FORMAT,
        "run": {
            "id": run.id,
            "name": run.name,
            "status": run.status,
            "task": run.task,
            "dataset_name": run.dataset_name,
            "dataset_hash": run.dataset_hash,
            "config": run.config,
            "manifest": manifest,
            "tags": run.tags,
            "n_expected": run.n_expected,
            "created_at": run.created_at.isoformat(),
        },
        "trials": [t.model_dump(mode="json") for t in store.trials(run.id)],
        "metric_values": [v.model_dump(mode="json") for v in store.metric_values(run.id)],
    }


def import_bundle(
    store: RunStore, data: dict[str, Any], *, base_dir: str | Path = ".", replace: bool = False
) -> str:
    """Load a bundle into ``store``; relative paths resolve against ``base_dir`` (project root)."""
    if data.get("format") != BUNDLE_FORMAT:
        raise LLMBenchError(f"not an LLMBench run bundle (format={data.get('format')!r})")
    run = data["run"]
    run_id = str(run["id"])
    existing = {r.id for r in store.list_runs(limit=100_000)}
    if run_id in existing:
        if not replace:
            raise LLMBenchError(f"run {run_id} already exists (use --replace)")
        store.delete_run(run_id)
    manifest = dict(run["manifest"])
    root = Path(base_dir).resolve()
    manifest["base_dir"] = str((root / manifest.get("base_dir", ".")).resolve())
    if "dataset" in manifest and not Path(manifest["dataset"]["path"]).is_absolute():
        manifest["dataset"]["path"] = str(root / manifest["dataset"]["path"])
    store.create_run(
        run_id=run_id,
        name=run["name"],
        task=run["task"],
        dataset_name=run["dataset_name"],
        dataset_hash=run["dataset_hash"],
        config=run["config"],
        manifest=manifest,
        tags=run.get("tags", {}),
        n_expected=int(run["n_expected"]),
    )
    for trial in data.get("trials", []):
        store.add_trial(TrialRecord.model_validate(trial))
    values = [MetricValue.model_validate(v) for v in data.get("metric_values", [])]
    store.replace_metric_values(run_id, values)
    store.refresh_run_totals(run_id)
    store.update_run(
        run_id,
        status=run.get("status", "completed"),
        created_at=dt.datetime.fromisoformat(run["created_at"]),
    )
    return run_id


def write_bundle(
    store: RunStore, run_ref: str, path: str | Path, root: str | Path | None = None
) -> Path:
    """Export a run bundle to a JSON file (paths relative to ``root``, default cwd)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = export_bundle(store, run_ref, root)
    out.write_text(json.dumps(bundle, indent=1, sort_keys=True), encoding="utf-8")
    return out


def read_bundle(path: str | Path) -> dict[str, Any]:
    """Read a bundle file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise LLMBenchError(f"{path}: not a run bundle")
    return data
