"""Run manifests: everything needed to reproduce a run."""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import re
import shutil
import subprocess  # nosec B404 - used only to read the git SHA with a fixed argv
import sys
from pathlib import Path
from typing import Any

from llmbench import __version__
from llmbench.datasets.loader import canonical_json
from llmbench.runner.prepare import PreparedExperiment


def config_hash(config_dump: dict[str, Any]) -> str:
    """Stable hash of a config dump."""
    return hashlib.sha256(canonical_json(config_dump).encode()).hexdigest()


def git_sha(cwd: Path | None = None) -> str | None:
    """Current git commit, or ``None`` outside a repository."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, no shell
            [git, "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def make_run_id(name: str, cfg_hash: str, created_at: dt.datetime) -> str:
    """Deterministic run id: ``<slug>-<UTC timestamp>-<hash of config+timestamp>``."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "run"
    stamp = created_at.strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha256(f"{cfg_hash}|{created_at.isoformat()}".encode()).hexdigest()[:8]
    return f"{slug}-{stamp}-{digest}"


def build_manifest(
    prepared: PreparedExperiment, run_id: str, created_at: dt.datetime
) -> dict[str, Any]:
    """Assemble the manifest stored with a run."""
    config = prepared.config
    dump = config.model_dump(mode="json")
    return {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "llmbench_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": git_sha(config.base_dir),
        "base_dir": str(config.base_dir),
        "config": dump,
        "config_hash": config_hash(dump),
        "task": prepared.task.name,
        "dataset": {
            "name": prepared.dataset.name,
            "path": prepared.dataset.path,
            "full_hash": prepared.dataset.content_hash,
            "evaluated_hash": prepared.evaluated_hash,
            "n_total": len(prepared.dataset),
            "n_evaluated": len(prepared.samples),
            "sample_ids": [s.id for s in prepared.samples],
        },
        "prompts": {pid: tpl.content_hash for pid, tpl in prepared.prompts.items()},
        "models": [
            {
                "id": m.id,
                "provider": m.provider,
                "model": m.model_name,
                "base_url": m.base_url,
                "params": m.params,
            }
            for m in prepared.models
        ],
        "skipped_models": prepared.skipped_models,
        "seeds": {"experiment": config.seed, "sampling": config.sampling.seed},
        "pricing": {
            "path": config.pricing,
            "as_of": prepared.pricing.as_of.isoformat() if prepared.pricing.models else None,
        },
        "grid_size": prepared.grid_size,
    }
