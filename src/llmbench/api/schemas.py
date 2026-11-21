"""Request and response models for the REST API."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Submit an experiment. Relative paths resolve inside the server's workspace."""

    config: dict[str, Any] = Field(description="Experiment config (same schema as the YAML file).")
    use_cache: bool = True
    tags: dict[str, str] = Field(default_factory=dict)


class JobStatus(BaseModel):
    """State of a submitted run."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    run_id: str | None = None
    run_status: str | None = None
    done: int = 0
    total: int = 0
    error: str | None = None
    submitted_at: dt.datetime


class RunInfo(BaseModel):
    """Stored run metadata."""

    run_id: str
    name: str
    status: str
    task: str
    dataset_hash: str
    n_done: int
    n_expected: int
    total_cost_usd: float | None
    created_at: dt.datetime
    finished_at: dt.datetime | None
    stop_reason: str | None


class CompareRequest(BaseModel):
    """Compare two stored runs."""

    baseline: str
    candidate: str
    metrics: list[str] | None = None
    pairs: list[tuple[str, str]] | None = None
    allow_dataset_drift: bool = False
    alpha: float = Field(default=0.05, gt=0, lt=1)
    correction: Literal["holm", "bh", "none"] = "holm"


class ErrorResponse(BaseModel):
    """Error envelope."""

    error: str
    request_id: str | None = None


class Health(BaseModel):
    """Liveness response."""

    status: Literal["ok"]
    version: str
