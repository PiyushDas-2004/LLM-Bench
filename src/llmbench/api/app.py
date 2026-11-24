"""FastAPI service: submit runs, poll status, fetch results, compare runs, health, metrics."""

from __future__ import annotations

import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST

from llmbench import __version__
from llmbench.api.jobs import Job, JobQueue
from llmbench.api.schemas import (
    CompareRequest,
    ErrorResponse,
    Health,
    JobStatus,
    RunInfo,
    RunRequest,
)
from llmbench.api.telemetry import Telemetry
from llmbench.config.loader import parse_experiment
from llmbench.config.models import ExperimentConfig
from llmbench.config.settings import Settings, get_settings
from llmbench.errors import ConfigError, LLMBenchError, RunNotFoundError
from llmbench.metrics.engine import evaluate_run
from llmbench.reporting.summary import RunSummary, summarize_run
from llmbench.runner.executor import Runner, RunProgress
from llmbench.tracking.compare import RunComparison, compare_runs
from llmbench.tracking.db import RunRow
from llmbench.tracking.store import RunStore

log = structlog.get_logger(__name__)


def _run_info(row: RunRow) -> RunInfo:
    return RunInfo(
        run_id=row.id,
        name=row.name,
        status=row.status,
        task=row.task,
        dataset_hash=row.dataset_hash,
        n_done=row.n_done,
        n_expected=row.n_expected,
        total_cost_usd=row.total_cost_usd,
        created_at=row.created_at,
        finished_at=row.finished_at,
        stop_reason=row.stop_reason,
    )


def _inside(workspace: Path, value: str, field: str) -> None:
    path = Path(value)
    resolved = (workspace / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(workspace):
        raise ConfigError(f"{field} must be a relative path inside the server workspace: {value!r}")


def validate_workspace_paths(config: ExperimentConfig, workspace: Path) -> None:
    """Reject absolute paths and ``..`` escapes so API users cannot read arbitrary files."""
    _inside(workspace, config.dataset, "dataset")
    if config.pricing:
        _inside(workspace, config.pricing, "pricing")
    if config.judge is not None:
        _inside(workspace, config.judge.rubric, "judge.rubric")
    for prompt in config.prompts:
        for field in ("template", "system_template"):
            value = getattr(prompt, field)
            if value is not None:
                _inside(workspace, value, f"prompts.{prompt.id}.{field}")


def create_app(
    settings: Settings | None = None,
    *,
    db_url: str | None = None,
    require_auth: bool = True,
    workers: int | None = None,
) -> FastAPI:
    """Build the application (a factory so tests and ``llmbench serve`` configure it explicitly)."""
    settings = settings or get_settings()
    workspace = Path(settings.workspace).resolve()
    store = RunStore(db_url or settings.db_url)
    telemetry = Telemetry()

    def runner_factory(progress: Callable[[RunProgress], None]) -> Runner:
        return Runner(store, settings, evaluator=evaluate_run, progress=progress)

    queue = JobQueue(
        runner_factory, workers or settings.api_workers, on_finished=telemetry.record_run
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        queue.start()
        try:
            yield
        finally:
            await queue.stop()
            store.close()

    app = FastAPI(
        title="LLMBench API",
        version=__version__,
        description="Submit benchmark runs, poll progress, fetch results with confidence intervals "
        "and compare runs statistically.",
        lifespan=lifespan,
        responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )
    app.state.store, app.state.queue, app.state.telemetry = store, queue, telemetry
    origins = [o.strip().rstrip("/") for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        telemetry.http_requests.labels(request.method, route_path, str(response.status_code)).inc()
        telemetry.http_latency.labels(route_path).observe(time.perf_counter() - start)
        response.headers["x-request-id"] = request_id
        log.info(
            "http_request", method=request.method, path=route_path, status=response.status_code
        )
        return response

    def error(request: Request, code: int, message: str) -> JSONResponse:
        body = ErrorResponse(error=message, request_id=getattr(request.state, "request_id", None))
        return JSONResponse(status_code=code, content=body.model_dump())

    @app.exception_handler(RunNotFoundError)
    async def not_found(request: Request, exc: RunNotFoundError) -> JSONResponse:
        return error(request, 404, str(exc))

    @app.exception_handler(LLMBenchError)
    async def bad_request(request: Request, exc: LLMBenchError) -> JSONResponse:
        return error(request, 400, str(exc))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return error(request, exc.status_code, str(exc.detail))

    async def authenticate(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if not require_auth:
            return
        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if expected is None or x_api_key is None or not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-API-Key header")

    auth = [Depends(authenticate)]

    @app.get("/health", response_model=Health, tags=["ops"])
    async def health() -> Health:
        """Liveness probe."""
        return Health(status="ok", version=__version__)

    @app.get("/metrics", tags=["ops"], response_class=Response)
    async def metrics() -> Response:
        """Prometheus metrics."""
        return Response(content=telemetry.render(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/runs", response_model=JobStatus, status_code=202, dependencies=auth, tags=["runs"])
    async def submit_run(body: RunRequest) -> JobStatus:
        """Queue an experiment; poll ``GET /runs/{job_id}`` for progress."""
        config = parse_experiment(body.config, workspace)
        validate_workspace_paths(config, workspace)
        return queue.submit(Job(config=config, use_cache=body.use_cache, tags=body.tags))

    @app.get("/runs", response_model=list[RunInfo], dependencies=auth, tags=["runs"])
    async def list_runs(limit: int = 50) -> list[RunInfo]:
        """Stored runs, newest first."""
        return [_run_info(r) for r in store.list_runs(limit=min(max(limit, 1), 500))]

    @app.get("/jobs", response_model=list[JobStatus], dependencies=auth, tags=["runs"])
    async def list_jobs() -> list[JobStatus]:
        """Jobs submitted to this server process."""
        return queue.jobs()

    @app.get("/runs/{ref}", response_model=JobStatus | RunInfo, dependencies=auth, tags=["runs"])
    async def get_run(ref: str) -> JobStatus | RunInfo:
        """Job status (by job id) or stored run metadata (run id, prefix, baseline or latest)."""
        job = queue.get(ref)
        if job is not None:
            return job
        return _run_info(store.resolve_run(ref))

    @app.get("/runs/{ref}/results", response_model=RunSummary, dependencies=auth, tags=["runs"])
    async def run_results(ref: str) -> RunSummary:
        """Full summary with confidence intervals, latency, cost, consistency and Pareto set."""
        job = queue.get(ref)
        if job is not None:
            if job.run_id is None:
                raise HTTPException(status.HTTP_409_CONFLICT, f"job {ref} has not started yet")
            ref = job.run_id
        return summarize_run(store, ref)

    @app.post("/compare", response_model=RunComparison, dependencies=auth, tags=["runs"])
    async def compare(body: CompareRequest) -> RunComparison:
        """Statistical comparison of two runs."""
        pairs: Any = body.pairs
        return compare_runs(
            store,
            body.baseline,
            body.candidate,
            metrics=body.metrics,
            pairs=pairs,
            allow_dataset_drift=body.allow_dataset_drift,
            alpha=body.alpha,
            correction=body.correction,
        )

    return app
