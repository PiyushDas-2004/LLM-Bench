"""In-process job queue for API-submitted runs.

A fixed pool of asyncio workers consumes an ``asyncio.Queue``. This keeps the
service dependency-free; for multi-instance deployments swap it for a real
queue (Arq/Redis, Celery, Cloud Tasks) - the worker body (``Runner.run``) and
the run store stay unchanged because all state lives in the database.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import structlog

from llmbench.api.schemas import JobStatus
from llmbench.config.models import ExperimentConfig
from llmbench.domain import TrialRecord, utcnow
from llmbench.errors import LLMBenchError
from llmbench.runner.executor import Runner, RunOutcome, RunProgress

log = structlog.get_logger(__name__)


@dataclass
class Job:
    """A queued experiment run."""

    config: ExperimentConfig
    use_cache: bool
    tags: dict[str, str]
    status: JobStatus = field(
        default_factory=lambda: JobStatus(
            job_id=uuid.uuid4().hex, status="queued", submitted_at=utcnow()
        )
    )


class JobQueue:
    """Bounded worker pool executing :class:`Job` objects."""

    def __init__(
        self,
        runner_factory: Callable[[Callable[[RunProgress], None]], Runner],
        workers: int = 2,
        on_finished: Callable[[RunOutcome, list[TrialRecord]], None] | None = None,
    ) -> None:
        self._runner_factory = runner_factory
        self._workers = workers
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._on_finished = on_finished

    def start(self) -> None:
        """Spawn worker tasks."""
        self._tasks = [asyncio.create_task(self._worker()) for _ in range(self._workers)]

    async def stop(self) -> None:
        """Cancel workers (in-flight runs keep their checkpointed trials and can be resumed)."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def submit(self, job: Job) -> JobStatus:
        """Enqueue a job and return its initial status."""
        self._jobs[job.status.job_id] = job
        self._queue.put_nowait(job)
        return job.status

    def get(self, job_id: str) -> JobStatus | None:
        """Current status of a job (``None`` if unknown)."""
        job = self._jobs.get(job_id)
        return job.status if job else None

    def find_by_run(self, run_id: str) -> JobStatus | None:
        """Job that produced ``run_id``, if it was submitted through this server."""
        return next((j.status for j in self._jobs.values() if j.status.run_id == run_id), None)

    def jobs(self) -> list[JobStatus]:
        """All jobs, newest first."""
        return sorted(
            (j.status for j in self._jobs.values()), key=lambda s: s.submitted_at, reverse=True
        )

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._execute(job)
            finally:
                self._queue.task_done()

    async def _execute(self, job: Job) -> None:
        status = job.status

        def progress(p: RunProgress) -> None:
            status.run_id, status.done, status.total = p.run_id, p.done, p.total

        status.status = "running"
        runner = self._runner_factory(progress)
        try:
            outcome = await runner.run(job.config, use_cache=job.use_cache, tags=job.tags)
        except asyncio.CancelledError:
            status.status, status.error = "failed", "cancelled at server shutdown (resumable)"
            raise
        except LLMBenchError as exc:
            status.status, status.error = "failed", str(exc)
            log.warning("job_failed", job_id=status.job_id, error=str(exc))
            return
        except Exception as exc:
            status.status, status.error = "failed", f"{type(exc).__name__}: {exc}"
            log.exception("job_crashed", job_id=status.job_id)
            return
        status.status = "completed"
        status.run_id, status.run_status = outcome.run_id, outcome.status
        status.done, status.total = outcome.n_done, outcome.n_expected
        if self._on_finished is not None:
            self._on_finished(outcome, runner.store.trials(outcome.run_id))
