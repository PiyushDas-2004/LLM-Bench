"""Repository over the run database: runs, trials, metrics, judge calls, baselines, cache."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.engine import Engine

from llmbench.domain import MetricValue, TrialRecord, utcnow
from llmbench.errors import RunNotFoundError
from llmbench.providers.base import ErrorClass
from llmbench.tracking import migrate
from llmbench.tracking.db import (
    ArtifactRow,
    BaselineRow,
    CacheRow,
    JudgeCallRow,
    MetricRow,
    RunRow,
    TrialRow,
    make_engine,
    make_session_factory,
)

TERMINAL_STATUSES = frozenset({"completed", "interrupted", "budget_exceeded", "failed", "partial"})


def _trial_from_row(row: TrialRow) -> TrialRecord:
    return TrialRecord(
        run_id=row.run_id,
        trial_key=row.trial_key,
        model_id=row.model_id,
        prompt_id=row.prompt_id,
        sample_id=row.sample_id,
        trial_index=row.trial_index,
        system_prompt=row.system_prompt,
        rendered_prompt=row.rendered_prompt,
        output_text=row.output_text,
        parsed_output=row.parsed_output,
        passed=row.passed,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        latency_ms=row.latency_ms,
        ttft_ms=row.ttft_ms,
        queue_wait_ms=row.queue_wait_ms,
        attempts=row.attempts,
        error_class=ErrorClass(row.error_class) if row.error_class else None,
        error_message=row.error_message,
        finish_reason=row.finish_reason,
        refused=row.refused,
        malformed=row.malformed,
        cost_usd=row.cost_usd,
        cached=row.cached,
        response_id=row.response_id,
        created_at=row.created_at,
    )


class RunStore:
    """Persistence for everything LLMBench records. Safe to share within one process."""

    def __init__(self, url: str, *, engine: Engine | None = None) -> None:
        self.url = url
        self.engine = engine or make_engine(url)
        migrate.upgrade(self.engine)
        self._session = make_session_factory(self.engine)

    def close(self) -> None:
        """Dispose of pooled connections."""
        self.engine.dispose()

    # ------------------------------------------------------------------ runs
    def create_run(
        self,
        *,
        run_id: str,
        name: str,
        task: str,
        dataset_name: str,
        dataset_hash: str,
        config: dict[str, Any],
        manifest: dict[str, Any],
        tags: dict[str, Any],
        n_expected: int,
    ) -> RunRow:
        """Insert a new run in ``running`` status."""
        row = RunRow(
            id=run_id,
            name=name,
            status="running",
            task=task,
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            config=config,
            manifest=manifest,
            tags=tags,
            n_expected=n_expected,
            n_done=0,
            created_at=utcnow(),
        )
        with self._session.begin() as session:
            session.add(row)
        return row

    def get_run(self, run_id: str) -> RunRow:
        """Fetch a run by exact id."""
        with self._session() as session:
            row = session.get(RunRow, run_id)
        if row is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return row

    def resolve_run(self, ref: str) -> RunRow:
        """Resolve ``latest``, a baseline name, an exact id, or a unique id prefix."""
        with self._session() as session:
            if ref == "latest":
                row = session.scalars(
                    select(RunRow).order_by(RunRow.created_at.desc()).limit(1)
                ).first()
                if row is None:
                    raise RunNotFoundError("no runs recorded yet")
                return row
            exact = session.get(RunRow, ref)
            if exact is not None:
                return exact
            baseline = session.get(BaselineRow, ref)
            if baseline is not None:
                return self.get_run(baseline.run_id)
            matches = list(session.scalars(select(RunRow).where(RunRow.id.startswith(ref))))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RunNotFoundError(f"run reference {ref!r} is ambiguous ({len(matches)} matches)")
        raise RunNotFoundError(f"run not found: {ref}")

    def list_runs(self, limit: int = 50, name: str | None = None) -> list[RunRow]:
        """Most recent runs first."""
        stmt = select(RunRow).order_by(RunRow.created_at.desc()).limit(limit)
        if name:
            stmt = stmt.where(RunRow.name == name)
        with self._session() as session:
            return list(session.scalars(stmt))

    def update_run(self, run_id: str, **fields: Any) -> None:
        """Update columns of a run (status, n_done, stop_reason, ...)."""
        with self._session.begin() as session:
            row = session.get(RunRow, run_id)
            if row is None:
                raise RunNotFoundError(f"run not found: {run_id}")
            for key, value in fields.items():
                setattr(row, key, value)
            if fields.get("status") in TERMINAL_STATUSES:
                row.finished_at = utcnow()

    def refresh_run_totals(self, run_id: str) -> None:
        """Recompute ``n_done`` and ``total_cost_usd`` from stored trials."""
        with self._session.begin() as session:
            n_done, cost = session.execute(
                select(func.count(TrialRow.id), func.sum(TrialRow.cost_usd)).where(
                    TrialRow.run_id == run_id
                )
            ).one()
            row = session.get(RunRow, run_id)
            if row is not None:
                row.n_done = int(n_done)
                row.total_cost_usd = float(cost) if cost is not None else None

    def delete_run(self, run_id: str) -> None:
        """Delete a run and everything attached to it."""
        with self._session.begin() as session:
            for table in (TrialRow, MetricRow, JudgeCallRow, ArtifactRow, BaselineRow):
                session.execute(delete(table).where(table.run_id == run_id))
            session.execute(delete(RunRow).where(RunRow.id == run_id))

    # ---------------------------------------------------------------- trials
    def add_trial(self, trial: TrialRecord) -> None:
        """Persist one trial immediately (the checkpoint unit)."""
        data = trial.model_dump()
        data["error_class"] = trial.error_class.value if trial.error_class else None
        with self._session.begin() as session:
            session.add(TrialRow(**data))

    def completed_trial_keys(self, run_id: str) -> set[str]:
        """Keys of trials already persisted for ``run_id`` (used by resume)."""
        with self._session() as session:
            return set(session.scalars(select(TrialRow.trial_key).where(TrialRow.run_id == run_id)))

    def trials(self, run_id: str) -> list[TrialRecord]:
        """All trials of a run in a stable order."""
        stmt = (
            select(TrialRow)
            .where(TrialRow.run_id == run_id)
            .order_by(
                TrialRow.model_id, TrialRow.prompt_id, TrialRow.sample_id, TrialRow.trial_index
            )
        )
        with self._session() as session:
            return [_trial_from_row(r) for r in session.scalars(stmt)]

    # --------------------------------------------------------------- metrics
    def replace_metric_values(
        self, run_id: str, values: Sequence[MetricValue], metrics: Iterable[str] | None = None
    ) -> None:
        """Replace stored values for the given metrics (default: those in ``values``)."""
        names = set(metrics) if metrics is not None else {v.metric for v in values}
        with self._session.begin() as session:
            if names:
                session.execute(
                    delete(MetricRow).where(MetricRow.run_id == run_id, MetricRow.metric.in_(names))
                )
            session.add_all(MetricRow(run_id=run_id, **v.model_dump()) for v in values)

    def metric_values(self, run_id: str, metric: str | None = None) -> list[MetricValue]:
        """Stored metric values for a run, optionally one metric only."""
        stmt = select(MetricRow).where(MetricRow.run_id == run_id)
        if metric:
            stmt = stmt.where(MetricRow.metric == metric)
        with self._session() as session:
            rows = list(session.scalars(stmt.order_by(MetricRow.id)))
        return [
            MetricValue(
                metric=r.metric,
                model_id=r.model_id,
                prompt_id=r.prompt_id,
                sample_id=r.sample_id,
                trial_index=r.trial_index,
                value=r.value,
                details=r.details or {},
            )
            for r in rows
        ]

    # ------------------------------------------------------------ judge calls
    def add_judge_call(self, **fields: Any) -> None:
        """Persist one judge call for auditability."""
        with self._session.begin() as session:
            session.add(JudgeCallRow(**fields))

    def judge_calls(self, run_id: str | None) -> list[JudgeCallRow]:
        """Judge calls for a run (or calibration calls when ``run_id`` is ``None``)."""
        stmt = (
            select(JudgeCallRow).where(JudgeCallRow.run_id.is_(None))
            if run_id is None
            else (select(JudgeCallRow).where(JudgeCallRow.run_id == run_id))
        )
        with self._session() as session:
            return list(session.scalars(stmt.order_by(JudgeCallRow.id)))

    # -------------------------------------------------------------- baselines
    def set_baseline(self, name: str, run_id: str, note: str | None = None) -> None:
        """Point baseline ``name`` at ``run_id`` (overwrites)."""
        self.get_run(run_id)
        with self._session.begin() as session:
            row = session.get(BaselineRow, name)
            if row is None:
                session.add(BaselineRow(name=name, run_id=run_id, note=note, created_at=utcnow()))
            else:
                row.run_id, row.note, row.created_at = run_id, note, utcnow()

    def get_baseline(self, name: str) -> BaselineRow:
        """Fetch a baseline by name."""
        with self._session() as session:
            row = session.get(BaselineRow, name)
        if row is None:
            raise RunNotFoundError(f"baseline not found: {name}")
        return row

    def list_baselines(self) -> list[BaselineRow]:
        """All baselines, newest first."""
        with self._session() as session:
            return list(
                session.scalars(select(BaselineRow).order_by(BaselineRow.created_at.desc()))
            )

    # ------------------------------------------------------------------ cache
    def cache_get(self, key: str) -> dict[str, Any] | None:
        """Cached payload for ``key`` or ``None``."""
        with self._session() as session:
            row = session.get(CacheRow, key)
        return dict(row.payload) if row is not None else None

    def cache_put(self, key: str, provider: str, model: str, payload: dict[str, Any]) -> None:
        """Insert or replace a cache entry."""
        with self._session.begin() as session:
            session.merge(
                CacheRow(
                    key=key, provider=provider, model=model, payload=payload, created_at=utcnow()
                )
            )

    def cache_clear(self, older_than: dt.datetime | None = None) -> int:
        """Delete cache entries (optionally only older ones); returns rows removed."""
        stmt = delete(CacheRow)
        if older_than is not None:
            stmt = stmt.where(CacheRow.created_at < older_than)
        with self._session.begin() as session:
            result = session.execute(stmt)
        return int(getattr(result, "rowcount", 0) or 0)

    # -------------------------------------------------------------- artifacts
    def latest_artifact_json(self, run_id: str, kind: str) -> dict[str, Any] | None:
        """Most recent artifact of ``kind`` for a run, decoded as JSON (``None`` if absent)."""
        stmt = (
            select(ArtifactRow)
            .where(ArtifactRow.run_id == run_id, ArtifactRow.kind == kind)
            .order_by(ArtifactRow.id.desc())
            .limit(1)
        )
        with self._session() as session:
            row = session.scalars(stmt).first()
        if row is None:
            return None
        data = json.loads(row.content)
        return data if isinstance(data, dict) else None

    def add_artifact(self, run_id: str, kind: str, name: str, content: str) -> None:
        """Attach a generated artifact (e.g. a rendered report) to a run."""
        with self._session.begin() as session:
            session.add(ArtifactRow(run_id=run_id, kind=kind, name=name, content=content))
