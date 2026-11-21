"""SQLAlchemy 2.0 ORM models and engine helpers for the run store.

The schema is managed by Alembic migrations in ``tracking/migrations``; a test
asserts these models and the migrations never drift apart.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from llmbench.domain import utcnow


class Base(DeclarativeBase):
    """Declarative base for all tables."""


class RunRow(Base):
    """One execution of an experiment."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), index=True)
    task: Mapped[str] = mapped_column(String(64))
    dataset_name: Mapped[str] = mapped_column(String(200))
    dataset_hash: Mapped[str] = mapped_column(String(64), index=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    tags: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    n_expected: Mapped[int] = mapped_column(Integer, default=0)
    n_done: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TrialRow(Base):
    """One provider call (or cache hit) for a (model, prompt, sample, trial)."""

    __tablename__ = "trials"
    __table_args__ = (UniqueConstraint("run_id", "trial_key", name="uq_trials_run_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    trial_key: Mapped[str] = mapped_column(String(512))
    model_id: Mapped[str] = mapped_column(String(128))
    prompt_id: Mapped[str] = mapped_column(String(128))
    sample_id: Mapped[str] = mapped_column(String(256))
    trial_index: Mapped[int] = mapped_column(Integer)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    rendered_prompt: Mapped[str] = mapped_column(Text)
    output_text: Mapped[str] = mapped_column(Text)
    parsed_output: Mapped[Any] = mapped_column(JSON, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    queue_wait_ms: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    error_class: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    refused: Mapped[bool] = mapped_column(Boolean, default=False)
    malformed: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    response_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MetricRow(Base):
    """A per-trial or per-sample metric value."""

    __tablename__ = "metric_values"
    __table_args__ = (Index("ix_metric_values_run_metric", "run_id", "metric"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    metric: Mapped[str] = mapped_column(String(128))
    model_id: Mapped[str] = mapped_column(String(128))
    prompt_id: Mapped[str] = mapped_column(String(128))
    sample_id: Mapped[str] = mapped_column(String(256))
    trial_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class JudgeCallRow(Base):
    """Audit record of every LLM-as-judge call."""

    __tablename__ = "judge_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    subject: Mapped[str] = mapped_column(String(512))
    judge_provider: Mapped[str] = mapped_column(String(64))
    judge_model: Mapped[str] = mapped_column(String(128))
    mode: Mapped[str] = mapped_column(String(32))
    prompt: Mapped[str] = mapped_column(Text)
    output: Mapped[str] = mapped_column(Text)
    parsed: Mapped[Any] = mapped_column(JSON, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BaselineRow(Base):
    """A named pointer to a run used as the regression baseline."""

    __tablename__ = "baselines"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CacheRow(Base):
    """Content-addressed response cache entry."""

    __tablename__ = "cache_entries"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ArtifactRow(Base):
    """Generated artifacts (reports, exports) attached to a run."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(256))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def make_engine(url: str) -> Engine:
    """Create an engine; for SQLite, create the parent directory and enable WAL + FKs."""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        Path(url.removeprefix("sqlite:///")).expanduser().parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, future=True)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _sqlite_pragmas)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory bound to ``engine`` (objects stay usable after commit)."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
