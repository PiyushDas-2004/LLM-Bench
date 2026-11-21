"""Initial schema: runs, trials, metric values, judge calls, baselines, cache, artifacts.

Revision ID: 0001
Revises:
Create Date: 2026-11-20 21:31:19.574969
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cache_entries",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("task", sa.String(length=64), nullable=False),
        sa.Column("dataset_name", sa.String(length=200), nullable=False),
        sa.Column("dataset_hash", sa.String(length=64), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("n_expected", sa.Integer(), nullable=False),
        sa.Column("n_done", sa.Integer(), nullable=False),
        sa.Column("total_cost_usd", sa.Float(), nullable=True),
        sa.Column("stop_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_runs_dataset_hash"), ["dataset_hash"], unique=False)
        batch_op.create_index(batch_op.f("ix_runs_status"), ["status"], unique=False)

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("artifacts", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_artifacts_run_id"), ["run_id"], unique=False)

    op.create_table(
        "baselines",
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_table(
        "judge_calls",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column("subject", sa.String(length=512), nullable=False),
        sa.Column("judge_provider", sa.String(length=64), nullable=False),
        sa.Column("judge_model", sa.String(length=128), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("output", sa.Text(), nullable=False),
        sa.Column("parsed", sa.JSON(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("judge_calls", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_judge_calls_run_id"), ["run_id"], unique=False)

    op.create_table(
        "metric_values",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("metric", sa.String(length=128), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("prompt_id", sa.String(length=128), nullable=False),
        sa.Column("sample_id", sa.String(length=256), nullable=False),
        sa.Column("trial_index", sa.Integer(), nullable=True),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("metric_values", schema=None) as batch_op:
        batch_op.create_index("ix_metric_values_run_metric", ["run_id", "metric"], unique=False)

    op.create_table(
        "trials",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("trial_key", sa.String(length=512), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("prompt_id", sa.String(length=128), nullable=False),
        sa.Column("sample_id", sa.String(length=256), nullable=False),
        sa.Column("trial_index", sa.Integer(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("rendered_prompt", sa.Text(), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=False),
        sa.Column("parsed_output", sa.JSON(), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("ttft_ms", sa.Float(), nullable=True),
        sa.Column("queue_wait_ms", sa.Float(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_class", sa.String(length=32), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("finish_reason", sa.String(length=64), nullable=True),
        sa.Column("refused", sa.Boolean(), nullable=False),
        sa.Column("malformed", sa.Boolean(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("response_id", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "trial_key", name="uq_trials_run_key"),
    )
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_trials_run_id"), ["run_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_trials_run_id"))

    op.drop_table("trials")
    with op.batch_alter_table("metric_values", schema=None) as batch_op:
        batch_op.drop_index("ix_metric_values_run_metric")

    op.drop_table("metric_values")
    with op.batch_alter_table("judge_calls", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_judge_calls_run_id"))

    op.drop_table("judge_calls")
    op.drop_table("baselines")
    with op.batch_alter_table("artifacts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_artifacts_run_id"))

    op.drop_table("artifacts")
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_runs_status"))
        batch_op.drop_index(batch_op.f("ix_runs_dataset_hash"))

    op.drop_table("runs")
    op.drop_table("cache_entries")
