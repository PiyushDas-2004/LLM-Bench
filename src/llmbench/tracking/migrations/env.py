"""Alembic environment: runs migrations on a connection handed in by LLMBench."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from llmbench.tracking.db import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    """Run migrations using the provided connection, or build one from the URL."""
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()
        return
    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


def run_migrations_offline() -> None:
    """Emit SQL for the configured URL without a live connection."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
