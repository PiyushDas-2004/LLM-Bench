"""Programmatic Alembic upgrades so ``llmbench`` needs no alembic.ini."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(url: str) -> Config:
    """Alembic config pointing at the packaged migration scripts."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(engine: Engine, revision: str = "head") -> None:
    """Upgrade the database behind ``engine`` to ``revision``."""
    cfg = alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.begin() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, revision)
