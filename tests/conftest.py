from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep tests hermetic: no real keys, a temp database, no stray .env."""
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LLMBENCH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLMBENCH_DB_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from llmbench.config.settings import get_settings

    get_settings.cache_clear()
