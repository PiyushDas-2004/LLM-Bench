from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from llmbench import __version__
from llmbench.config import (
    ExperimentConfig,
    PricingTable,
    PromptSpec,
    Settings,
    load_experiment,
    parse_experiment,
)
from llmbench.errors import ConfigError

MINIMAL = {
    "name": "t",
    "dataset": "d.jsonl",
    "models": [{"id": "m", "provider": "mock"}],
    "prompts": [{"id": "p", "text": "{{ input }}"}],
}


def test_version_is_exposed() -> None:
    assert __version__


def test_metrics_accept_strings_and_mappings(tmp_path: Path) -> None:
    data = {**MINIMAL, "metrics": ["exact_match", {"name": "f1", "params": {"x": 1}}]}
    cfg = parse_experiment(data, tmp_path)
    assert [m.name for m in cfg.metrics] == ["exact_match", "f1"]
    assert cfg.metrics[1].params == {"x": 1}


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    data = {**MINIMAL, "models": [{"id": "m", "provider": "mock"}] * 2}
    with pytest.raises(ConfigError, match="duplicate model ids"):
        parse_experiment(data, tmp_path)


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        parse_experiment({**MINIMAL, "bogus": 1}, tmp_path)


def test_non_mapping_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        parse_experiment([1, 2], tmp_path)


def test_prompt_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PromptSpec(id="p")
    with pytest.raises(ValueError, match="exactly one"):
        PromptSpec(id="p", text="a", template="b")
    with pytest.raises(ValueError, match="at most one"):
        PromptSpec(id="p", text="a", system="s", system_template="t")


def test_placeholder_models_detected(tmp_path: Path) -> None:
    data = {
        **MINIMAL,
        "models": [
            {"id": "a", "provider": "openai", "model": "<set me>"},
            {"id": "b", "provider": "openai", "model": "real-model"},
            {"id": "c", "provider": "mock"},
        ],
    }
    cfg = parse_experiment(data, tmp_path)
    assert [m.is_placeholder for m in cfg.models] == [True, False, False]
    assert cfg.model_by_id("c").model_name == "mock"
    assert cfg.prompt_by_id("p").text == "{{ input }}"
    with pytest.raises(KeyError):
        cfg.model_by_id("zzz")
    with pytest.raises(KeyError):
        cfg.prompt_by_id("zzz")


def test_resolve_path_prefers_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "here.txt").write_text("x")
    other = tmp_path / "cwd"
    other.mkdir()
    (other / "there.txt").write_text("y")
    monkeypatch.chdir(other)
    cfg = parse_experiment(MINIMAL, cfg_dir)
    assert cfg.resolve_path("here.txt") == (cfg_dir / "here.txt").resolve()
    assert cfg.resolve_path("there.txt") == (other / "there.txt").resolve()
    assert cfg.resolve_path("missing.txt") == (cfg_dir / "missing.txt").resolve()
    assert cfg.resolve_path(str(tmp_path)) == tmp_path


def test_load_experiment_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_experiment(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1,")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_experiment(bad)


def test_load_experiment_from_file(tmp_path: Path) -> None:
    f = tmp_path / "e.yaml"
    f.write_text(
        "name: x\ndataset: d.jsonl\nmodels: [{id: m, provider: mock}]\n"
        "prompts: [{id: p, text: hi}]\n"
    )
    cfg = load_experiment(f)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.base_dir == tmp_path.resolve()


def test_pricing_lookup_and_cost(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text(
        "as_of: 2026-01-01\nmodels:\n  openai/x: {input_per_1m: 1.0, output_per_1m: 2.0}\n"
        "  gemini/*: {input_per_1m: 0.5, output_per_1m: 0.5}\n"
    )
    table = PricingTable.from_yaml(f)
    assert table.as_of == dt.date(2026, 1, 1)
    assert table.cost("openai", "x", 1_000_000, 500_000) == pytest.approx(2.0)
    assert table.cost("gemini", "anything", 2_000_000, 0) == pytest.approx(1.0)
    assert table.cost("openai", "unknown", 10, 10) is None
    assert PricingTable.empty().models == {}


def test_settings_reads_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("LLMBENCH_DB_URL", "sqlite://")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test-not-real"
    assert "sk-test" not in repr(settings)
    assert settings.db_url == "sqlite://"
