from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.config import PromptSpec, parse_experiment
from llmbench.datasets import Sample
from llmbench.errors import PromptError
from llmbench.prompts import PromptTemplate, load_prompt

SAMPLE = Sample(
    id="s1",
    input="What is 2+2?",
    reference="4",
    tags=["math"],
    context="ctx",
    metadata={"labels": ["a", "b"]},
)


def test_render_user_and_system() -> None:
    tpl = PromptTemplate("p", "Q: {{ input }} ({{ tags | join(',') }})", "Ctx: {{ context }}")
    out = tpl.render(SAMPLE)
    assert out.user == "Q: What is 2+2? (math)"
    assert out.system == "Ctx: ctx"
    assert len(out.content_hash) == 64
    assert tpl.content_hash != PromptTemplate("p", "other").content_hash


def test_reference_is_not_available_to_templates() -> None:
    with pytest.raises(PromptError, match="unknown template variables"):
        PromptTemplate("p", "{{ reference }}")


def test_syntax_errors_are_reported() -> None:
    with pytest.raises(PromptError, match="syntax"):
        PromptTemplate("p", "{{ input ")


def test_sandbox_blocks_unsafe_attribute_access() -> None:
    tpl = PromptTemplate("p", "{{ input.__class__.__mro__ }}")
    with pytest.raises(PromptError):
        tpl.render(SAMPLE)


def test_extra_variables_override() -> None:
    tpl = PromptTemplate("p", "{{ context }}")
    assert tpl.render(SAMPLE, context="override").user == "override"


def test_load_prompt_from_files(tmp_path: Path) -> None:
    (tmp_path / "u.j2").write_text("U {{ input }}")
    (tmp_path / "s.j2").write_text("S")
    cfg = parse_experiment(
        {
            "name": "x",
            "dataset": "d",
            "models": [{"id": "m", "provider": "mock"}],
            "prompts": [{"id": "p", "template": "u.j2", "system_template": "s.j2"}],
        },
        tmp_path,
    )
    tpl = load_prompt(cfg.prompts[0], cfg)
    assert tpl.render(SAMPLE).user == "U What is 2+2?"
    assert tpl.render(SAMPLE).system == "S"
    with pytest.raises(PromptError, match="not found"):
        load_prompt(PromptSpec(id="q", template=str(tmp_path / "missing.j2")))
    inline = load_prompt(PromptSpec(id="q", text="hi", system="sys"))
    assert inline.render(SAMPLE).system == "sys"


def test_shipped_prompts_render(repo_root: Path) -> None:
    for path in sorted((repo_root / "prompts").glob("*.j2")):
        PromptTemplate(path.stem, path.read_text()).render(SAMPLE, context="[1] passage")
