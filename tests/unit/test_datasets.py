from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llmbench.datasets import (
    Sample,
    hash_samples,
    load_dataset,
    load_samples,
    stratified_sample,
    train_dev_split,
)
from llmbench.errors import DatasetError


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    return path


def test_load_jsonl_and_hash(tmp_path: Path) -> None:
    f = _write(tmp_path / "d.jsonl", [{"id": "a", "input": "x", "reference": "y", "tags": ["t"]}])
    ds = load_dataset(f)
    assert len(ds) == 1
    assert ds.name == "d"
    assert len(ds.content_hash) == 64
    assert ds.by_id()["a"].primary_tag == "t"


def test_load_yaml_variants(tmp_path: Path) -> None:
    f = tmp_path / "d.yaml"
    f.write_text("samples:\n  - {id: a, input: x}\n")
    assert load_samples(f)[0].primary_tag == "untagged"
    g = tmp_path / "e.yml"
    g.write_text("- {id: a, input: x}\n")
    assert len(load_samples(g)) == 1
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: bar\n")
    with pytest.raises(DatasetError, match="list of mappings"):
        load_samples(bad)


def test_directory_dataset(tmp_path: Path) -> None:
    _write(tmp_path / "questions.jsonl", [{"id": "q", "input": "x"}])
    assert load_samples(tmp_path)[0].id == "q"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetError, match="directory has no"):
        load_samples(empty)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json}\n", "invalid JSON"),
        ("[1, 2]\n", "JSON object"),
        ('{"id": "a"}\n', "invalid"),
        ('{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}\n', "duplicate"),
        ("\n", "empty"),
    ],
)
def test_invalid_jsonl(tmp_path: Path, content: str, match: str) -> None:
    f = tmp_path / "d.jsonl"
    f.write_text(content)
    with pytest.raises(DatasetError, match=match):
        load_samples(f)


def test_missing_and_unsupported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_samples(tmp_path / "nope.jsonl")
    f = tmp_path / "d.csv"
    f.write_text("a,b")
    with pytest.raises(DatasetError, match="unsupported"):
        load_samples(f)


def test_hash_changes_with_content() -> None:
    a = [Sample(id="1", input="x", reference="y")]
    b = [Sample(id="1", input="x", reference="z")]
    assert hash_samples(a) != hash_samples(b)


_json_scalars = st.one_of(st.integers(), st.text(max_size=8), st.booleans(), st.none())


@given(meta=st.dictionaries(st.text(min_size=1, max_size=6), _json_scalars, max_size=6))
def test_hash_stable_under_key_reordering(meta: dict[str, object]) -> None:
    reordered = dict(reversed(list(meta.items())))
    s1 = Sample(id="1", input="x", metadata=meta)
    s2 = Sample(id="1", input="x", metadata=reordered)
    assert hash_samples([s1]) == hash_samples([s2])


@given(st.permutations(list(range(6))))
def test_hash_stable_under_line_reordering(order: list[int]) -> None:
    samples = [Sample(id=f"s{i}", input=str(i)) for i in range(6)]
    assert hash_samples([samples[i] for i in order]) == hash_samples(samples)


def _tagged(n_per_tag: dict[str, int]) -> list[Sample]:
    out = []
    for tag, n in n_per_tag.items():
        out += [Sample(id=f"{tag}-{i}", input="x", tags=[tag]) for i in range(n)]
    return out


def test_stratified_sample_is_proportional_and_reproducible() -> None:
    samples = _tagged({"a": 50, "b": 30, "c": 20})
    picked = stratified_sample(samples, 10, seed=7)
    assert len(picked) == 10
    counts = {t: sum(s.primary_tag == t for s in picked) for t in "abc"}
    assert counts == {"a": 5, "b": 3, "c": 2}
    assert picked == stratified_sample(samples, 10, seed=7)
    assert picked != stratified_sample(samples, 10, seed=8)
    assert stratified_sample(samples, 1000) == samples
    unstrat = stratified_sample(samples, 10, seed=1, stratify=False)
    assert len(unstrat) == 10


def test_train_dev_split() -> None:
    samples = _tagged({"a": 10, "b": 10})
    train, dev = train_dev_split(samples, 0.3, seed=1)
    assert len(dev) == 6
    assert len(train) == 14
    assert not {s.id for s in train} & {s.id for s in dev}
    with pytest.raises(ValueError, match="dev_fraction"):
        train_dev_split(samples, 1.5)


@pytest.mark.parametrize(
    "name",
    [
        "qa_basic.jsonl",
        "extraction_json.jsonl",
        "classification_intent.jsonl",
        "safety_injection.jsonl",
        "rag_mini",
    ],
)
def test_shipped_datasets_validate(repo_root: Path, name: str) -> None:
    ds = load_dataset(repo_root / "datasets" / name)
    assert len(ds) >= 25
