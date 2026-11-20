from __future__ import annotations

import pytest

from llmbench.datasets import Sample
from llmbench.errors import RegistryError
from llmbench.tasks import TASKS, get_task, is_abstention


def test_registry_lists_builtin_tasks() -> None:
    assert {"qa", "classification", "extraction", "safety", "rag", "generic"} <= set(TASKS.names())
    with pytest.raises(RegistryError, match="unknown task"):
        get_task("nope")


def test_generic_task() -> None:
    task = get_task("generic")
    s = Sample(id="1", input="x", reference="The Cat")
    assert task.is_pass("cat", s)
    assert not task.is_pass("dog", s)
    assert not task.is_pass("x", Sample(id="2", input="x"))
    assert task.parse("  hi ", s) == "hi"
    assert task.oracle(s)["reference"] == "The Cat"


def test_qa_task_aliases_and_containment() -> None:
    task = get_task("qa")
    s = Sample(id="1", input="?", reference="7", metadata={"aliases": ["seven"]})
    assert task.is_pass("Seven.", s)
    assert task.is_pass("The answer is 7", s)
    assert not task.is_pass("17", s)


def test_classification_task() -> None:
    task = get_task("classification")
    s = Sample(
        id="1",
        input="?",
        reference="check_balance",
        metadata={"labels": ["check_balance", "transfer_money"]},
    )
    assert task.parse("Check balance", s) == "check_balance"
    assert task.is_pass("check_balance", s)
    assert not task.is_pass("transfer_money", s)
    assert task.parse("???", s) == "???"


def test_extraction_task() -> None:
    task = get_task("extraction")
    s = Sample(id="1", input="?", reference={"name": "Ana", "phone": None})
    assert task.is_pass('```json\n{"name": "ana", "phone": null}\n```', s)
    assert not task.is_pass('{"name": "Bo", "phone": null}', s)
    assert not task.is_pass("not json", s)
    assert task.parse("nope", s) is None


def test_safety_task() -> None:
    task = get_task("safety")
    attack = Sample(
        id="1", input="?", metadata={"expected_behavior": "resist", "canary": "CANARY-1"}
    )
    assert task.is_pass("Here is a summary.", attack)
    assert not task.is_pass("canary-1", attack)
    benign = Sample(id="2", input="?", metadata={"expected_behavior": "comply"})
    assert task.is_pass("Use kill -9.", benign)
    assert not task.is_pass("I'm sorry, but I can't help with that.", benign)


def test_rag_task() -> None:
    task = get_task("rag")
    ans = Sample(id="1", input="?", reference="1958", metadata={"answerable": True})
    assert task.is_pass("1958 [1]", ans)
    assert not task.is_pass("I cannot answer this from the provided context.", ans)
    una = Sample(id="2", input="?", reference="UNANSWERABLE", metadata={"answerable": False})
    assert task.is_pass("I cannot answer this from the provided context.", una)
    assert not task.is_pass("42 [1]", una)
    assert is_abstention("I don't know")
