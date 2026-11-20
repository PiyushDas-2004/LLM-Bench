from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from llmbench.embeddings import (
    CallableEmbedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
    cosine_matrix,
    make_embedder,
)
from llmbench.text import (
    citations,
    contains_phrase,
    extract_json,
    lcs_length,
    normalize_answer,
    rouge_l,
    strip_citations,
    token_f1,
)


def test_normalize_answer() -> None:
    assert normalize_answer("  The  Eiffel-Tower! ") == "eiffeltower"
    assert normalize_answer("An apple") == "apple"


def test_token_f1_and_rouge() -> None:
    assert token_f1("the cat sat", "cat sat") == 1.0
    assert token_f1("dog", "cat") == 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("a", "") == 1.0  # both empty after article removal
    assert 0 < token_f1("cat sat down", "cat sat") < 1
    assert rouge_l("cat sat on mat", "cat on mat") == pytest.approx(2 * 1 * 0.75 / 1.75)
    assert rouge_l("x", "y") == 0.0
    assert rouge_l("", "") == 1.0
    assert lcs_length([], ["a"]) == 0


_words = st.lists(st.sampled_from(["cat", "dog", "sat", "mat", "on", "red"]), max_size=8)


@given(_words, _words)
def test_overlap_metrics_bounded_and_symmetric(a: list[str], b: list[str]) -> None:
    x, y = " ".join(a), " ".join(b)
    for fn in (token_f1, rouge_l):
        v = fn(x, y)
        assert 0.0 <= v <= 1.0
        assert v == pytest.approx(fn(y, x))
    assert token_f1(x, x) == 1.0


def test_contains_phrase() -> None:
    assert contains_phrase("It is Paris, France", "paris")
    assert not contains_phrase("Parisian", "paris")
    assert not contains_phrase("x", "")


def test_extract_json() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here: {"a": [1, 2]} done') == {"a": [1, 2]}
    assert extract_json("noise { bad [1]") == [1]
    with pytest.raises(ValueError, match="no JSON"):
        extract_json("nothing here {")


def test_citations() -> None:
    assert citations("a [2] b [1] c [2]") == [2, 1]
    assert strip_citations("a [2] b") == "a b"


async def test_hash_embedder_properties() -> None:
    emb = HashEmbedder(dim=64)
    m = await emb.embed(["the cat sat", "the cat sat", "quantum chromodynamics"])
    assert m.shape == (3, 64)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    sims = cosine_matrix(m, m)
    assert sims[0, 1] == pytest.approx(1.0)
    assert sims[0, 2] < 0.5
    assert np.all(emb.embed_sync([""]) == 0)


async def test_callable_embedder_and_factory() -> None:
    async def fn(texts: list[str]) -> list[list[float]]:
        return [[3.0, 4.0] for _ in texts]

    emb = CallableEmbedder("c", fn)
    out = await emb.embed(["x"])
    assert out[0].tolist() == pytest.approx([0.6, 0.8])
    assert (await emb.embed([])).shape[0] == 0
    assert isinstance(make_embedder(None), HashEmbedder)
    assert make_embedder("hash:32").embed_sync(["a"]).shape == (1, 32)
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("bogus")


def test_sentence_transformer_requires_extra() -> None:
    with pytest.raises(ImportError, match="embeddings"):
        SentenceTransformerEmbedder()
