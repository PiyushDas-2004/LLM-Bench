"""Okapi BM25 lexical retrieval (implemented directly; no external dependency)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from numpy.typing import NDArray

from llmbench.embeddings import tokenize

_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "with",
        "as",
        "at",
        "from",
        "that",
        "this",
        "it",
        "its",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "does",
        "do",
        "did",
    ]
)


def bm25_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens without common stop words."""
    return [t for t in tokenize(text) if t not in _STOP]


class BM25:
    """BM25 index over a list of texts.

    Args:
        texts: Documents to index.
        k1: Term-frequency saturation.
        b: Length normalisation strength.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs = [Counter(bm25_tokens(t)) for t in texts]
        self._lengths = np.array([sum(d.values()) for d in self._docs], dtype=np.float64)
        self._avg = float(self._lengths.mean()) if len(texts) else 0.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(texts)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> NDArray[np.float64]:
        """BM25 score of every indexed text for ``query``."""
        out = np.zeros(len(self._docs), dtype=np.float64)
        if not self._docs or self._avg == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self._lengths / self._avg)
        for term in set(bm25_tokens(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self._docs], dtype=np.float64)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
