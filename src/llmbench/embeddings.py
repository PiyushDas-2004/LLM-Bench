"""Embedders: a deterministic hashing embedder (offline, for tests and CI) plus adapters.

All embedders return L2-normalised ``float64`` matrices of shape ``(n, dim)``.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


def _normalise(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: FloatMatrix = matrix / norms
    return out


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into unit vectors."""

    name: str

    async def embed(self, texts: list[str]) -> FloatMatrix:
        """Embed ``texts`` into an ``(n, dim)`` unit-norm matrix."""
        ...


class HashEmbedder:
    """Deterministic feature-hashing embedder over unigrams, bigrams and character trigrams.

    It captures lexical overlap only (no real semantics) but is fast, needs no
    network or model download, and gives identical vectors on every machine,
    which is what tests and offline CI need.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        feats = list(tokens)
        feats += [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for tok in tokens:
            padded = f"#{tok}#"
            feats += [f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed_sync(self, texts: list[str]) -> FloatMatrix:
        """Synchronous embedding (the computation is CPU-only and cheap)."""
        matrix = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                sign = 1.0 if value & 1 else -1.0
                weight = 0.5 if feat.startswith("c:") else 1.0
                matrix[row, (value >> 1) % self.dim] += sign * weight
        return _normalise(matrix)

    async def embed(self, texts: list[str]) -> FloatMatrix:
        """Embed ``texts``."""
        return self.embed_sync(texts)


class CallableEmbedder:
    """Wrap an async ``list[str] -> list[list[float]]`` function (e.g. a provider's embed API)."""

    def __init__(self, name: str, fn: Callable[[list[str]], Awaitable[list[list[float]]]]) -> None:
        self.name = name
        self._fn = fn

    async def embed(self, texts: list[str]) -> FloatMatrix:
        """Embed ``texts`` via the wrapped function."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)
        return _normalise(np.asarray(await self._fn(texts), dtype=np.float64))


class SentenceTransformerEmbedder:
    """Local embeddings via ``sentence-transformers`` (install the ``embeddings`` extra)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedder needs the 'embeddings' extra: "
                "pip install 'llmbench[embeddings]'"
            ) from exc
        self.name = f"st-{model_name}"
        self._model = SentenceTransformer(model_name)  # pragma: no cover - heavy optional dep

    async def embed(self, texts: list[str]) -> FloatMatrix:  # pragma: no cover - optional dep
        """Embed ``texts`` locally."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float64)


def cosine_matrix(a: FloatMatrix, b: FloatMatrix) -> FloatMatrix:
    """Pairwise cosine similarities of two unit-norm matrices."""
    out: FloatMatrix = a @ b.T
    return np.clip(out, -1.0, 1.0)


def make_embedder(spec: str | None) -> Embedder:
    """Build an embedder from a short spec: ``hash``, ``hash:256`` or ``st:<model>``."""
    if not spec or spec == "hash":
        return HashEmbedder()
    kind, _, arg = spec.partition(":")
    if kind == "hash":
        return HashEmbedder(int(arg))
    if kind == "st":
        return SentenceTransformerEmbedder(arg or "all-MiniLM-L6-v2")
    raise ValueError(f"unknown embedder spec {spec!r}; use 'hash', 'hash:<dim>' or 'st:<model>'")
