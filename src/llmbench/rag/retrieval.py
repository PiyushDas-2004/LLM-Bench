"""Retrievers (BM25, vector, hybrid with reciprocal rank fusion) and rerankers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from llmbench.embeddings import Embedder
from llmbench.providers.base import (
    GenerationParams,
    GenerationRequest,
    Message,
    Provider,
    ProviderError,
)
from llmbench.rag.bm25 import BM25
from llmbench.rag.chunking import Chunk
from llmbench.rag.stores import InMemoryVectorStore, VectorStore
from llmbench.text import extract_json

#: Standard RRF constant (Cormack et al., 2009).
RRF_K = 60


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with its score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int


@runtime_checkable
class Retriever(Protocol):
    """Anything that returns ranked chunks for a query; implement this to benchmark your RAG."""

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Top-``k`` chunks for ``query``."""
        ...


def _ranked(pairs: list[tuple[Chunk, float]]) -> list[RetrievedChunk]:
    return [RetrievedChunk(c, s, i + 1) for i, (c, s) in enumerate(pairs)]


class BM25Retriever:
    """Lexical retrieval."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._index = BM25([c.text for c in chunks])

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Top-k by BM25."""
        scores = self._index.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return _ranked([(self.chunks[i], float(scores[i])) for i in order])


class VectorRetriever:
    """Dense retrieval over a :class:`VectorStore`."""

    def __init__(self, embedder: Embedder, store: VectorStore | None = None) -> None:
        self.embedder = embedder
        self.store = store or InMemoryVectorStore()

    async def index(self, chunks: list[Chunk]) -> None:
        """Embed and store chunks."""
        if chunks:
            self.store.add(chunks, await self.embedder.embed([c.text for c in chunks]))

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Top-k by cosine similarity."""
        vector = await self.embedder.embed([query])
        return _ranked(self.store.search(vector[0], k))


class HybridRetriever:
    """BM25 + vector retrieval fused with reciprocal rank fusion: score = sum 1 / (RRF_K + rank)."""

    def __init__(self, lexical: Retriever, dense: Retriever, candidates: int = 20) -> None:
        self.lexical = lexical
        self.dense = dense
        self.candidates = candidates

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Fuse both rankings."""
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        depth = max(k, self.candidates)
        for ranking in (
            await self.lexical.retrieve(query, depth),
            await self.dense.retrieve(query, depth),
        ):
            for item in ranking:
                fused[item.chunk.id] = fused.get(item.chunk.id, 0.0) + 1.0 / (RRF_K + item.rank)
                by_id[item.chunk.id] = item.chunk
        order = sorted(fused, key=lambda cid: (-fused[cid], cid))[:k]
        return _ranked([(by_id[cid], fused[cid]) for cid in order])


@runtime_checkable
class Reranker(Protocol):
    """Re-scores retrieved chunks for a query."""

    async def rerank(self, query: str, items: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Return the best ``k`` items in new order."""
        ...


class LLMReranker:
    """Asks an LLM to rate each chunk's relevance 0-10 (JSON), then sorts by that score."""

    def __init__(self, provider: Provider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def _score(self, query: str, chunk: Chunk) -> float:
        prompt = (
            "Rate how useful the passage is for answering the question, from 0 (irrelevant) "
            'to 10 (essential). Reply ONLY with JSON {"score": <0-10>}.\n\n'
            f"Question: {query}\n\nPassage: {chunk.text}"
        )
        request = GenerationRequest(
            model=self.model,
            messages=(Message(role="user", content=prompt),),
            params=GenerationParams(temperature=0.0, json_mode=True),
            metadata={
                "mock_judge": {
                    "candidate": chunk.text,
                    "reference": query,
                    "scale_min": 0,
                    "scale_max": 10,
                }
            },
        )
        try:
            result = await self.provider.generate(request)
            return float(extract_json(result.text)["score"])
        except (ProviderError, ValueError, KeyError, TypeError):
            return -1.0

    async def rerank(self, query: str, items: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Score every candidate and keep the top ``k`` (ties keep the original order)."""
        scores = [await self._score(query, item.chunk) for item in items]
        order = sorted(range(len(items)), key=lambda i: (-scores[i], items[i].rank))[:k]
        return _ranked([(items[i].chunk, scores[i]) for i in order])


class CrossEncoderReranker:  # pragma: no cover - requires the optional 'embeddings' extra
    """Cross-encoder reranking via ``sentence-transformers``."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, items: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Score (query, passage) pairs and keep the top ``k``."""
        scores = self._model.predict([(query, i.chunk.text) for i in items])
        order = np.argsort(-np.asarray(scores), kind="stable")[:k]
        return _ranked([(items[i].chunk, float(scores[i])) for i in order])


class RerankingRetriever:
    """Retrieve ``candidates`` chunks, then rerank down to ``k``."""

    def __init__(self, base: Retriever, reranker: Reranker, candidates: int = 10) -> None:
        self.base = base
        self.reranker = reranker
        self.candidates = candidates

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Two-stage retrieval."""
        items = await self.base.retrieve(query, max(k, self.candidates))
        return await self.reranker.rerank(query, items, k)


RetrievalMode = Literal["bm25", "vector", "hybrid"]


def describe(mode: RetrievalMode, reranker: str) -> str:
    """Human-readable retriever label for reports."""
    return json.dumps({"retrieval": mode, "reranker": reranker})
