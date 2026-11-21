"""Vector stores behind one interface: in-memory numpy (default) and optional pgvector."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from llmbench.embeddings import FloatMatrix
from llmbench.rag.chunking import Chunk


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk embeddings and answers nearest-neighbour queries (cosine)."""

    def add(self, chunks: list[Chunk], vectors: FloatMatrix) -> None:
        """Index chunks with their unit-norm vectors."""
        ...

    def search(self, vector: FloatMatrix, k: int) -> list[tuple[Chunk, float]]:
        """Top-``k`` chunks by cosine similarity to a unit-norm query vector."""
        ...


class InMemoryVectorStore:
    """Exact cosine search with numpy; fine for corpora up to ~10^5 chunks."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: FloatMatrix | None = None

    def add(self, chunks: list[Chunk], vectors: FloatMatrix) -> None:
        """Append chunks and vectors."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks.extend(chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, vector: FloatMatrix, k: int) -> list[tuple[Chunk, float]]:
        """Exact top-k by dot product (vectors are unit norm)."""
        if self._matrix is None or not self._chunks:
            return []
        sims = self._matrix @ vector.reshape(-1)
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self._chunks[i], float(sims[i])) for i in order]


class PgVectorStore:
    """Postgres + pgvector backend (``pgvector`` extra; see docker-compose ``postgres`` profile).

    Args:
        dsn: libpq connection string,
            e.g. ``postgresql://llmbench:llmbench@localhost:5432/llmbench``.
        dim: Embedding dimension.
        table: Table name (created if missing; quoted as an SQL identifier).
    """

    def __init__(self, dsn: str, dim: int, table: str = "llmbench_chunks") -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg import sql

        name = sql.Identifier(table)
        self.table = table
        self._sql = {
            "create": sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (id text PRIMARY KEY, doc_id text, idx int, "
                "body text, embedding vector({}))"
            ).format(name, sql.Literal(int(dim))),
            "upsert": sql.SQL(
                "INSERT INTO {} (id, doc_id, idx, body, embedding) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET body = EXCLUDED.body, "
                "embedding = EXCLUDED.embedding"
            ).format(name),
            "search": sql.SQL(
                "SELECT id, doc_id, idx, body, 1 - (embedding <=> %s) AS sim FROM {} "
                "ORDER BY embedding <=> %s LIMIT %s"
            ).format(name),
            "clear": sql.SQL("TRUNCATE {}").format(name),
        }
        self._conn: Any = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._conn.execute(self._sql["create"])

    def add(self, chunks: list[Chunk], vectors: FloatMatrix) -> None:
        """Upsert chunks."""
        with self._conn.cursor() as cur:
            for chunk, vec in zip(chunks, vectors, strict=True):
                cur.execute(
                    self._sql["upsert"],
                    (chunk.id, chunk.doc_id, chunk.index, chunk.text, np.asarray(vec)),
                )

    def search(self, vector: FloatMatrix, k: int) -> list[tuple[Chunk, float]]:
        """Top-k by cosine distance (``<=>``)."""
        query = np.asarray(vector).reshape(-1)
        rows = self._conn.execute(self._sql["search"], (query, query, int(k))).fetchall()
        return [(Chunk(id=r[0], doc_id=r[1], index=r[2], text=r[3]), float(r[4])) for r in rows]

    def clear(self) -> None:
        """Delete all rows."""
        self._conn.execute(self._sql["clear"])
