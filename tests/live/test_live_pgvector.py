"""pgvector backend test; needs a running Postgres with the vector extension.

    docker compose --profile postgres up -d postgres
    LLMBENCH_PGVECTOR_URL=postgresql://llmbench:llmbench@localhost:5432/llmbench \
        pytest -m live -o addopts="" tests/live/test_live_pgvector.py
"""

from __future__ import annotations

import os

import pytest

from llmbench.embeddings import HashEmbedder
from llmbench.rag import Chunk

pytestmark = pytest.mark.live


def test_pgvector_round_trip() -> None:
    dsn = os.environ.get("LLMBENCH_PGVECTOR_URL")
    if not dsn:
        pytest.skip("set LLMBENCH_PGVECTOR_URL to run the pgvector test")
    pytest.importorskip("psycopg")
    from llmbench.rag.stores import PgVectorStore

    embedder = HashEmbedder(dim=64)
    store = PgVectorStore(dsn, dim=64, table="llmbench_test_chunks")
    store.clear()
    chunks = [
        Chunk(id=f"c{i}", doc_id="d", text=t, index=i)
        for i, t in enumerate(["pump station", "water tests"])
    ]
    store.add(chunks, embedder.embed_sync([c.text for c in chunks]))
    top = store.search(embedder.embed_sync(["pump station"])[0], 1)
    assert top[0][0].id == "c0"
