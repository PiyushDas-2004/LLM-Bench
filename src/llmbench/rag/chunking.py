"""Chunking strategies: fixed-size, sentence-packing and recursive splitting.

Sizes are in characters; overlap is in characters for ``fixed``/``recursive``
and approximated by whole trailing sentences for ``sentence``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmbench.rag.documents import Document

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Chunk:
    """A contiguous piece of a document."""

    id: str
    doc_id: str
    text: str
    index: int


class ChunkingConfig(BaseModel):
    """How to split documents."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["fixed", "sentence", "recursive"] = "recursive"
    size: int = Field(default=500, ge=50)
    overlap: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller(self) -> ChunkingConfig:
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        return self


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (regex heuristic, keeps paragraph breaks as boundaries)."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if para:
            out.extend(s.strip() for s in _SENTENCE.split(para) if s.strip())
    return out


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        length, end = 0, start
        while end < len(words) and (length + len(words[end]) + 1 <= size or end == start):
            length += len(words[end]) + 1
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        back, overlap_len = end, 0
        while back > start + 1 and overlap_len + len(words[back - 1]) + 1 <= overlap:
            back -= 1
            overlap_len += len(words[back]) + 1
        start = back
    return chunks


def _sentence(text: str, size: int, overlap: int) -> list[str]:
    sentences: list[str] = []
    for sentence in split_sentences(text):
        sentences.extend(_fixed(sentence, size, 0) if len(sentence) > size else [sentence])
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join([*current, sentence])) > size:
            chunks.append(" ".join(current))
            carry: list[str] = []
            for prev in reversed(current):
                if len(" ".join([prev, *carry])) > overlap:
                    break
                carry.insert(0, prev)
            current = carry
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


_SEPARATORS = ("\n\n", "\n", ". ", " ")


def _recursive(text: str, size: int, overlap: int, depth: int = 0) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [" ".join(text.split())] if text else []
    if depth >= len(_SEPARATORS):
        return _fixed(text, size, overlap)
    sep = _SEPARATORS[depth]
    pieces = [p for p in text.split(sep) if p.strip()]
    if len(pieces) == 1:
        return _recursive(text, size, overlap, depth + 1)
    merged: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            merged.append(current)
        if len(piece) > size:
            merged.extend(_recursive(piece, size, overlap, depth + 1))
            current = ""
        else:
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}{sep}{piece}" if tail else piece
    if current:
        merged.append(current)
    return [" ".join(m.split()) for m in merged if m.strip()]


def chunk_document(doc: Document, config: ChunkingConfig) -> list[Chunk]:
    """Split one document with the configured strategy."""
    splitter = {"fixed": _fixed, "sentence": _sentence, "recursive": _recursive}[config.strategy]
    texts = splitter(doc.text, config.size, config.overlap)
    return [Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=t, index=i) for i, t in enumerate(texts)]


def chunk_corpus(docs: list[Document], config: ChunkingConfig) -> list[Chunk]:
    """Split every document."""
    return [chunk for doc in docs for chunk in chunk_document(doc, config)]
