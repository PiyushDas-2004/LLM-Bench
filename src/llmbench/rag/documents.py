"""Document ingestion for the reference RAG pipeline (txt / md, optional pdf)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmbench.errors import ConfigError


@dataclass(frozen=True)
class Document:
    """A source document; ``id`` is the file name relative to the corpus root."""

    id: str
    text: str
    source: str


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ConfigError(f"{path}: reading PDFs needs the 'pdf' extra") from exc
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def load_corpus(
    root: str | Path, patterns: tuple[str, ...] = ("*.md", "*.txt", "*.pdf")
) -> list[Document]:
    """Load every matching file under ``root`` (sorted for determinism)."""
    base = Path(root)
    if not base.is_dir():
        raise ConfigError(f"corpus directory not found: {base}")
    files = sorted({p for pattern in patterns for p in base.rglob(pattern) if p.is_file()})
    docs = []
    for path in files:
        text = (
            _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
        )
        docs.append(Document(id=str(path.relative_to(base)), text=text, source=str(path)))
    if not docs:
        raise ConfigError(f"corpus {base} contains no {', '.join(patterns)} files")
    return docs
