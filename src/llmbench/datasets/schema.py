"""Validated golden-dataset records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JSONValue = Any


class Sample(BaseModel):
    """One evaluation item.

    ``reference`` may be a string (QA, classification), a JSON object
    (extraction) or ``None`` (open-ended tasks scored by a judge).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    input: str
    reference: JSONValue = None
    context: str | None = None
    documents: list[str] | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_tag(self) -> str:
        """First tag, or ``"untagged"``; used for stratification."""
        return self.tags[0] if self.tags else "untagged"


class Dataset(BaseModel):
    """A loaded dataset plus its content hash (the dataset version)."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    samples: list[Sample]
    content_hash: str

    def __len__(self) -> int:
        return len(self.samples)

    def by_id(self) -> dict[str, Sample]:
        """Index samples by id."""
        return {s.id: s for s in self.samples}
