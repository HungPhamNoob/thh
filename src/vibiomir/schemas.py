from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    id: str
    text: str
    title: str = ""
    language: str = ""
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Query:
    id: str
    text: str
    language: str = "vi"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryVariant:
    query_id: str
    text: str
    variant_id: str = "original"
    language: str = "vi"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    index_text: str
    order: int
    start_char: int
    end_char: int
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def submission_dict(self) -> dict[str, Any]:
        """Return only fields accepted by the competition submission format."""
        return {"doc_id": self.doc_id, "chunk_text": self.text, "chunk_order": self.order}


@dataclass(slots=True)
class SearchItem:
    id: str
    text: str
    doc_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    item_id: str
    score: float
    rank: int = 0
    source: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    query_id: str
    relevant_docs: list[str]
    relevant_chunks: list[Chunk]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def submission_dict(self) -> dict[str, Any]:
        try:
            query_id: int | str = int(self.query_id)
        except ValueError:
            query_id = self.query_id
        return {
            "id": query_id,
            "relevant_docs": self.relevant_docs,
            "relevant_chunks": [chunk.submission_dict() for chunk in self.relevant_chunks],
        }

    def debug_dict(self) -> dict[str, Any]:
        data = self.submission_dict()
        data["diagnostics"] = self.diagnostics
        return data


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)

