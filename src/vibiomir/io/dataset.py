from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from vibiomir.schemas import Chunk, Document, Query


def _jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def load_documents(path: str | Path) -> list[Document]:
    documents: list[Document] = []
    for row in _jsonl(path):
        text = row.get("text") or row.get("contents") or row.get("content")
        if text is None:
            raise ValueError(f"Document {row.get('id')} has no text/content field")
        known = {"id", "text", "contents", "content", "title", "language", "lang", "url"}
        documents.append(
            Document(
                id=str(row["id"]),
                text=str(text),
                title=str(row.get("title", "")),
                language=str(row.get("language", row.get("lang", ""))),
                url=str(row.get("url", "")),
                metadata={key: value for key, value in row.items() if key not in known},
            )
        )
    return documents


def load_queries(path: str | Path) -> list[Query]:
    queries: list[Query] = []
    for row in _jsonl(path):
        text = row.get("text") or row.get("query") or row.get("question")
        if text is None:
            raise ValueError(f"Query {row.get('id')} has no text/query/question field")
        queries.append(
            Query(
                id=str(row["id"]),
                text=str(text),
                language=str(row.get("language", row.get("lang", "vi"))),
            )
        )
    return queries


def write_documents_jsonl(documents: Iterable[Document], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {
                        "id": document.id,
                        "text": document.text,
                        "title": document.title,
                        "language": document.language,
                        "url": document.url,
                        **document.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_chunks_jsonl(chunks: Iterable[Chunk], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "id": chunk.id,
                        "contents": chunk.index_text,
                        "doc_id": chunk.doc_id,
                        "chunk_text": chunk.text,
                        "chunk_order": chunk.order,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "language": chunk.language,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

