from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
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


CorpusPath = str | Path | Sequence[str | Path]


def _parquet_rows(path: CorpusPath) -> Iterator[dict]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install `vibiomir-lab[ingest]` to read parquet input") from exc
    source = [str(value) for value in path] if isinstance(path, Sequence) and not isinstance(path, (str, bytes, Path)) else str(path)
    dataset = ds.dataset(source, format="parquet", exclude_invalid_files=True)
    yield from dataset.to_batches(batch_size=4096)


def _rows(path: CorpusPath) -> Iterator[dict]:
    if isinstance(path, Sequence) and not isinstance(path, (str, bytes, Path)):
        for batch in _parquet_rows(path):
            yield from batch.to_pylist()
        return
    path = Path(path)
    if path.suffix == ".parquet" or path.is_dir():
        for batch in _parquet_rows(path):
            yield from batch.to_pylist()
        return
    yield from _jsonl(path)


def iter_documents(path: CorpusPath) -> Iterator[Document]:
    for row in _rows(path):
        text = row.get("text")
        if text is None:
            text = row.get("contents")
        if text is None:
            text = row.get("content")
        if text is None:
            raise ValueError(f"Document {row.get('id')} has no text/content field")
        if not str(text).strip():
            continue
        known = {"id", "text", "contents", "content", "title", "language", "lang", "url"}
        yield Document(
            id=str(row["id"]),
            text=str(text),
            title=str(row.get("title", "") or ""),
            language=str(row.get("language", row.get("lang", "")) or ""),
            url=str(row.get("url", "") or ""),
            metadata={key: value for key, value in row.items() if key not in known},
        )


def load_documents(path: str | Path) -> list[Document]:
    return list(iter_documents(path))


def load_queries(path: str | Path) -> list[Query]:
    queries: list[Query] = []
    for row in _rows(path):
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
