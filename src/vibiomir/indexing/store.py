from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from vibiomir.schemas import Chunk, Document, SearchItem


class ArtifactStore:
    """Local document/chunk store used by full-corpus indexes.

    SQLite is deliberately the source of submitted chunk text.  Vector and
    Lucene indexes may be rebuilt, while this store preserves the exact text and
    character offsets needed by the competition validator.
    """

    def __init__(self, path: str | Path, read_only: bool = True):
        self.path = Path(path)
        if read_only:
            uri = f"file:{self.path.resolve()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ArtifactStore:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def count(self, stage: str, languages: Iterable[str] | None = None) -> int:
        table = _table(stage)
        where, parameters = _language_filter(list(languages or []))
        return int(
            self.connection.execute(
                f"SELECT COUNT(*) FROM {table}{where}", parameters
            ).fetchone()[0]
        )

    def iter_items(
        self,
        stage: str,
        batch_size: int = 2048,
        languages: Iterable[str] | None = None,
    ) -> Iterator[list[SearchItem]]:
        _table(stage)
        where, parameters = _language_filter(list(languages or []))
        if stage == "document":
            sql = f"SELECT id, title, text, language FROM documents{where} ORDER BY id"
        else:
            sql = (
                "SELECT id, doc_id, index_text, chunk_order, start_char, end_char, language "
                f"FROM chunks{where} ORDER BY id"
            )
        cursor = self.connection.execute(sql, parameters)
        while rows := cursor.fetchmany(batch_size):
            if stage == "document":
                yield [
                    SearchItem(
                        id=row["id"],
                        text=(f"{row['title']}\n{row['text']}" if row["title"] else row["text"]),
                        doc_id=row["id"],
                        kind="document",
                        payload={"doc_id": row["id"], "language": row["language"]},
                    )
                    for row in rows
                ]
            else:
                yield [
                    SearchItem(
                        id=row["id"],
                        text=row["index_text"],
                        doc_id=row["doc_id"],
                        kind="chunk",
                        payload={
                            "doc_id": row["doc_id"],
                            "order": row["chunk_order"],
                            "start_char": row["start_char"],
                            "end_char": row["end_char"],
                            "language": row["language"],
                        },
                    )
                    for row in rows
                ]

    def get_items(self, stage: str, ids: Iterable[str]) -> dict[str, SearchItem]:
        wanted = list(dict.fromkeys(str(value) for value in ids))
        output: dict[str, SearchItem] = {}
        table = _table(stage)
        for start in range(0, len(wanted), 900):
            part = wanted[start : start + 900]
            placeholders = ",".join("?" for _ in part)
            if stage == "document":
                sql = (
                    f"SELECT id, title, text, language FROM {table} "
                    f"WHERE id IN ({placeholders})"
                )
            else:
                sql = (
                    f"SELECT id, doc_id, index_text, chunk_order, start_char, end_char, language "
                    f"FROM {table} WHERE id IN ({placeholders})"
                )
            for row in self.connection.execute(sql, part):
                if stage == "document":
                    item = SearchItem(
                        id=row["id"],
                        text=(f"{row['title']}\n{row['text']}" if row["title"] else row["text"]),
                        doc_id=row["id"],
                        kind="document",
                        payload={"doc_id": row["id"], "language": row["language"]},
                    )
                else:
                    item = SearchItem(
                        id=row["id"],
                        text=row["index_text"],
                        doc_id=row["doc_id"],
                        kind="chunk",
                        payload={
                            "doc_id": row["doc_id"],
                            "order": row["chunk_order"],
                            "start_char": row["start_char"],
                            "end_char": row["end_char"],
                            "language": row["language"],
                        },
                    )
                output[item.id] = item
        return output

    def iter_document_chunks(
        self, languages: Iterable[str] | None = None
    ) -> Iterator[tuple[str, list[SearchItem]]]:
        """Yield original document text and its chunks without N+1 SQL queries."""
        language_list = list(languages or [])
        where, parameters = _language_filter(language_list)
        if where:
            where = where.replace("language", "d.language", 1)
        cursor = self.connection.execute(
            "SELECT d.id AS document_id, d.text AS document_text, "
            "c.id, c.doc_id, c.index_text, c.chunk_order, c.start_char, c.end_char, "
            f"c.language FROM documents d JOIN chunks c ON c.doc_id = d.id{where} "
            "ORDER BY d.id, c.chunk_order",
            parameters,
        )
        document_id: str | None = None
        document_text = ""
        items: list[SearchItem] = []
        for row in cursor:
            current_id = str(row["document_id"])
            if document_id is not None and current_id != document_id:
                yield document_text, items
                items = []
            document_id = current_id
            document_text = str(row["document_text"])
            items.append(
                SearchItem(
                    id=row["id"],
                    text=row["index_text"],
                    doc_id=row["doc_id"],
                    kind="chunk",
                    payload={
                        "doc_id": row["doc_id"],
                        "order": row["chunk_order"],
                        "start_char": row["start_char"],
                        "end_char": row["end_char"],
                        "language": row["language"],
                    },
                )
            )
        if document_id is not None:
            yield document_text, items

    def get_chunks(self, ids: Iterable[str]) -> dict[str, Chunk]:
        wanted = list(dict.fromkeys(str(value) for value in ids))
        output: dict[str, Chunk] = {}
        for start in range(0, len(wanted), 900):
            part = wanted[start : start + 900]
            placeholders = ",".join("?" for _ in part)
            sql = (
                "SELECT id, doc_id, text, index_text, chunk_order, start_char, end_char, "
                f"language, metadata FROM chunks WHERE id IN ({placeholders})"
            )
            for row in self.connection.execute(sql, part):
                output[row["id"]] = Chunk(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    index_text=row["index_text"],
                    order=int(row["chunk_order"]),
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    language=row["language"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
        return output

    def export_jsonl(
        self,
        stage: str,
        path: str | Path,
        languages: Iterable[str] | None = None,
    ) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for batch in self.iter_items(stage, languages=languages):
                for item in batch:
                    handle.write(
                        json.dumps(
                            {"id": item.id, "contents": item.text, **item.payload},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1
        return count

    def export_colbert(
        self,
        stage: str,
        collection: str | Path,
        id_map: str | Path,
        languages: Iterable[str] | None = None,
    ) -> int:
        collection = Path(collection)
        id_map = Path(id_map)
        collection.parent.mkdir(parents=True, exist_ok=True)
        id_map.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with collection.open("w", encoding="utf-8") as corpus, id_map.open(
            "w", encoding="utf-8"
        ) as mapping:
            for batch in self.iter_items(stage, languages=languages):
                for item in batch:
                    clean = item.text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                    corpus.write(f"{count}\t{clean}\n")
                    mapping.write(json.dumps(item.id, ensure_ascii=False) + "\n")
                    count += 1
        return count


def _table(stage: str) -> str:
    if stage == "document":
        return "documents"
    if stage == "chunk":
        return "chunks"
    raise ValueError("stage must be 'document' or 'chunk'")


def _language_filter(languages: list[str]) -> tuple[str, list[str]]:
    if not languages:
        return "", []
    placeholders = ",".join("?" for _ in languages)
    return f" WHERE language IN ({placeholders})", languages


def prepare_store(
    documents: Iterable[Document], chunker, path: str | Path, commit_every: int = 1000
) -> tuple[int, int]:
    """Create a new store atomically from a streaming document iterator."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL,
            language TEXT NOT NULL, url TEXT NOT NULL, metadata TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, text TEXT NOT NULL,
            index_text TEXT NOT NULL, chunk_order INTEGER NOT NULL,
            start_char INTEGER NOT NULL, end_char INTEGER NOT NULL,
            language TEXT NOT NULL, metadata TEXT NOT NULL
        );
        CREATE INDEX chunks_doc_id ON chunks(doc_id);
        """
    )
    document_count = chunk_count = 0
    try:
        for document in documents:
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    document.id,
                    document.title,
                    document.text,
                    document.language,
                    document.url,
                    json.dumps(document.metadata, ensure_ascii=False, default=str),
                ),
            )
            chunks = chunker.split(document)
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.index_text,
                        chunk.order,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.language,
                        json.dumps(chunk.metadata, ensure_ascii=False, default=str),
                    )
                    for chunk in chunks
                ],
            )
            document_count += 1
            chunk_count += len(chunks)
            if document_count % commit_every == 0:
                connection.commit()
        connection.commit()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    if path.exists():
        path.unlink()
    os.replace(temporary, path)
    return document_count, chunk_count
