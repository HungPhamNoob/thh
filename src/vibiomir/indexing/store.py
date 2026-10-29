from __future__ import annotations

import gzip
import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from vibiomir.schemas import Chunk, Document, SearchItem

# Chunk text is stored exactly once (in documents.text).  Chunks carry the
# character span and an optional index_text override; the verbatim chunk text
# is derived with SQLite substr (character-based, matching Python slicing).
# prepare_store enforces chunk.text == document.text[start:end] at write time,
# so derivation cannot drift from the chunker output.
_CHUNK_TEXT_SQL = "substr(d.text, c.start_char + 1, c.end_char - c.start_char)"
_CHUNK_INDEX_SQL = f"COALESCE(c.index_text_override, {_CHUNK_TEXT_SQL})"


class ArtifactStore:
    """Local document/chunk store used by full-corpus indexes.

    SQLite is deliberately the source of submitted chunk text.  Vector and
    Lucene indexes may be rebuilt, while this store preserves the exact text
    (once, in ``documents``) and the character offsets that derive every chunk
    verbatim for the competition validator.
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

    def metadata(self) -> dict[str, object]:
        """Return reproducibility metadata embedded in the SQLite source of truth."""
        try:
            rows = self.connection.execute("SELECT key, value FROM store_metadata").fetchall()
        except sqlite3.OperationalError:
            return {}
        output: dict[str, object] = {}
        for row in rows:
            try:
                output[str(row["key"])] = json.loads(row["value"])
            except json.JSONDecodeError:
                output[str(row["key"])] = row["value"]
        return output

    def get_documents(self, ids: Iterable[str]) -> dict[str, Document]:
        """Load only requested source documents for submission provenance checks."""
        wanted = list(dict.fromkeys(str(value) for value in ids))
        output: dict[str, Document] = {}
        for start in range(0, len(wanted), 900):
            part = wanted[start : start + 900]
            if not part:
                continue
            placeholders = ",".join("?" for _ in part)
            sql = (
                "SELECT id, title, text, language, url FROM documents "
                f"WHERE id IN ({placeholders})"
            )
            for row in self.connection.execute(sql, part):
                document = Document(
                    id=str(row["id"]),
                    title=str(row["title"]),
                    text=str(row["text"]),
                    language=str(row["language"]),
                    url=str(row["url"]),
                )
                output[document.id] = document
        return output

    def iter_items(
        self,
        stage: str,
        batch_size: int = 2048,
        languages: Iterable[str] | None = None,
    ) -> Iterator[list[SearchItem]]:
        _table(stage)
        where, parameters = _language_filter(
            list(languages or []), "" if stage == "document" else "c."
        )
        if stage == "document":
            sql = f"SELECT id, title, text, language, url FROM documents{where} ORDER BY id"
        else:
            sql = (
                f"SELECT c.id, c.doc_id, {_CHUNK_INDEX_SQL} AS index_text, c.chunk_order, "
                "c.start_char, c.end_char, c.language "
                f"FROM chunks c JOIN documents d ON d.id = c.doc_id{where} ORDER BY c.id"
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
                        payload={
                            "doc_id": row["id"],
                            "language": row["language"],
                            "source_url": row["url"],
                        },
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
                            "chunk_id": row["id"],
                            "chunk_order": row["chunk_order"],
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
                    f"SELECT id, title, text, language, url FROM {table} "
                    f"WHERE id IN ({placeholders})"
                )
            else:
                sql = (
                    f"SELECT c.id, c.doc_id, {_CHUNK_INDEX_SQL} AS index_text, c.chunk_order, "
                    "c.start_char, c.end_char, c.language "
                    f"FROM {table} c JOIN documents d ON d.id = c.doc_id "
                    f"WHERE c.id IN ({placeholders})"
                )
            for row in self.connection.execute(sql, part):
                if stage == "document":
                    item = SearchItem(
                        id=row["id"],
                        text=(f"{row['title']}\n{row['text']}" if row["title"] else row["text"]),
                        doc_id=row["id"],
                        kind="document",
                        payload={
                            "doc_id": row["id"],
                            "language": row["language"],
                            "source_url": row["url"],
                        },
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
                            "chunk_id": row["id"],
                            "chunk_order": row["chunk_order"],
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
        where, parameters = _language_filter(language_list, "d.")
        cursor = self.connection.execute(
            "SELECT d.id AS document_id, d.text AS document_text, "
            f"c.id, c.doc_id, {_CHUNK_INDEX_SQL} AS index_text, c.chunk_order, "
            "c.start_char, c.end_char, "
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
                        "chunk_id": row["id"],
                        "chunk_order": row["chunk_order"],
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
                f"SELECT c.id, c.doc_id, {_CHUNK_TEXT_SQL} AS text, "
                f"{_CHUNK_INDEX_SQL} AS index_text, c.chunk_order, c.start_char, c.end_char, "
                f"c.language, c.metadata FROM chunks c JOIN documents d ON d.id = c.doc_id "
                f"WHERE c.id IN ({placeholders})"
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
        handle = (
            gzip.open(path, "wt", encoding="utf-8", compresslevel=6)
            if path.suffix == ".gz"
            else path.open("w", encoding="utf-8")
        )
        with handle:
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


def _language_filter(languages: list[str], prefix: str = "") -> tuple[str, list[str]]:
    if not languages:
        return "", []
    placeholders = ",".join("?" for _ in languages)
    return f" WHERE {prefix}language IN ({placeholders})", languages


def prepare_store(
    documents: Iterable[Document],
    chunker,
    path: str | Path,
    commit_every: int = 1000,
    store_metadata: dict[str, object] | None = None,
) -> tuple[int, int]:
    """Create a new store atomically from a streaming document iterator."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.executescript(
        f"""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL,
            language TEXT NOT NULL, url TEXT NOT NULL, metadata TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
            index_text_override TEXT, chunk_order INTEGER NOT NULL,
            start_char INTEGER NOT NULL, end_char INTEGER NOT NULL,
            language TEXT NOT NULL, metadata TEXT NOT NULL
        );
        CREATE INDEX chunks_doc_id ON chunks(doc_id);
        CREATE TABLE store_metadata (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE VIEW chunk_provenance AS
        SELECT c.doc_id, c.id AS chunk_id, c.chunk_order,
               {_CHUNK_TEXT_SQL} AS text, {_CHUNK_INDEX_SQL} AS index_text,
               c.start_char, c.end_char, c.language, d.url AS source_url
        FROM chunks c JOIN documents d ON d.id = c.doc_id;
        """
    )
    metadata = {"store_format_version": 3, **(store_metadata or {})}
    connection.executemany(
        "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
        [
            (str(key), json.dumps(value, ensure_ascii=False, default=str))
            for key, value in sorted(metadata.items())
        ],
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
            for chunk in chunks:
                if chunk.doc_id != document.id:
                    raise ValueError(f"Chunk {chunk.id} has wrong doc_id {chunk.doc_id}")
                if chunk.text != document.text[chunk.start_char : chunk.end_char]:
                    raise ValueError(f"Chunk {chunk.id} is not an exact source substring")
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.id,
                        chunk.doc_id,
                        # Store the embedding text only when it differs from the
                        # verbatim chunk text; readers derive it otherwise.
                        chunk.index_text if chunk.index_text != chunk.text else None,
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
