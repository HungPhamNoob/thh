from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vibiomir.schemas import Document, Prediction


def write_submission(predictions: Iterable[Prediction], path: str | Path, debug: bool = False) -> None:
    rows = [prediction.debug_dict() if debug else prediction.submission_dict() for prediction in predictions]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def load_submission(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Submission root must be a JSON array")
    return data


def validate_submission(
    rows: list[dict[str, Any]], documents: Iterable[Document] | None = None
) -> list[str]:
    errors: list[str] = []
    document_map = {document.id: document for document in documents or []}
    query_ids: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: expected object")
            continue
        if "id" not in row:
            errors.append(f"{prefix}: missing id")
            continue
        query_id = str(row["id"])
        if query_id in query_ids:
            errors.append(f"{prefix}: duplicate query id {query_id}")
        query_ids.add(query_id)
        docs = row.get("relevant_docs")
        chunks = row.get("relevant_chunks")
        if not isinstance(docs, list) or not all(isinstance(item, (str, int)) for item in docs):
            errors.append(f"{prefix}.relevant_docs: expected list of ids")
        if not isinstance(chunks, list):
            errors.append(f"{prefix}.relevant_chunks: expected list")
            continue
        for chunk_index, chunk in enumerate(chunks):
            chunk_prefix = f"{prefix}.relevant_chunks[{chunk_index}]"
            if not isinstance(chunk, dict) or "doc_id" not in chunk or "chunk_text" not in chunk:
                errors.append(f"{chunk_prefix}: require doc_id and chunk_text")
                continue
            doc_id, text = str(chunk["doc_id"]), chunk["chunk_text"]
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{chunk_prefix}.chunk_text: expected non-empty string")
            if document_map:
                if doc_id not in document_map:
                    errors.append(f"{chunk_prefix}: unknown doc_id {doc_id}")
                elif text not in document_map[doc_id].text:
                    errors.append(f"{chunk_prefix}: chunk_text is not an exact source substring")
    return errors

