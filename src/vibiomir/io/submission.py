from __future__ import annotations

import json
import zipfile
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


def write_submission_zip(submission: str | Path, output: str | Path) -> None:
    """Create the organizer format: one root-level file named submission.json."""
    submission_path = Path(submission)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, arcname="submission.json")


def validate_submission(
    rows: list[dict[str, Any]],
    documents: Iterable[Document] | None = None,
    expected_query_ids: set[str] | None = None,
    require_numeric_ids: bool = False,
    strict_fields: bool = False,
) -> list[str]:
    errors: list[str] = []
    check_documents = documents is not None
    document_map = {document.id: document for document in documents or []}
    query_ids: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: expected object")
            continue
        if strict_fields:
            unknown = sorted(set(row) - {"id", "relevant_docs", "relevant_chunks"})
            if unknown:
                errors.append(f"{prefix}: unexpected fields {unknown}")
        if "id" not in row:
            errors.append(f"{prefix}: missing id")
            continue
        query_id = str(row["id"])
        if require_numeric_ids and not isinstance(row["id"], int):
            errors.append(f"{prefix}.id: expected integer")
        if query_id in query_ids:
            errors.append(f"{prefix}: duplicate query id {query_id}")
        query_ids.add(query_id)
        docs = row.get("relevant_docs")
        chunks = row.get("relevant_chunks")
        if not isinstance(docs, list) or not all(isinstance(item, (str, int)) for item in docs):
            errors.append(f"{prefix}.relevant_docs: expected list of ids")
            doc_ids: set[str] = set()
        elif require_numeric_ids and not all(isinstance(item, int) for item in docs):
            errors.append(f"{prefix}.relevant_docs: expected integer ids")
            doc_ids = {str(item) for item in docs}
        else:
            doc_ids = {str(item) for item in docs}
        if isinstance(docs, list) and len(doc_ids) != len(docs):
            errors.append(f"{prefix}.relevant_docs: duplicate ids")
        if check_documents:
            for doc_id in sorted(doc_ids - set(document_map)):
                errors.append(f"{prefix}.relevant_docs: unknown doc_id {doc_id}")
        if not isinstance(chunks, list):
            errors.append(f"{prefix}.relevant_chunks: expected list")
            continue
        for chunk_index, chunk in enumerate(chunks):
            chunk_prefix = f"{prefix}.relevant_chunks[{chunk_index}]"
            if not isinstance(chunk, dict) or "doc_id" not in chunk or "chunk_text" not in chunk:
                errors.append(f"{chunk_prefix}: require doc_id and chunk_text")
                continue
            if strict_fields:
                unknown = sorted(set(chunk) - {"doc_id", "chunk_text", "chunk_order"})
                if unknown:
                    errors.append(f"{chunk_prefix}: unexpected fields {unknown}")
            doc_id, text = str(chunk["doc_id"]), chunk["chunk_text"]
            if doc_id not in doc_ids:
                errors.append(f"{chunk_prefix}: doc_id is absent from relevant_docs")
            if require_numeric_ids and not isinstance(chunk["doc_id"], int):
                errors.append(f"{chunk_prefix}.doc_id: expected integer")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{chunk_prefix}.chunk_text: expected non-empty string")
            if "chunk_order" in chunk and (
                not isinstance(chunk["chunk_order"], int) or chunk["chunk_order"] < 0
            ):
                errors.append(f"{chunk_prefix}.chunk_order: expected non-negative integer")
            if check_documents:
                if doc_id not in document_map:
                    errors.append(f"{chunk_prefix}: unknown doc_id {doc_id}")
                elif text not in document_map[doc_id].text:
                    errors.append(f"{chunk_prefix}: chunk_text is not an exact source substring")
    if expected_query_ids is not None:
        expected = {str(value) for value in expected_query_ids}
        missing = sorted(expected - query_ids)
        extra = sorted(query_ids - expected)
        if missing:
            errors.append(f"missing query ids: {missing[:20]}" + (" ..." if len(missing) > 20 else ""))
        if extra:
            errors.append(f"unexpected query ids: {extra[:20]}" + (" ..." if len(extra) > 20 else ""))
    return errors

