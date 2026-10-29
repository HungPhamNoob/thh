from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from vibiomir.retrieval.late_chunking import LateChunkingRetriever

from .qdrant import (
    _payloads,
    _upsert_batch,
    ensure_collection,
    make_qdrant_client,
    resolve_collection_name,
)
from .store import ArtifactStore


def build_late_chunking_qdrant_index(
    store_path: str | Path,
    collection_name: str,
    state_dir: str | Path,
    model_config: dict[str, Any],
    query_encoder_config: dict[str, Any],
    *,
    languages: Iterable[str] | None = None,
    distance: str = "cosine",
    recreate: bool = False,
    on_disk: bool = True,
    store_text_payload: bool = True,
    url_env: str = "QDRANT_URL",
    api_key_env: str = "QDRANT_API_KEY",
    prefix_env: str = "QDRANT_COLLECTION_PREFIX",
    local_path: str | Path | None = None,
    timeout: float = 60.0,
    prefer_grpc: bool = False,
    upload_workers: int = 1,
    quantization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode each long document once and stream pooled chunk vectors to Qdrant."""
    if upload_workers > 1:
        logging.getLogger(__name__).warning(
            "Late chunking uploads serially; upload_workers=%d is ignored", upload_workers
        )
    collection = resolve_collection_name(collection_name, prefix_env)
    state_root = Path(state_dir)
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "build_state.json"
    manifest_path = state_root / "manifest.json"
    language_list = list(languages or [])
    vectorizer = LateChunkingRetriever(**model_config)
    client = make_qdrant_client(
        url_env=url_env,
        api_key_env=api_key_env,
        local_path=local_path,
        timeout=timeout,
        prefer_grpc=prefer_grpc,
    )
    try:
        with ArtifactStore(store_path) as store:
            total = store.count("chunk", language_list)
            if total == 0:
                raise ValueError(f"No chunk items in {store_path}")
            expected = {
                "format_version": 1,
                "backend": "qdrant",
                "stage": "chunk",
                "items": total,
                "collection_name": collection,
                "indexing": model_config,
                "encoder": query_encoder_config,
                "languages": language_list,
                "distance": distance,
                "on_disk": on_disk,
                "store_text_payload": store_text_payload,
                "quantization": quantization,
            }
            state = _load_state(state_path)
            if recreate:
                state = {}
                state_path.unlink(missing_ok=True)
            if state:
                for key, value in expected.items():
                    if state.get(key) != value:
                        raise ValueError(f"Cannot resume late chunking build: {key} changed")
            previous_manifest = _load_state(manifest_path)
            if previous_manifest and not recreate:
                for key, value in expected.items():
                    if previous_manifest.get(key) != value:
                        raise ValueError(f"Cannot reuse late chunking collection: {key} changed")
            dimension = int(state.get("dimension", 0))
            offset = int(state.get("offset", 0))
            if dimension:
                ensure_collection(
                    client,
                    collection,
                    dimension,
                    distance=distance,
                    on_disk=on_disk,
                    quantization=quantization,
                )

            seen = 0
            for document_text, items in store.iter_document_chunks(language_list):
                document_end = seen + len(items)
                if document_end <= offset:
                    seen = document_end
                    continue
                encoded = vectorizer._document_vectors(document_text, items)
                remaining = items[max(0, offset - seen) :]
                vectors = np.asarray([encoded[item.id] for item in remaining], dtype=np.float32)
                if not dimension:
                    dimension = int(vectors.shape[1])
                    ensure_collection(
                        client,
                        collection,
                        dimension,
                        distance=distance,
                        recreate=recreate,
                        on_disk=on_disk,
                        quantization=quantization,
                    )
                elif int(vectors.shape[1]) != dimension:
                    raise ValueError("Late chunking dimension changed during indexing")
                _upsert_batch(
                    client,
                    collection,
                    vectors,
                    remaining,
                    _payloads(store, "chunk", remaining, store_text_payload),
                )
                offset += len(remaining)
                seen = document_end
                _write_state(state_path, {**expected, "dimension": dimension, "offset": offset})
            if offset != total:
                raise RuntimeError(f"Encoded {offset} chunks but store reported {total}")

        indexed = int(client.count(collection_name=collection, exact=True).count)
        if indexed != total:
            raise RuntimeError(
                f"Qdrant collection {collection!r} has {indexed} points, expected {total}; "
                "recreate it to remove stale points"
            )

        manifest = {**expected, "dimension": dimension}
        _write_state(manifest_path, manifest)
        state_path.unlink(missing_ok=True)
        return manifest
    finally:
        client.close()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
