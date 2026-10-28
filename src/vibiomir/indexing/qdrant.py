from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from vibiomir.schemas import SearchItem

from .encoders import build_text_encoder
from .store import ArtifactStore


def qdrant_point_id(collection_name: str, item_id: str) -> str:
    """Map an arbitrary competition item id to a stable Qdrant UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vibiomir:{collection_name}:{item_id}"))


def resolve_collection_name(name: str, prefix_env: str = "QDRANT_COLLECTION_PREFIX") -> str:
    prefix = os.getenv(prefix_env, "").strip().strip("_-")
    return f"{prefix}_{name}" if prefix else name


def make_qdrant_client(
    *,
    url_env: str = "QDRANT_URL",
    api_key_env: str = "QDRANT_API_KEY",
    local_path: str | Path | None = None,
    timeout: float = 60.0,
    prefer_grpc: bool = False,
):
    """Create a local or Cloud client without ever putting credentials in YAML."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the project dependencies for Qdrant indexing") from exc
    if local_path:
        return QdrantClient(path=str(local_path))
    url = os.getenv(url_env)
    api_key = os.getenv(api_key_env)
    if not url or not api_key:
        raise RuntimeError(
            f"Set {url_env} and {api_key_env}; credentials must not be committed to config files"
        )
    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=timeout,
        prefer_grpc=prefer_grpc,
    )


def ensure_collection(
    client,
    collection_name: str,
    dimension: int,
    *,
    distance: str = "cosine",
    recreate: bool = False,
    on_disk: bool = True,
) -> None:
    from qdrant_client import models

    distances = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
    }
    try:
        metric = distances[distance.casefold()]
    except KeyError as exc:
        raise ValueError("distance must be cosine, dot, or euclid") from exc
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=metric,
                on_disk=on_disk,
            ),
        )
        return
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        raise ValueError(f"Collection {collection_name!r} uses named vectors; one vector is required")
    if int(vectors.size) != dimension:
        raise ValueError(
            f"Collection {collection_name!r} dimension is {vectors.size}, expected {dimension}"
        )
    if vectors.distance != metric:
        raise ValueError(
            f"Collection {collection_name!r} distance is {vectors.distance}, expected {metric}"
        )


def _payloads(
    store: ArtifactStore,
    stage: str,
    items: list[SearchItem],
    store_text_payload: bool,
) -> Iterator[dict[str, Any]]:
    chunks = store.get_chunks(item.id for item in items) if stage == "chunk" else {}
    for item in items:
        payload: dict[str, Any] = {
            "item_id": item.id,
            "doc_id": item.doc_id,
            "stage": stage,
            "kind": item.kind,
            **item.payload,
        }
        if store_text_payload:
            payload["index_text"] = item.text
            if stage == "chunk":
                payload["chunk_text"] = chunks[item.id].text
        yield payload


def _upsert_batch(
    client,
    collection_name: str,
    vectors: np.ndarray,
    items: list[SearchItem],
    payloads: Iterable[dict[str, Any]],
) -> None:
    from qdrant_client import models

    points = [
        models.PointStruct(
            id=qdrant_point_id(collection_name, item.id),
            vector=np.asarray(vector, dtype=np.float32).tolist(),
            payload=payload,
        )
        for item, vector, payload in zip(items, vectors, payloads)
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)


def build_qdrant_index(
    store_path: str | Path,
    stage: str,
    encoder_config: dict[str, Any],
    collection_name: str,
    *,
    state_dir: str | Path,
    batch_size: int = 256,
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
) -> dict[str, Any]:
    """Stream embeddings to Qdrant with idempotent IDs and resumable local state."""
    collection = resolve_collection_name(collection_name, prefix_env)
    state_root = Path(state_dir)
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "build_state.json"
    manifest_path = state_root / "manifest.json"
    language_list = list(languages or [])
    encoder = build_text_encoder(encoder_config)
    client = make_qdrant_client(
        url_env=url_env,
        api_key_env=api_key_env,
        local_path=local_path,
        timeout=timeout,
        prefer_grpc=prefer_grpc,
    )
    try:
        with ArtifactStore(store_path) as store:
            total = store.count(stage, language_list)
            if total == 0:
                raise ValueError(f"No {stage} items in {store_path}")
            expected = {
                "format_version": 1,
                "backend": "qdrant",
                "stage": stage,
                "items": total,
                "collection_name": collection,
                "encoder": encoder_config,
                "languages": language_list,
                "distance": distance,
                "on_disk": on_disk,
                "store_text_payload": store_text_payload,
            }
            state = _load_json(state_path)
            if recreate:
                state = {}
                state_path.unlink(missing_ok=True)
            if state:
                _validate_resume(state, expected)
            previous_manifest = _load_json(manifest_path)
            if previous_manifest and not recreate:
                _validate_resume(previous_manifest, expected)
            offset = int(state.get("offset", 0))
            dimension = int(state.get("dimension", 0))
            if dimension:
                ensure_collection(
                    client, collection, dimension, distance=distance, on_disk=on_disk
                )

            seen = 0
            for items in store.iter_items(stage, batch_size=batch_size, languages=language_list):
                batch_end = seen + len(items)
                if batch_end <= offset:
                    seen = batch_end
                    continue
                remaining = items[max(0, offset - seen) :]
                vectors = np.asarray(
                    encoder.encode_documents([item.text for item in remaining]),
                    dtype=np.float32,
                )
                if vectors.ndim != 2 or len(vectors) != len(remaining):
                    raise ValueError("Encoder returned an invalid document vector matrix")
                if not dimension:
                    dimension = int(vectors.shape[1])
                    ensure_collection(
                        client,
                        collection,
                        dimension,
                        distance=distance,
                        recreate=recreate,
                        on_disk=on_disk,
                    )
                elif int(vectors.shape[1]) != dimension:
                    raise ValueError("Encoder dimension changed during indexing")
                _upsert_batch(
                    client,
                    collection,
                    vectors,
                    remaining,
                    _payloads(store, stage, remaining, store_text_payload),
                )
                offset += len(remaining)
                seen = batch_end
                _write_json(state_path, {**expected, "dimension": dimension, "offset": offset})
            if offset != total:
                raise RuntimeError(f"Encoded {offset} items but store reported {total}")

        indexed = int(client.count(collection_name=collection, exact=True).count)
        if indexed != total:
            raise RuntimeError(
                f"Qdrant collection {collection!r} has {indexed} points, expected {total}; "
                "rerun with --recreate to remove stale points"
            )

        manifest = {**expected, "dimension": dimension}
        _write_json(manifest_path, manifest)
        state_path.unlink(missing_ok=True)
        return manifest
    finally:
        client.close()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_resume(state: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Cannot resume Qdrant build: {key} changed")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
