from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vibiomir.schemas import SearchItem

from .encoders import build_text_encoder
from .store import ArtifactStore

log = logging.getLogger(__name__)


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
    corpus_revision = str(store.metadata().get("corpus_revision", "unknown"))
    for item in items:
        payload: dict[str, Any] = {
            "item_id": item.id,
            "store_key": item.id,
            "doc_id": item.doc_id,
            "stage": stage,
            "kind": item.kind,
            "corpus_revision": corpus_revision,
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
    _retry(
        lambda: client.upsert(collection_name=collection_name, points=points, wait=True),
        operation=f"upsert {len(points)} points into {collection_name}",
    )


def _retry(
    fn,
    *,
    operation: str,
    attempts: int = 8,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    """Retry transient Qdrant operations with bounded exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - client exposes multiple transport errors
            last_error = exc
            if attempt == attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            log.warning(
                "Qdrant %s failed (%d/%d); retrying in %.1fs: %s",
                operation,
                attempt,
                attempts,
                delay,
                type(exc).__name__,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _verify_qdrant_index(
    client,
    collection: str,
    *,
    total: int,
    dimension: int,
    store: ArtifactStore,
    stage: str,
) -> dict[str, Any]:
    """Check count, payload provenance and an actual vector search."""
    indexed = int(
        _retry(
            lambda: client.count(collection_name=collection, exact=True).count,
            operation=f"count {collection}",
        )
    )
    if indexed != total:
        raise RuntimeError(
            f"Qdrant collection {collection!r} has {indexed} points, expected {total}; "
            "build a new versioned collection instead of deleting this one automatically"
        )
    points, _ = _retry(
        lambda: client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=True,
            with_vectors=True,
        ),
        operation=f"read sample payload from {collection}",
    )
    if not points:
        raise RuntimeError(f"Qdrant collection {collection!r} has no sample point")
    point = points[0]
    payload = dict(point.payload or {})
    item_id = str(payload.get("item_id", ""))
    item = store.get_items(stage, [item_id]).get(item_id)
    if item is None or str(payload.get("doc_id")) != item.doc_id:
        raise RuntimeError(f"Qdrant sample payload does not resolve in store: {item_id!r}")
    expected_revision = str(store.metadata().get("corpus_revision", "unknown"))
    if payload.get("corpus_revision") != expected_revision:
        raise RuntimeError("Qdrant sample payload has the wrong corpus revision")
    vector = point.vector
    if isinstance(vector, dict):
        raise RuntimeError("Qdrant verification requires one unnamed vector")
    if vector is None or len(vector) != dimension:
        raise RuntimeError("Qdrant sample vector has the wrong dimension")
    response = _retry(
        lambda: client.query_points(
            collection_name=collection,
            query=vector,
            limit=1,
            with_payload=True,
        ),
        operation=f"sample search in {collection}",
    )
    if not response.points or not str((response.points[0].payload or {}).get("item_id", "")):
        raise RuntimeError("Qdrant sample search did not return a store item id")
    return {
        "collection_exists": True,
        "point_count": indexed,
        "sample_item_id": item_id,
        "sample_search_item_id": str(response.points[0].payload["item_id"]),
        "payload_provenance": True,
    }


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
    upload_workers: int = 1,
) -> dict[str, Any]:
    """Stream embeddings to Qdrant with idempotent IDs and resumable local state.

    ``upload_workers`` > 1 overlaps the network round-trips of consecutive
    upsert batches (the GPU keeps encoding while uploads are in flight).  The
    checkpointed offset still advances strictly in submission order, so a
    resume only replays batches whose acknowledgement was not durably
    recorded; deterministic point IDs make replays idempotent.
    """
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
            store_metadata = store.metadata()
            expected = {
                "format_version": 2,
                "backend": "qdrant",
                "method": store_metadata.get("method", "unknown"),
                "stage": stage,
                "items": total,
                "collection_name": collection,
                "encoder": encoder_config,
                "model": encoder_config.get("model_name", encoder_config.get("type")),
                "model_revision": encoder_config.get("revision"),
                "corpus_revision": store_metadata.get("corpus_revision", "unknown"),
                "chunker_config": store_metadata.get("chunker_config", {}),
                "code_commit": store_metadata.get("code_commit", "unknown"),
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
                remote_count = int(client.count(collection_name=collection, exact=True).count)
                if remote_count < offset:
                    log.warning(
                        "Qdrant has %d points but local resume offset is %d; replaying from zero",
                        remote_count,
                        offset,
                    )
                    offset = 0

            seen = 0
            base_offset = offset
            sizes: dict[int, int] = {}
            futures: dict[int, Future] = {}
            done: set[int] = set()
            failed: set[int] = set()
            next_ack = 0
            acked_items = 0
            max_in_flight = max(2 * int(upload_workers), 2)
            executor = (
                ThreadPoolExecutor(max_workers=upload_workers) if upload_workers > 1 else None
            )

            def _checkpoint() -> None:
                _write_json(
                    state_path,
                    {**expected, "dimension": dimension, "offset": base_offset + acked_items},
                )

            def _harvest(block: bool, raise_errors: bool = True) -> None:
                nonlocal next_ack, acked_items
                if executor is None:
                    return
                if block:
                    target = futures.get(next_ack)
                    if target is not None and not target.done():
                        wait([target])
                error: Exception | None = None
                for index, future in futures.items():
                    if index in done or index in failed or not future.done():
                        continue
                    if future.cancelled():
                        failed.add(index)
                        continue
                    exception = future.exception()
                    if exception is not None:
                        failed.add(index)
                        if error is None:
                            error = exception
                        continue
                    done.add(index)
                while next_ack in done:
                    acked_items += sizes[next_ack]
                    next_ack += 1
                _checkpoint()
                if error is not None and raise_errors:
                    raise error

            try:
                for items in store.iter_items(stage, batch_size=batch_size, languages=language_list):
                    batch_end = seen + len(items)
                    if batch_end <= base_offset:
                        seen = batch_end
                        continue
                    remaining = items[max(0, base_offset - seen) :]
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
                    # Materialise payloads on the producer thread: SQLite must
                    # not be touched from upload workers.
                    payloads = list(_payloads(store, stage, remaining, store_text_payload))
                    if executor is None:
                        _upsert_batch(client, collection, vectors, remaining, payloads)
                        acked_items += len(remaining)
                        _checkpoint()
                    else:
                        while len(futures) - len(done) - len(failed) >= max_in_flight:
                            _harvest(block=True)
                        index = len(futures)
                        sizes[index] = len(remaining)
                        futures[index] = executor.submit(
                            _upsert_batch, client, collection, vectors, remaining, payloads
                        )
                        _harvest(block=False)
                    seen = batch_end
                if executor is not None:
                    while next_ack < len(futures):
                        _harvest(block=True)
            finally:
                if executor is not None:
                    for future in futures.values():
                        future.cancel()
                    executor.shutdown(wait=True)
                    _harvest(block=False, raise_errors=False)
            offset = base_offset + acked_items
            if offset != total:
                raise RuntimeError(f"Encoded {offset} items but store reported {total}")

            verification = _verify_qdrant_index(
                client,
                collection,
                total=total,
                dimension=dimension,
                store=store,
                stage=stage,
            )

        manifest = {
            **expected,
            "dimension": dimension,
            "vector_size": dimension,
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "verification": verification,
        }
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
