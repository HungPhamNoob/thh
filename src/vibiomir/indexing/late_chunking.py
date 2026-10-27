from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from vibiomir.retrieval.late_chunking import LateChunkingRetriever

from .store import ArtifactStore


def build_late_chunking_faiss_index(
    store_path: str | Path,
    output_dir: str | Path,
    model_config: dict[str, Any],
    query_encoder_config: dict[str, Any],
    factory: str = "Flat",
    train_size: int = 100_000,
    storage_dtype: str = "float16",
    languages: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Persist exact late-chunk vectors, then index them with FAISS."""
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install `vibiomir-lab[dense]` for FAISS indexing") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    language_list = list(languages or [])
    vectorizer = LateChunkingRetriever(**model_config)
    dtype = np.dtype(storage_dtype)
    state_path = output / "build_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    with ArtifactStore(store_path) as store:
        total = store.count("chunk", language_list)
        if total == 0:
            raise ValueError(f"No chunk items in {store_path}")
        expected = {
            "stage": "chunk",
            "total": total,
            "indexing": model_config,
            "encoder": query_encoder_config,
            "storage_dtype": storage_dtype,
            "languages": language_list,
        }
        if state:
            for key, value in expected.items():
                if state.get(key) != value:
                    raise ValueError(f"Cannot resume late chunking build: {key} changed")
            dimension = int(state["dimension"])
            offset = int(state["offset"])
            vectors = np.load(output / "vectors.npy", mmap_mode="r+")
        else:
            dimension = 0
            offset = 0
            vectors = None

        seen = 0
        for document_text, items in store.iter_document_chunks(language_list):
            document_end = seen + len(items)
            if document_end <= offset:
                seen = document_end
                continue
            encoded = vectorizer._document_vectors(document_text, items)
            if vectors is None:
                dimension = int(next(iter(encoded.values())).shape[0])
                vectors = np.lib.format.open_memmap(
                    output / "vectors.npy",
                    mode="w+",
                    dtype=dtype,
                    shape=(total, dimension),
                )
            for item in items[max(0, offset - seen) :]:
                vectors[offset] = encoded[item.id].astype(dtype)
                offset += 1
            vectors.flush()
            _write_state(
                state_path, {**expected, "dimension": dimension, "offset": offset}
            )
            seen = document_end
        if vectors is None:
            raise RuntimeError("Late chunking produced no vectors")
        vectors.flush()
        if offset != total:
            raise RuntimeError(f"Encoded {offset} chunks but store reported {total}")
        with (output / "ids.jsonl").open("w", encoding="utf-8") as ids:
            for _document_text, items in store.iter_document_chunks(language_list):
                for item in items:
                    ids.write(json.dumps(item.id, ensure_ascii=False) + "\n")

    index = faiss.index_factory(dimension, factory, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        sample_count = min(total, train_size)
        sample_indices = np.linspace(0, total - 1, sample_count, dtype=np.int64)
        index.train(np.asarray(vectors[sample_indices], dtype=np.float32))
    for start in range(0, total, 2048):
        index.add(np.asarray(vectors[start : start + 2048], dtype=np.float32))
    faiss.write_index(index, str(output / "index.faiss"))
    manifest = {
        "format_version": 1,
        "stage": "chunk",
        "items": total,
        "dimension": dimension,
        "factory": factory,
        "metric": "inner_product",
        "encoder": query_encoder_config,
        "indexing": {"type": "late_chunking", **model_config},
        "storage_dtype": storage_dtype,
        "languages": language_list,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state_path.unlink(missing_ok=True)
    return manifest


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
