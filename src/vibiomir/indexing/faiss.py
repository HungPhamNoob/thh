from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .encoders import build_text_encoder
from .store import ArtifactStore


def build_faiss_index(
    store_path: str | Path,
    stage: str,
    encoder_config: dict[str, Any],
    output_dir: str | Path,
    factory: str = "Flat",
    batch_size: int = 256,
    train_size: int = 100_000,
    storage_dtype: str = "float16",
    languages: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Encode the store once, persist vectors locally, then build a FAISS index.

    ``build_state.json`` records the committed vector offset.  Re-running the
    command resumes model inference after an interruption and rewrites any
    uncommitted partial batch at the same offsets.
    """
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install `vibiomir-lab[dense]` for FAISS indexing") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoder = build_text_encoder(encoder_config)
    language_list = list(languages or [])
    state_path = output / "build_state.json"
    state = _load_state(state_path)
    with ArtifactStore(store_path) as store:
        total = store.count(stage, language_list)
        if total == 0:
            raise ValueError(f"No {stage} items in {store_path}")
        dtype = np.dtype(storage_dtype)
        expected = {
            "stage": stage,
            "total": total,
            "encoder": encoder_config,
            "storage_dtype": storage_dtype,
            "languages": language_list,
        }
        if state:
            for key, value in expected.items():
                if state.get(key) != value:
                    raise ValueError(f"Cannot resume FAISS build: {key} changed")
            offset = int(state["offset"])
            dimension = int(state["dimension"])
            vectors = np.load(output / "vectors.npy", mmap_mode="r+")
        else:
            batches = store.iter_items(
                stage, batch_size=batch_size, languages=language_list
            )
            first_items = next(batches)
            first = encoder.encode_documents([item.text for item in first_items])
            dimension = int(first.shape[1])
            vectors = np.lib.format.open_memmap(
                output / "vectors.npy",
                mode="w+",
                dtype=dtype,
                shape=(total, dimension),
            )
            offset = len(first_items)
            vectors[:offset] = first.astype(dtype)
            vectors.flush()
            _write_state(
                state_path, {**expected, "dimension": dimension, "offset": offset}
            )

        seen = 0
        for items in store.iter_items(
            stage, batch_size=batch_size, languages=language_list
        ):
            batch_end = seen + len(items)
            if batch_end <= offset:
                seen = batch_end
                continue
            remaining = items[max(0, offset - seen) :]
            encoded = encoder.encode_documents([item.text for item in remaining])
            end = offset + len(remaining)
            vectors[offset:end] = encoded.astype(dtype)
            vectors.flush()
            offset = end
            seen = batch_end
            _write_state(
                state_path, {**expected, "dimension": dimension, "offset": offset}
            )
        vectors.flush()
        if offset != total:
            raise RuntimeError(f"Encoded {offset} items but store reported {total}")
        with (output / "ids.jsonl").open("w", encoding="utf-8") as ids:
            for items in store.iter_items(
                stage, batch_size=batch_size, languages=language_list
            ):
                for item in items:
                    ids.write(json.dumps(item.id, ensure_ascii=False) + "\n")

    index = faiss.index_factory(dimension, factory, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        sample_count = min(total, train_size)
        sample_indices = np.linspace(0, total - 1, sample_count, dtype=np.int64)
        index.train(np.asarray(vectors[sample_indices], dtype=np.float32))
    for start in range(0, total, batch_size * 8):
        index.add(np.asarray(vectors[start : start + batch_size * 8], dtype=np.float32))
    faiss.write_index(index, str(output / "index.faiss"))
    manifest = {
        "format_version": 1,
        "stage": stage,
        "items": total,
        "dimension": dimension,
        "factory": factory,
        "metric": "inner_product",
        "encoder": encoder_config,
        "storage_dtype": storage_dtype,
        "languages": language_list,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state_path.unlink(missing_ok=True)
    return manifest


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
