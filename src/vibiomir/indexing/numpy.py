from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .encoders import build_text_encoder
from .store import ArtifactStore


def build_numpy_index(
    store_path: str | Path,
    stage: str,
    encoder_config: dict[str, Any],
    output_dir: str | Path,
    batch_size: int = 256,
    languages: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build an exact local matrix index used for deterministic smoke tests."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoder = build_text_encoder(encoder_config)
    language_list = list(languages or [])
    with ArtifactStore(store_path) as store:
        batches = store.iter_items(stage, batch_size=batch_size, languages=language_list)
        ids: list[str] = []
        encoded_batches: list[np.ndarray] = []
        for items in batches:
            ids.extend(item.id for item in items)
            encoded_batches.append(encoder.encode_documents([item.text for item in items]))
    if not encoded_batches:
        raise ValueError(f"No {stage} items in {store_path}")
    vectors = np.concatenate(encoded_batches).astype(np.float32)
    np.save(output / "vectors.npy", vectors)
    with (output / "ids.jsonl").open("w", encoding="utf-8") as handle:
        for item_id in ids:
            handle.write(json.dumps(item_id, ensure_ascii=False) + "\n")
    manifest = {
        "format_version": 1,
        "stage": stage,
        "items": len(ids),
        "dimension": int(vectors.shape[1]),
        "metric": "inner_product",
        "encoder": encoder_config,
        "languages": language_list,
        "smoke_test_only": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
