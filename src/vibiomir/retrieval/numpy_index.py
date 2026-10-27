from __future__ import annotations

import heapq
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vibiomir.indexing.encoders import build_text_encoder
from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class NumpyIndexRetriever(Retriever):
    """Exact persisted matrix search for smoke tests and tiny corpora."""

    index_dir: str
    encoder: dict[str, Any] | None = None
    name: str = "numpy_index"
    block_size: int = 8192
    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _vectors: np.ndarray | None = field(default=None, init=False, repr=False)
    _encoder: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._vectors is not None:
            return
        root = Path(self.index_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self._encoder = build_text_encoder(self.encoder or manifest["encoder"])
        self._vectors = np.load(root / "vectors.npy", mmap_mode="r")
        with (root / "ids.jsonl").open("r", encoding="utf-8") as handle:
            self._ids = [str(json.loads(line)) for line in handle if line.strip()]
        if len(self._ids) != len(self._vectors):
            raise ValueError("NumPy index and ids.jsonl have different item counts")

    def fit(self, items: Sequence[SearchItem]) -> None:
        self._load()

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        self._load()
        assert self._vectors is not None and self._encoder is not None
        query_vector = self._encoder.encode_queries([query])[0]
        candidates: list[tuple[float, int]] = []
        for start in range(0, len(self._ids), self.block_size):
            block = np.asarray(self._vectors[start : start + self.block_size])
            scores = block @ query_vector
            for relative, score in enumerate(scores):
                offset = start + relative
                if allowed_ids is None or self._ids[offset] in allowed_ids:
                    candidates.append((float(score), offset))
        output: list[SearchResult] = []
        for rank, (score, offset) in enumerate(heapq.nlargest(k, candidates), 1):
            output.append(
                SearchResult(
                    item_id=self._ids[offset],
                    score=score,
                    rank=rank,
                    source=self.name,
                )
            )
        return output
