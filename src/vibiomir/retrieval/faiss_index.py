from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vibiomir.indexing.encoders import build_text_encoder
from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class FaissIndexRetriever(Retriever):
    index_dir: str
    encoder: dict[str, Any] | None = None
    name: str = "faiss"
    overfetch_factor: int = 10
    _index: object | None = field(default=None, init=False, repr=False)
    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _encoder: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._index is not None:
            return
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install `vibiomir-lab[dense]` for FAISS search") from exc
        root = Path(self.index_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        encoder_config = self.encoder or manifest["encoder"]
        self._encoder = build_text_encoder(encoder_config)
        self._index = faiss.read_index(str(root / "index.faiss"))
        with (root / "ids.jsonl").open("r", encoding="utf-8") as handle:
            self._ids = [str(json.loads(line)) for line in handle if line.strip()]
        if int(self._index.ntotal) != len(self._ids):
            raise ValueError("FAISS index and ids.jsonl have different item counts")

    def fit(self, items: Sequence[SearchItem]) -> None:
        # The full-corpus index is built by `vibiomir index build-faiss`.
        self._load()

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        self._load()
        assert self._index is not None and self._encoder is not None
        vector = self._encoder.encode_queries([query])
        fetch = k if allowed_ids is None else min(len(self._ids), max(k, k * self.overfetch_factor))
        scores, offsets = self._index.search(vector, fetch)
        output: list[SearchResult] = []
        for score, offset in zip(scores[0], offsets[0]):
            if offset < 0:
                continue
            item_id = self._ids[int(offset)]
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            output.append(
                SearchResult(
                    item_id=item_id,
                    score=float(score),
                    rank=len(output) + 1,
                    source=self.name,
                )
            )
            if len(output) >= k:
                break
        return output
