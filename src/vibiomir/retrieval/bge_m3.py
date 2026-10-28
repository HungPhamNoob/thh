from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vibiomir.gpu import apply_gpu_memory_limit
from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class BGEM3Retriever(Retriever):
    """Dense or learned-sparse BGE-M3 retrieval; use separate instances for fusion."""

    model_name: str = "BAAI/bge-m3"
    mode: str = "dense"
    name: str = "bge_m3"
    batch_size: int = 16
    max_length: int = 1024
    use_fp16: bool = True
    _model: object | None = field(default=None, init=False, repr=False)
    _items: list[SearchItem] = field(default_factory=list, init=False, repr=False)
    _vectors: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> object:
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[bge]` for BGE-M3 retrieval") from exc
            apply_gpu_memory_limit()
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def _encode(self, texts: list[str]) -> dict[str, object]:
        return self._load().encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=self.mode == "dense",
            return_sparse=self.mode == "sparse",
            return_colbert_vecs=self.mode == "colbert",
        )

    def fit(self, items: Sequence[SearchItem]) -> None:
        if self.mode not in {"dense", "sparse", "colbert"}:
            raise ValueError("BGE-M3 mode must be dense, sparse, or colbert")
        self._items = list(items)
        encoded = self._encode([item.text for item in self._items])
        key = {"dense": "dense_vecs", "sparse": "lexical_weights", "colbert": "colbert_vecs"}[self.mode]
        self._vectors = encoded[key]

    @staticmethod
    def _sparse_dot(left: dict, right: dict) -> float:
        if len(left) > len(right):
            left, right = right, left
        return float(sum(float(value) * float(right.get(key, 0.0)) for key, value in left.items()))

    @staticmethod
    def _maxsim(query: np.ndarray, document: np.ndarray) -> float:
        return float((query @ document.T).max(axis=1).sum())

    def search(
        self, query: str, k: int, allowed_ids: set[str] | None = None
    ) -> list[SearchResult]:
        if self._vectors is None:
            raise RuntimeError("Retriever must be fitted before search")
        encoded = self._encode([query])
        key = {"dense": "dense_vecs", "sparse": "lexical_weights", "colbert": "colbert_vecs"}[self.mode]
        query_vector = encoded[key][0]
        candidates: list[tuple[float, int]] = []
        for index, item in enumerate(self._items):
            if allowed_ids is not None and item.id not in allowed_ids:
                continue
            if self.mode == "dense":
                score = float(np.asarray(self._vectors[index]) @ np.asarray(query_vector))
            elif self.mode == "sparse":
                score = self._sparse_dot(query_vector, self._vectors[index])
            else:
                score = self._maxsim(np.asarray(query_vector), np.asarray(self._vectors[index]))
            candidates.append((score, index))
        best = heapq.nlargest(k, candidates)
        return [
            SearchResult(
                item_id=self._items[index].id,
                score=score,
                rank=rank,
                source=f"{self.name}_{self.mode}",
                payload=self._items[index].payload,
            )
            for rank, (score, index) in enumerate(best, 1)
        ]

