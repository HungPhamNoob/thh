from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class SentenceTransformerRetriever(Retriever):
    model_name: str
    name: str = "dense"
    batch_size: int = 64
    normalize_embeddings: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    query_prompt_name: str | None = None
    device: str | None = None
    faiss_factory: str | None = None
    trust_remote_code: bool = False
    _model: object | None = field(default=None, init=False, repr=False)
    _items: list[SearchItem] = field(default_factory=list, init=False, repr=False)
    _embeddings: np.ndarray | None = field(default=None, init=False, repr=False)
    _faiss_index: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[dense]` for dense retrieval") from exc
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )
        return self._model

    def fit(self, items: Sequence[SearchItem]) -> None:
        self._items = list(items)
        model = self._load()
        texts = [self.document_prefix + item.text for item in self._items]
        self._embeddings = np.asarray(
            model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        if self.faiss_factory:
            try:
                import faiss
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[dense]` for FAISS indexing") from exc
            dimension = int(self._embeddings.shape[1])
            self._faiss_index = faiss.index_factory(
                dimension, self.faiss_factory, faiss.METRIC_INNER_PRODUCT
            )
            if not self._faiss_index.is_trained:
                self._faiss_index.train(self._embeddings)
            self._faiss_index.add(self._embeddings)

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        if self._embeddings is None:
            raise RuntimeError("Retriever must be fitted before search")
        model = self._load()
        query_vector = np.asarray(
            model.encode(
                [self.query_prefix + query],
                normalize_embeddings=self.normalize_embeddings,
                **({"prompt_name": self.query_prompt_name} if self.query_prompt_name else {}),
            )[0],
            dtype=np.float32,
        )
        if self._faiss_index is not None and allowed_ids is None:
            scores, indices = self._faiss_index.search(query_vector[None, :], k)
            return [
                SearchResult(
                    item_id=self._items[int(index)].id,
                    score=float(score),
                    rank=rank,
                    source=self.name,
                    payload=self._items[int(index)].payload,
                )
                for rank, (score, index) in enumerate(zip(scores[0], indices[0]), 1)
                if index >= 0
            ]
        scores = self._embeddings @ query_vector
        candidates = (
            (float(score), index)
            for index, score in enumerate(scores)
            if allowed_ids is None or self._items[index].id in allowed_ids
        )
        best = heapq.nlargest(k, candidates)
        return [
            SearchResult(
                item_id=self._items[index].id,
                score=score,
                rank=rank,
                source=self.name,
                payload=self._items[index].payload,
            )
            for rank, (score, index) in enumerate(best, 1)
        ]


@dataclass(slots=True)
class AsymmetricSentenceTransformerRetriever(SentenceTransformerRetriever):
    """Separate query/document towers, e.g. MedCPT."""

    document_model_name: str = ""
    _document_model: object | None = field(default=None, init=False, repr=False)

    def _load_document(self) -> object:
        if self._document_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[dense]` for dense retrieval") from exc
            self._document_model = SentenceTransformer(
                self.document_model_name,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )
        return self._document_model

    def fit(self, items: Sequence[SearchItem]) -> None:
        self._items = list(items)
        texts = [self.document_prefix + item.text for item in self._items]
        self._embeddings = np.asarray(
            self._load_document().encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
