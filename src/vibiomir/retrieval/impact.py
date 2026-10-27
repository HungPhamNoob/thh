from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vibiomir.indexing.sparse import build_sparse_encoder
from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


class _QueryEncoder:
    def __init__(self, config: dict[str, Any]):
        self.encoder = build_sparse_encoder(config)

    def encode(self, query: str) -> dict[str, int]:
        return self.encoder.encode([query])[0]


@dataclass(slots=True)
class LuceneImpactRetriever(Retriever):
    index_dir: str
    encoder: dict[str, Any]
    name: str = "lucene_impact"
    overfetch_factor: int = 5
    _searcher: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> object:
        if self._searcher is None:
            try:
                from pyserini.search.lucene import LuceneImpactSearcher
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[lucene]` for impact search") from exc
            self._searcher = LuceneImpactSearcher(
                self.index_dir, query_encoder=_QueryEncoder(self.encoder)
            )
        return self._searcher

    def fit(self, items: Sequence[SearchItem]) -> None:
        self._load()

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        fetch = k if allowed_ids is None else max(k, k * self.overfetch_factor)
        hits = self._load().search(query, fetch)
        output: list[SearchResult] = []
        for hit in hits:
            item_id = str(hit.docid)
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            output.append(
                SearchResult(
                    item_id=item_id,
                    score=float(hit.score),
                    rank=len(output) + 1,
                    source=self.name,
                )
            )
            if len(output) >= k:
                break
        return output
