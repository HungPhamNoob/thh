from __future__ import annotations

from vibiomir.schemas import SearchItem, SearchResult

from .base import Reranker


class IdentityReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        items: dict[str, SearchItem],
        k: int,
    ) -> list[SearchResult]:
        return candidates[:k]

