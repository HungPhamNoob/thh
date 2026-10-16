from __future__ import annotations

from abc import ABC, abstractmethod

from vibiomir.schemas import SearchItem, SearchResult


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        items: dict[str, SearchItem],
        k: int,
    ) -> list[SearchResult]:
        raise NotImplementedError

