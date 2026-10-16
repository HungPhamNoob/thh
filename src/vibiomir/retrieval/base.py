from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from vibiomir.schemas import SearchItem, SearchResult


class Retriever(ABC):
    name: str

    @abstractmethod
    def fit(self, items: Sequence[SearchItem]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query: str, k: int, allowed_ids: set[str] | None = None
    ) -> list[SearchResult]:
        raise NotImplementedError

