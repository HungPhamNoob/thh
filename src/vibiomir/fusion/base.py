from __future__ import annotations

from abc import ABC, abstractmethod

from vibiomir.schemas import SearchResult


class Fusion(ABC):
    @abstractmethod
    def fuse(
        self,
        runs: dict[str, list[SearchResult]],
        weights: dict[str, float] | None = None,
        k: int | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError

