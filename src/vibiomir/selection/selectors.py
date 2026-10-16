from __future__ import annotations

from dataclasses import dataclass

from vibiomir.schemas import SearchItem, SearchResult
from vibiomir.text import lexical_tokens


@dataclass(slots=True)
class ThresholdTopKSelector:
    max_k: int
    min_k: int = 1
    relative_threshold: float = 0.0

    def select(
        self, results: list[SearchResult], items: dict[str, SearchItem] | None = None
    ) -> list[SearchResult]:
        if not results:
            return []
        threshold = results[0].score * self.relative_threshold
        selected = [result for result in results[: self.max_k] if result.score >= threshold]
        return results[: self.min_k] if len(selected) < self.min_k else selected


@dataclass(slots=True)
class AdaptiveGapSelector:
    """Recall-oriented variable cutoff at the largest early score drop."""

    max_k: int
    min_k: int = 3
    search_after: int = 2

    def select(
        self, results: list[SearchResult], items: dict[str, SearchItem] | None = None
    ) -> list[SearchResult]:
        candidates = results[: self.max_k]
        if len(candidates) <= self.min_k:
            return candidates
        high, low = candidates[0].score, candidates[-1].score
        scale = max(high - low, 1e-12)
        gaps = [
            (candidates[index].score - candidates[index + 1].score) / scale
            for index in range(len(candidates) - 1)
        ]
        start = max(self.min_k - 1, self.search_after - 1)
        cut = max(range(start, len(gaps)), key=lambda index: gaps[index]) + 1
        return candidates[: max(cut, self.min_k)]


@dataclass(slots=True)
class MMRSelector:
    max_k: int
    lambda_relevance: float = 0.8
    pool_k: int = 100

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        left_tokens, right_tokens = set(lexical_tokens(left)), set(lexical_tokens(right))
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 0.0

    def select(
        self, results: list[SearchResult], items: dict[str, SearchItem] | None = None
    ) -> list[SearchResult]:
        if items is None:
            raise ValueError("MMRSelector requires item texts")
        pool = list(results[: self.pool_k])
        if not pool:
            return []
        high, low = pool[0].score, pool[-1].score
        scale = max(high - low, 1e-12)
        selected: list[SearchResult] = []
        while pool and len(selected) < self.max_k:
            def mmr(candidate: SearchResult) -> float:
                relevance = (candidate.score - low) / scale
                redundancy = max(
                    (
                        self._similarity(
                            items[candidate.item_id].text,
                            items[chosen.item_id].text,
                        )
                        for chosen in selected
                    ),
                    default=0.0,
                )
                return self.lambda_relevance * relevance - (1.0 - self.lambda_relevance) * redundancy

            chosen = max(pool, key=mmr)
            selected.append(chosen)
            pool.remove(chosen)
        return selected

