from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from vibiomir.schemas import SearchItem, SearchResult
from vibiomir.text import lexical_tokens

from .base import Retriever


@dataclass(slots=True)
class BM25Retriever(Retriever):
    name: str = "bm25"
    k1: float = 1.2
    b: float = 0.75
    analyzer: Callable[[str], list[str]] = lexical_tokens
    _items: dict[str, SearchItem] = field(default_factory=dict, init=False, repr=False)
    _lengths: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _postings: dict[str, list[tuple[str, int]]] = field(default_factory=dict, init=False, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _avg_length: float = field(default=0.0, init=False, repr=False)

    def fit(self, items: Sequence[SearchItem]) -> None:
        self._items = {item.id: item for item in items}
        postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self._lengths = {}
        for item in items:
            counts = Counter(self.analyzer(item.text))
            self._lengths[item.id] = sum(counts.values())
            for term, frequency in counts.items():
                postings[term].append((item.id, frequency))
        count = len(items)
        self._avg_length = sum(self._lengths.values()) / max(count, 1)
        self._postings = dict(postings)
        self._idf = {
            term: math.log(1.0 + (count - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, rows in postings.items()
        }

    def search(
        self, query: str, k: int, allowed_ids: set[str] | None = None
    ) -> list[SearchResult]:
        scores: dict[str, float] = defaultdict(float)
        query_counts = Counter(self.analyzer(query))
        for term, query_frequency in query_counts.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for item_id, frequency in self._postings[term]:
                if allowed_ids is not None and item_id not in allowed_ids:
                    continue
                length = self._lengths[item_id]
                norm = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self._avg_length, 1e-9)
                )
                scores[item_id] += query_frequency * idf * frequency * (self.k1 + 1.0) / norm
        best = heapq.nlargest(k, scores.items(), key=lambda row: row[1])
        return [
            SearchResult(
                item_id=item_id,
                score=float(score),
                rank=rank,
                source=self.name,
                payload=self._items[item_id].payload,
            )
            for rank, (item_id, score) in enumerate(best, 1)
        ]

