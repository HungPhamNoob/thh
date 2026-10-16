from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class PyseriniRetriever(Retriever):
    index_dir: str
    name: str = "pyserini_bm25"
    k1: float = 0.9
    b: float = 0.4
    overfetch_factor: int = 5
    _searcher: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> object:
        if self._searcher is None:
            try:
                from pyserini.search.lucene import LuceneSearcher
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[lucene]` for Pyserini") from exc
            self._searcher = LuceneSearcher(self.index_dir)
            self._searcher.set_bm25(self.k1, self.b)
        return self._searcher

    def fit(self, items: Sequence[SearchItem]) -> None:
        # A Lucene index is built once with scripts/build_pyserini_index.sh.
        self._load()

    def search(
        self, query: str, k: int, allowed_ids: set[str] | None = None
    ) -> list[SearchResult]:
        fetch = k if allowed_ids is None else max(k * self.overfetch_factor, k)
        hits = self._load().search(query, fetch)
        results: list[SearchResult] = []
        for hit in hits:
            item_id = str(hit.docid)
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            results.append(
                SearchResult(item_id=item_id, score=float(hit.score), source=self.name)
            )
            if len(results) >= k:
                break
        for rank, result in enumerate(results, 1):
            result.rank = rank
        return results

