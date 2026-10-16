from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from vibiomir.schemas import SearchResult


@dataclass(slots=True)
class DocumentAggregator:
    """Aggregate passage evidence into document scores (MaxP/PARADE-style baselines)."""

    method: str = "topk_mean"
    top_k: int = 3
    temperature: float = 1.0
    position_decay: float = 0.0

    def aggregate(self, chunk_results: list[SearchResult]) -> list[SearchResult]:
        grouped: dict[str, list[SearchResult]] = defaultdict(list)
        for result in chunk_results:
            doc_id = str(result.payload["doc_id"])
            grouped[doc_id].append(result)
        documents: list[SearchResult] = []
        for doc_id, rows in grouped.items():
            scored: list[float] = []
            for row in rows:
                order = int(row.payload.get("order", 0))
                scored.append(row.score * math.exp(-self.position_decay * order))
            scored.sort(reverse=True)
            if self.method == "max":
                score = scored[0]
            elif self.method == "topk_mean":
                top = scored[: self.top_k]
                score = sum(top) / len(top)
            elif self.method == "logsumexp":
                maximum = max(scored)
                total = sum(math.exp((value - maximum) / self.temperature) for value in scored)
                score = maximum + self.temperature * math.log(total)
            else:
                raise ValueError(f"Unknown document aggregation method: {self.method}")
            documents.append(
                SearchResult(
                    item_id=doc_id,
                    score=score,
                    source=f"chunk_{self.method}",
                    payload={"doc_id": doc_id, "matched_chunks": len(rows)},
                )
            )
        documents.sort(key=lambda row: (-row.score, row.item_id))
        for rank, result in enumerate(documents, 1):
            result.rank = rank
        return documents

