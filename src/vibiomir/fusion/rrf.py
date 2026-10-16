from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from vibiomir.schemas import SearchResult

from .base import Fusion


@dataclass(slots=True)
class ReciprocalRankFusion(Fusion):
    rank_constant: int = 60

    def fuse(
        self,
        runs: dict[str, list[SearchResult]],
        weights: dict[str, float] | None = None,
        k: int | None = None,
    ) -> list[SearchResult]:
        weights = weights or {}
        scores: dict[str, float] = defaultdict(float)
        payloads: dict[str, dict] = {}
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        for source, results in runs.items():
            weight = float(weights.get(source, 1.0))
            for fallback_rank, result in enumerate(results, 1):
                rank = result.rank or fallback_rank
                contribution = weight / (self.rank_constant + rank)
                scores[result.item_id] += contribution
                contributions[result.item_id][source] = contribution
                payloads.setdefault(result.item_id, result.payload)
        ordered = sorted(scores.items(), key=lambda row: (-row[1], row[0]))
        if k is not None:
            ordered = ordered[:k]
        return [
            SearchResult(
                item_id=item_id,
                score=score,
                rank=rank,
                source="rrf",
                payload={**payloads.get(item_id, {}), "fusion": contributions[item_id]},
            )
            for rank, (item_id, score) in enumerate(ordered, 1)
        ]

