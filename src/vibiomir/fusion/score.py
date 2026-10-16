from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from vibiomir.schemas import SearchResult

from .base import Fusion


@dataclass(slots=True)
class NormalizedScoreFusion(Fusion):
    """Weighted CombSUM after per-retriever min-max normalization."""

    floor: float = 0.0

    def fuse(
        self,
        runs: dict[str, list[SearchResult]],
        weights: dict[str, float] | None = None,
        k: int | None = None,
    ) -> list[SearchResult]:
        weights = weights or {}
        totals: dict[str, float] = defaultdict(float)
        payloads: dict[str, dict] = {}
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        for source, results in runs.items():
            if not results:
                continue
            values = [result.score for result in results]
            low, high = min(values), max(values)
            scale = high - low
            weight = float(weights.get(source, 1.0))
            for result in results:
                normalized = 1.0 if scale <= 1e-12 else (result.score - low) / scale
                contribution = weight * max(self.floor, normalized)
                totals[result.item_id] += contribution
                contributions[result.item_id][source] = contribution
                payloads.setdefault(result.item_id, result.payload)
        ordered = sorted(totals.items(), key=lambda row: (-row[1], row[0]))
        if k is not None:
            ordered = ordered[:k]
        return [
            SearchResult(
                item_id=item_id,
                score=score,
                rank=rank,
                source="combsum",
                payload={**payloads.get(item_id, {}), "fusion": contributions[item_id]},
            )
            for rank, (item_id, score) in enumerate(ordered, 1)
        ]

