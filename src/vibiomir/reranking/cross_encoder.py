from __future__ import annotations

from dataclasses import dataclass, field

from vibiomir.gpu import apply_gpu_memory_limit
from vibiomir.schemas import SearchItem, SearchResult

from .base import Reranker


@dataclass(slots=True)
class CrossEncoderReranker(Reranker):
    model_name: str
    batch_size: int = 32
    device: str | None = None
    max_length: int = 1024
    blend_retrieval: float = 0.0
    trust_remote_code: bool = False
    _model: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[dense]` for cross-encoder reranking") from exc
            apply_gpu_memory_limit()
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
                trust_remote_code=self.trust_remote_code,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        items: dict[str, SearchItem],
        k: int,
    ) -> list[SearchResult]:
        pairs = [(query, items[result.item_id].text) for result in candidates]
        scores = self._load().predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        if candidates:
            retrieval_values = [result.score for result in candidates]
            low, high = min(retrieval_values), max(retrieval_values)
        else:
            low = high = 0.0
        reranked: list[SearchResult] = []
        for result, model_score in zip(candidates, scores):
            retrieval_score = 0.0 if high == low else (result.score - low) / (high - low)
            score = float(model_score) + self.blend_retrieval * retrieval_score
            reranked.append(
                SearchResult(
                    item_id=result.item_id,
                    score=score,
                    source="cross_encoder",
                    payload={**result.payload, "first_stage_score": result.score},
                )
            )
        reranked.sort(key=lambda result: (-result.score, result.item_id))
        for rank, result in enumerate(reranked[:k], 1):
            result.rank = rank
        return reranked[:k]

