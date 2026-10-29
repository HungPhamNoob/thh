from __future__ import annotations

from dataclasses import dataclass, field

from vibiomir.gpu import apply_gpu_memory_limit
from vibiomir.models import resolve_hf_model
from vibiomir.schemas import SearchItem, SearchResult

from .base import Reranker


@dataclass(slots=True)
class BGEM3MultiVectorReranker(Reranker):
    """Score a candidate pool with BGE-M3 dense+sparse+ColBERT views.

    Full-corpus MaxSim is prohibitively expensive without a dedicated multi-vector
    index.  The authors' ``compute_score`` API is exact on a first-stage candidate
    pool and exposes the same three representations as the BGE-M3 paper.
    """

    model_name: str = "BAAI/bge-m3"
    batch_size: int = 8
    max_query_length: int = 512
    max_passage_length: int = 1024
    use_fp16: bool = True
    revision: str | None = None
    weights: list[float] = field(default_factory=lambda: [0.4, 0.2, 0.4])
    score_key: str = "colbert+sparse+dense"
    blend_retrieval: float = 0.0
    _model: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> object:
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[bge]` for BGE-M3 reranking") from exc
            apply_gpu_memory_limit()
            self._model = BGEM3FlagModel(
                resolve_hf_model(self.model_name, self.revision),
                use_fp16=self.use_fp16,
            )
        return self._model

    def _score(self, query: str, documents: list[str]) -> list[float]:
        model = self._load()
        scores = model.compute_score(
            [[query, document] for document in documents],
            batch_size=self.batch_size,
            max_query_length=self.max_query_length,
            max_passage_length=self.max_passage_length,
            weights_for_different_modes=self.weights,
        )
        if self.score_key not in scores:
            raise ValueError(f"BGE-M3 did not return score key {self.score_key!r}")
        return [float(score) for score in scores[self.score_key]]

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        items: dict[str, SearchItem],
        k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        model_scores = self._score(query, [items[result.item_id].text for result in candidates])
        retrieval_scores = [result.score for result in candidates]
        low, high = min(retrieval_scores), max(retrieval_scores)
        output: list[SearchResult] = []
        for result, score in zip(candidates, model_scores):
            normalized = 0.0 if high == low else (result.score - low) / (high - low)
            output.append(
                SearchResult(
                    item_id=result.item_id,
                    score=score + self.blend_retrieval * normalized,
                    source="bge_m3_multivector",
                    payload={**result.payload, "first_stage_score": result.score},
                )
            )
        output.sort(key=lambda row: (-row.score, row.item_id))
        for rank, row in enumerate(output[:k], 1):
            row.rank = rank
        return output[:k]
