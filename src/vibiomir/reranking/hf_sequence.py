from __future__ import annotations

from dataclasses import dataclass, field

from vibiomir.gpu import apply_gpu_memory_fraction
from vibiomir.schemas import SearchItem, SearchResult

from .base import Reranker


@dataclass(slots=True)
class HFSequenceClassificationReranker(Reranker):
    """Reranker following HF model-card logits for mGTE and MedCPT."""

    model_name: str
    batch_size: int = 32
    device: str | None = None
    max_length: int = 1024
    score_index: int | None = None
    apply_softmax: bool = False
    trust_remote_code: bool = False
    revision: str | None = None
    blend_retrieval: float = 0.0
    _model: object | None = field(default=None, init=False, repr=False)
    _tokenizer: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[object, object]:
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[transformers]` for HF reranking") from exc
            apply_gpu_memory_fraction()
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
            )
            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(target)
            self._model.eval()
        return self._tokenizer, self._model

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        tokenizer, model = self._load()
        target = next(model.parameters()).device
        output: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = tokenizer(
                [query for query, _document in batch],
                [document for _query, document in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded, return_dict=True).logits.float()
            if logits.ndim == 1:
                values = logits
            elif logits.shape[1] == 1:
                values = logits[:, 0]
            else:
                index = self.score_index if self.score_index is not None else logits.shape[1] - 1
                values = (
                    torch.softmax(logits, dim=1)[:, index]
                    if self.apply_softmax
                    else logits[:, index]
                )
            output.extend(float(value) for value in values.cpu())
        return output

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        items: dict[str, SearchItem],
        k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        pairs = [(query, items[result.item_id].text) for result in candidates]
        model_scores = self._score(pairs)
        retrieval = [result.score for result in candidates]
        low, high = min(retrieval), max(retrieval)
        rows: list[SearchResult] = []
        for result, model_score in zip(candidates, model_scores):
            normalized = 0.0 if high == low else (result.score - low) / (high - low)
            rows.append(
                SearchResult(
                    item_id=result.item_id,
                    score=model_score + self.blend_retrieval * normalized,
                    source="hf_sequence_classifier",
                    payload={**result.payload, "first_stage_score": result.score},
                )
            )
        rows.sort(key=lambda row: (-row.score, row.item_id))
        for rank, row in enumerate(rows[:k], 1):
            row.rank = rank
        return rows[:k]
