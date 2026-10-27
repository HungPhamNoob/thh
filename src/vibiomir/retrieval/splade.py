from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class SpladeRetriever(Retriever):
    """Exact SPLADE max-pooling scorer for subset experiments and smoke tests.

    It implements the authors' ``max(log(1 + relu(MLM logits)))`` representation.
    Full-corpus runs should export these weights to a Lucene impact index.
    """

    model_name: str = "naver/splade-cocondenser-ensembledistil"
    name: str = "splade"
    batch_size: int = 8
    max_length: int = 256
    device: str | None = None
    _model: object | None = field(default=None, init=False, repr=False)
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _items: list[SearchItem] = field(default_factory=list, init=False, repr=False)
    _vectors: np.ndarray | None = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[object, object]:
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[transformers]` for SPLADE") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(target)
            self._model.eval()
        return self._tokenizer, self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch

        tokenizer, model = self._load()
        target = next(model.parameters()).device
        output: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
                weights = torch.log1p(torch.relu(logits))
                weights = weights * encoded["attention_mask"].unsqueeze(-1)
                vectors = weights.max(dim=1).values
            output.append(vectors.float().cpu().numpy())
        return np.concatenate(output).astype(np.float32)

    def fit(self, items: Sequence[SearchItem]) -> None:
        self._items = list(items)
        self._vectors = self._encode([item.text for item in self._items])

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        if self._vectors is None:
            raise RuntimeError("Retriever must be fitted before search")
        query_vector = self._encode([query])[0]
        scores = self._vectors @ query_vector
        candidates = (
            (float(score), index)
            for index, score in enumerate(scores)
            if allowed_ids is None or self._items[index].id in allowed_ids
        )
        return [
            SearchResult(
                item_id=self._items[index].id,
                score=score,
                rank=rank,
                source=self.name,
                payload=self._items[index].payload,
            )
            for rank, (score, index) in enumerate(heapq.nlargest(k, candidates), 1)
        ]
