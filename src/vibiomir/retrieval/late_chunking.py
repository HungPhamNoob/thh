from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class LateChunkingRetriever(Retriever):
    """Encode each full document first, then pool token vectors over chunk spans.

    This is the defining operation in Jina AI's Late Chunking release.  It is
    intentionally separate from adding neighboring text to ``index_text``.
    """

    model_name: str = "jinaai/jina-embeddings-v2-base-multilingual"
    name: str = "late_chunking"
    max_length: int = 8192
    overlap_tokens: int = 512
    device: str | None = None
    trust_remote_code: bool = True
    revision: str | None = None
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _items: list[SearchItem] = field(default_factory=list, init=False, repr=False)
    _vectors: np.ndarray | None = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[object, object]:
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError("Install `vibiomir-lab[transformers]` for late chunking") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
            )
            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(target)
            self._model.eval()
        return self._tokenizer, self._model

    def _document_vectors(self, text: str, items: list[SearchItem]) -> dict[str, np.ndarray]:
        import torch

        tokenizer, model = self._load()
        target = next(model.parameters()).device
        encoded = tokenizer(
            text,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(target) for key, value in encoded.items()}
        token_count = int(encoded["input_ids"].shape[1])
        if token_count <= self.max_length:
            with torch.no_grad():
                hidden = model(**encoded).last_hidden_state[0]
        else:
            if not 0 <= self.overlap_tokens < self.max_length:
                raise ValueError("overlap_tokens must be in [0, max_length)")
            windows = []
            step = self.max_length - self.overlap_tokens
            for start in range(0, token_count, step):
                end = min(start + self.max_length, token_count)
                batch = {key: value[:, start:end] for key, value in encoded.items()}
                with torch.no_grad():
                    window = model(**batch).last_hidden_state[0]
                if start:
                    window = window[self.overlap_tokens :]
                windows.append(window)
            hidden = torch.cat(windows, dim=0)
        output: dict[str, np.ndarray] = {}
        for item in items:
            start = int(item.payload["start_char"])
            end = int(item.payload["end_char"])
            token_indices = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > token_start and token_start < end and token_end > start
            ]
            if not token_indices:
                vector = torch.zeros(hidden.shape[-1], device=hidden.device)
            else:
                vector = hidden[token_indices].mean(dim=0)
            # The authors' chunked_pooling averages token embeddings without an
            # extra per-chunk normalization step.
            output[item.id] = vector.float().cpu().numpy()
        return output

    def _query_vector(self, query: str) -> np.ndarray:
        import torch

        tokenizer, model = self._load()
        target = next(model.parameters()).device
        encoded = tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(target) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"]
        vector = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        vector = torch.nn.functional.normalize(vector.float(), p=2, dim=1)
        return vector[0].cpu().numpy()

    def fit(self, items: Sequence[SearchItem]) -> None:
        self._items = list(items)
        if not self._items:
            self._vectors = np.empty((0, 0), dtype=np.float32)
            return
        grouped: dict[str, list[tuple[int, SearchItem]]] = defaultdict(list)
        for index, item in enumerate(self._items):
            grouped[item.doc_id].append((index, item))
        vectors: list[np.ndarray | None] = [None] * len(self._items)
        for rows in grouped.values():
            document_text = str(rows[0][1].payload.get("document_text", ""))
            if not document_text:
                raise ValueError("LateChunkingRetriever requires document_text in chunk payload")
            encoded = self._document_vectors(document_text, [item for _, item in rows])
            for index, item in rows:
                vectors[index] = encoded[item.id]
        if any(vector is None for vector in vectors):
            raise RuntimeError("Late chunking did not encode every chunk")
        self._vectors = np.stack(vectors).astype(np.float32)  # type: ignore[arg-type]

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        if self._vectors is None:
            raise RuntimeError("Retriever must be fitted before search")
        scores = self._vectors @ self._query_vector(query)
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
