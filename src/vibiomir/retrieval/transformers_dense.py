from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class TransformersEncoderRetriever(Retriever):
    """Generic HF encoder for checkpoints without SentenceTransformers metadata."""

    model_name: str
    name: str = "transformers_dense"
    pooling: str = "mean"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    query_max_length: int | None = None
    document_max_length: int | None = None
    padding_side: str = "right"
    add_eos: bool = False
    document_as_pair: bool = False
    dimension: int | None = None
    trust_remote_code: bool = False
    device: str | None = None
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _items: list[SearchItem] = field(default_factory=list, init=False, repr=False)
    _embeddings: np.ndarray | None = field(default=None, init=False, repr=False)

    def _load_tower(self, model_name: str) -> tuple[object, object]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install `vibiomir-lab[transformers]` for HF encoder retrieval"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side=self.padding_side,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_name, trust_remote_code=self.trust_remote_code)
        target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target)
        model.eval()
        return tokenizer, model

    def _query_tower(self) -> tuple[object, object]:
        if self._model is None or self._tokenizer is None:
            self._tokenizer, self._model = self._load_tower(self.model_name)
        return self._tokenizer, self._model

    def _document_tower(self) -> tuple[object, object]:
        return self._query_tower()

    def _encode(
        self,
        texts: list[str],
        tower: tuple[object, object],
        *,
        max_length: int,
        as_pair: bool = False,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        tokenizer, model = tower
        device = next(model.parameters()).device
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            values = texts[start : start + self.batch_size]
            tokenize_kwargs = {
                "padding": True,
                "truncation": True,
                "max_length": max_length - (1 if self.add_eos else 0),
                "return_tensors": "pt",
            }
            if as_pair:
                encoded = tokenizer(
                    [""] * len(values),
                    values,
                    **tokenize_kwargs,
                )
            elif not self.add_eos:
                encoded = tokenizer(values, **tokenize_kwargs)
            else:
                tokenize_kwargs.pop("padding")
                tokenize_kwargs.pop("return_tensors")
                tokenize_kwargs["return_attention_mask"] = False
                encoded = tokenizer(values, **tokenize_kwargs)
                if tokenizer.eos_token_id is None:
                    raise ValueError(f"Tokenizer for {self.model_name} has no eos_token_id")
                encoded["input_ids"] = [
                    ids + [tokenizer.eos_token_id] for ids in encoded["input_ids"]
                ]
                encoded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"]
            if self.pooling == "cls":
                embeddings = hidden[:, 0]
            elif self.pooling == "mean":
                embeddings = (hidden * mask[..., None]).sum(1) / mask.sum(
                    1, keepdim=True
                ).clamp_min(1)
            elif self.pooling == "last_token":
                if bool((mask[:, -1].sum() == mask.shape[0]).item()):
                    embeddings = hidden[:, -1]
                else:
                    indices = mask.sum(dim=1) - 1
                    embeddings = hidden[torch.arange(hidden.shape[0], device=device), indices]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
            if self.dimension is not None:
                embeddings = embeddings[:, : self.dimension]
            if self.normalize_embeddings:
                embeddings = functional.normalize(embeddings, p=2, dim=1)
            batches.append(embeddings.float().cpu().numpy())
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(batches, axis=0).astype(np.float32)

    def fit(self, items: Sequence[SearchItem]) -> None:
        self._items = list(items)
        texts = [self.document_prefix + item.text for item in self._items]
        self._embeddings = self._encode(
            texts,
            self._document_tower(),
            max_length=self.document_max_length or self.max_length,
            as_pair=self.document_as_pair,
        )

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        if self._embeddings is None:
            raise RuntimeError("Retriever must be fitted before search")
        query_vector = self._encode(
            [self.query_prefix + query],
            self._query_tower(),
            max_length=self.query_max_length or self.max_length,
        )[0]
        scores = self._embeddings @ query_vector
        candidates = (
            (float(score), index)
            for index, score in enumerate(scores)
            if allowed_ids is None or self._items[index].id in allowed_ids
        )
        best = heapq.nlargest(k, candidates)
        return [
            SearchResult(
                item_id=self._items[index].id,
                score=score,
                rank=rank,
                source=self.name,
                payload=self._items[index].payload,
            )
            for rank, (score, index) in enumerate(best, 1)
        ]


@dataclass(slots=True)
class AsymmetricTransformersRetriever(TransformersEncoderRetriever):
    document_model_name: str = ""
    _document_tokenizer: object | None = field(default=None, init=False, repr=False)
    _document_model: object | None = field(default=None, init=False, repr=False)

    def _document_tower(self) -> tuple[object, object]:
        if self._document_model is None or self._document_tokenizer is None:
            self._document_tokenizer, self._document_model = self._load_tower(
                self.document_model_name
            )
        return self._document_tokenizer, self._document_model
