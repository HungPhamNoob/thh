from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vibiomir.indexing.encoders import build_text_encoder
from vibiomir.indexing.qdrant import make_qdrant_client, resolve_collection_name
from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class QdrantIndexRetriever(Retriever):
    collection_name: str
    encoder: dict[str, Any]
    name: str = "qdrant"
    url_env: str = "QDRANT_URL"
    api_key_env: str = "QDRANT_API_KEY"
    prefix_env: str = "QDRANT_COLLECTION_PREFIX"
    local_path: str | Path | None = None
    timeout: float = 60.0
    prefer_grpc: bool = False
    overfetch_factor: int = 10
    _client: object | None = field(default=None, init=False, repr=False)
    _encoder: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._client is not None:
            return
        self._client = make_qdrant_client(
            url_env=self.url_env,
            api_key_env=self.api_key_env,
            local_path=self.local_path,
            timeout=self.timeout,
            prefer_grpc=self.prefer_grpc,
        )
        self._encoder = build_text_encoder(self.encoder)

    def fit(self, items: Sequence[SearchItem]) -> None:
        # Full-corpus vectors are built separately by `vibiomir index build-qdrant`.
        self._load()

    def search(
        self, query: str, k: int, allowed_ids: set[str] | None = None
    ) -> list[SearchResult]:
        self._load()
        assert self._client is not None and self._encoder is not None
        from qdrant_client import models

        query_filter = None
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="item_id", match=models.MatchAny(any=sorted(allowed_ids))
                    )
                ]
            )
        fetch = k if allowed_ids is None else max(k, k * self.overfetch_factor)
        vector = np.asarray(self._encoder.encode_queries([query])[0], dtype=np.float32)
        response = self._client.query_points(
            collection_name=resolve_collection_name(self.collection_name, self.prefix_env),
            query=vector.tolist(),
            query_filter=query_filter,
            limit=fetch,
            with_payload=True,
        )
        output: list[SearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            item_id = str(payload.get("item_id", ""))
            if not item_id or (allowed_ids is not None and item_id not in allowed_ids):
                continue
            output.append(
                SearchResult(
                    item_id=item_id,
                    score=float(point.score),
                    rank=len(output) + 1,
                    source=self.name,
                    payload=payload,
                )
            )
            if len(output) >= k:
                break
        return output

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
