from __future__ import annotations

from typing import Any

from .base import Reranker
from .cross_encoder import CrossEncoderReranker
from .none import IdentityReranker


def build_reranker(config: dict[str, Any]) -> Reranker:
    name = config.get("type", "none")
    params = {k: v for k, v in config.items() if k not in {"type", "candidate_k", "output_k"}}
    if name == "none":
        return IdentityReranker()
    if name == "cross_encoder":
        return CrossEncoderReranker(**params)
    raise ValueError(f"Unknown reranker type: {name}")

