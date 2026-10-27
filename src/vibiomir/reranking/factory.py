from __future__ import annotations

from typing import Any

from .base import Reranker
from .bge_m3 import BGEM3MultiVectorReranker
from .cross_encoder import CrossEncoderReranker
from .hf_sequence import HFSequenceClassificationReranker
from .none import IdentityReranker
from .qwen3 import Qwen3Reranker


def build_reranker(config: dict[str, Any]) -> Reranker:
    name = config.get("type", "none")
    controls = {"type", "candidate_k", "output_k", "query_variant_id"}
    params = {k: v for k, v in config.items() if k not in controls}
    if name == "none":
        return IdentityReranker()
    if name == "cross_encoder":
        return CrossEncoderReranker(**params)
    if name == "hf_sequence_classifier":
        return HFSequenceClassificationReranker(**params)
    if name == "qwen3":
        return Qwen3Reranker(**params)
    if name == "bge_m3_multivector":
        return BGEM3MultiVectorReranker(**params)
    raise ValueError(f"Unknown reranker type: {name}")
