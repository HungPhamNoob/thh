from __future__ import annotations

from typing import Any

from .base import Retriever
from .bge_m3 import BGEM3Retriever
from .bm25 import BM25Retriever
from .dense import AsymmetricSentenceTransformerRetriever, SentenceTransformerRetriever
from .pyserini import PyseriniRetriever
from .transformers_dense import AsymmetricTransformersRetriever, TransformersEncoderRetriever


def build_retriever(config: dict[str, Any]) -> Retriever:
    name = config.get("type", "bm25")
    params = {k: v for k, v in config.items() if k not in {"type", "weight", "k"}}
    if name == "bm25":
        return BM25Retriever(**params)
    if name == "dense":
        return SentenceTransformerRetriever(**params)
    if name == "asymmetric_dense":
        return AsymmetricSentenceTransformerRetriever(**params)
    if name == "transformers_dense":
        return TransformersEncoderRetriever(**params)
    if name == "asymmetric_transformers":
        return AsymmetricTransformersRetriever(**params)
    if name == "bge_m3":
        return BGEM3Retriever(**params)
    if name == "pyserini":
        return PyseriniRetriever(**params)
    raise ValueError(f"Unknown retriever type: {name}")

