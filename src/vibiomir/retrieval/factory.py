from __future__ import annotations

from typing import Any

from .base import Retriever
from .bge_m3 import BGEM3Retriever
from .bm25 import BM25Retriever
from .colbert import ColBERTv2Retriever
from .dense import AsymmetricSentenceTransformerRetriever, SentenceTransformerRetriever
from .impact import LuceneImpactRetriever
from .late_chunking import LateChunkingRetriever
from .numpy_index import NumpyIndexRetriever
from .pyserini import PyseriniRetriever
from .qdrant_index import QdrantIndexRetriever
from .splade import SpladeRetriever
from .transformers_dense import AsymmetricTransformersRetriever, TransformersEncoderRetriever


def build_retriever(config: dict[str, Any]) -> Retriever:
    name = config.get("type", "bm25")
    controls = {
        "type",
        "weight",
        "k",
        "languages",
        "variant_ids",
        "cascade_overfetch_factor",
        "indexing",
        "model_revision",
        "source_revision",
        "factory",
        "state_dir",
        "distance",
        "on_disk",
        "store_text_payload",
        "recreate",
    }
    params = {k: v for k, v in config.items() if k not in controls}
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
    if name == "qdrant":
        return QdrantIndexRetriever(**params)
    if name == "colbertv2":
        return ColBERTv2Retriever(**params)
    if name == "splade":
        return SpladeRetriever(**params)
    if name == "late_chunking":
        return LateChunkingRetriever(**params)
    if name == "lucene_impact":
        return LuceneImpactRetriever(**params)
    if name == "numpy_index":
        return NumpyIndexRetriever(**params)
    raise ValueError(f"Unknown retriever type: {name}")
