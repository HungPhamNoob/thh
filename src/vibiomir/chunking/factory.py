from __future__ import annotations

from typing import Any

from .base import Chunker
from .contextual import ContextualWindowChunker
from .fixed import FixedTokenChunker
from .recursive import RecursiveChunker
from .sentence import SentenceChunker


def build_chunker(config: dict[str, Any]) -> Chunker:
    name = config.get("type", "fixed")
    params = {k: v for k, v in config.items() if k != "type"}
    if name == "fixed":
        return FixedTokenChunker(**params)
    if name == "sentence":
        return SentenceChunker(**params)
    if name == "recursive":
        return RecursiveChunker(**params)
    if name == "contextual":
        base_config = params.pop("base", {"type": "sentence"})
        return ContextualWindowChunker(base=build_chunker(base_config), **params)
    raise ValueError(f"Unknown chunker type: {name}")

