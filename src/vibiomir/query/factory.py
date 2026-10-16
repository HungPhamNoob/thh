from __future__ import annotations

from typing import Any

from .base import QueryProcessor
from .identity import IdentityQueryProcessor
from .multi import BiomedicalSynonymExpander, CompositeQueryProcessor, LocalLLMQueryExpander
from .translation import NLLBQueryTranslator


def build_query_processor(config: dict[str, Any]) -> QueryProcessor:
    name = config.get("type", "identity")
    params = {k: v for k, v in config.items() if k != "type"}
    if name == "identity":
        return IdentityQueryProcessor()
    if name == "biomedical_synonyms":
        return BiomedicalSynonymExpander(**params)
    if name == "nllb":
        return NLLBQueryTranslator(**params)
    if name == "local_llm":
        return LocalLLMQueryExpander(**params)
    if name == "composite":
        processors = [build_query_processor(item) for item in params.pop("processors")]
        return CompositeQueryProcessor(processors=processors, **params)
    raise ValueError(f"Unknown query processor type: {name}")

