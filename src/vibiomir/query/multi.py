from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from vibiomir.schemas import Query, QueryVariant

from .base import QueryProcessor


@dataclass(slots=True)
class CompositeQueryProcessor(QueryProcessor):
    processors: list[QueryProcessor]
    deduplicate: bool = True

    def transform(self, query: Query) -> list[QueryVariant]:
        variants: list[QueryVariant] = []
        for processor in self.processors:
            variants.extend(processor.transform(query))
        if not self.deduplicate:
            return variants
        seen: set[str] = set()
        output: list[QueryVariant] = []
        for variant in variants:
            key = variant.text.casefold().strip()
            if key not in seen:
                seen.add(key)
                output.append(variant)
        return output


@dataclass(slots=True)
class BiomedicalSynonymExpander(QueryProcessor):
    """Dictionary expansion inspired by SapBERT/BioLORD entity-level supervision."""

    synonym_path: str
    include_original: bool = True
    max_synonyms_per_term: int = 3
    weight: float = 0.75
    _synonyms: dict[str, list[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw = json.loads(Path(self.synonym_path).read_text(encoding="utf-8"))
        self._synonyms = {str(key).casefold(): list(value) for key, value in raw.items()}

    def transform(self, query: Query) -> list[QueryVariant]:
        variants: list[QueryVariant] = []
        if self.include_original:
            variants.append(QueryVariant(query.id, query.text, "original", query.language, 1.0))
        additions: list[str] = []
        folded = query.text.casefold()
        for term, synonyms in self._synonyms.items():
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded):
                additions.extend(synonyms[: self.max_synonyms_per_term])
        if additions:
            text = f"{query.text} {' '.join(dict.fromkeys(additions))}"
            variants.append(
                QueryVariant(query.id, text, "biomedical_synonyms", query.language, self.weight)
            )
        return variants


@dataclass(slots=True)
class LocalLLMQueryExpander(QueryProcessor):
    """Generate exploratory search views with an open local causal LM.

    This generic prompt set is not a reproduction of Query2doc, GRF, or BRIGHT.
    """

    model_name: str
    modes: list[str] = field(default_factory=lambda: ["keywords", "entities", "reasoning"])
    max_new_tokens: int = 160
    weight: float = 0.65
    include_original: bool = True
    device: str = "auto"
    _generator: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> object:
        if self._generator is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "Install `vibiomir-lab[transformers]` for local LLM expansion"
                ) from exc
            self._generator = pipeline(
                "text-generation", model=self.model_name, device_map=self.device
            )
        return self._generator

    def transform(self, query: Query) -> list[QueryVariant]:
        variants: list[QueryVariant] = []
        if self.include_original:
            variants.append(QueryVariant(query.id, query.text, "original", query.language, 1.0))
        generator = self._load()
        prompts = {
            "keywords": "Extract precise multilingual biomedical search terms for this query. Return terms only:\n",
            "entities": "List diseases, symptoms, drugs, anatomy and aliases implied by this query. Return terms only:\n",
            "reasoning": "Reason briefly about what evidence would answer this query, then write a compact search query:\n",
            "pseudo_document": "Write a short factual passage that a relevant medical web page might contain:\n",
        }
        for mode in self.modes:
            if mode not in prompts:
                raise ValueError(f"Unknown expansion mode: {mode}")
            prompt = prompts[mode] + query.text
            generated = generator(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)[0][
                "generated_text"
            ]
            expansion = (
                generated[len(prompt) :].strip()
                if generated.startswith(prompt)
                else generated.strip()
            )
            variants.append(
                QueryVariant(
                    query.id, f"{query.text}\n{expansion}", mode, query.language, self.weight
                )
            )
        return variants
