from __future__ import annotations

from dataclasses import dataclass, field

from vibiomir.schemas import Query, QueryVariant

from .base import QueryProcessor

_NLLB_CODES = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans"}


@dataclass(slots=True)
class NLLBQueryTranslator(QueryProcessor):
    model_name: str = "facebook/nllb-200-distilled-600M"
    targets: list[str] = field(default_factory=lambda: ["en", "zh"])
    include_original: bool = True
    weight: float = 0.8
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install `vibiomir-lab[transformers]` for NLLB translation") from exc
        source_code = _NLLB_CODES.get("vi", "vie_Latn")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=source_code)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def transform(self, query: Query) -> list[QueryVariant]:
        self._load()
        assert self._tokenizer is not None and self._model is not None
        variants: list[QueryVariant] = []
        if self.include_original:
            variants.append(QueryVariant(query.id, query.text, "original", query.language, 1.0))
        for target in self.targets:
            target_code = _NLLB_CODES[target]
            inputs = self._tokenizer(query.text, return_tensors="pt")
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(target_code),
                max_new_tokens=192,
            )
            text = self._tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            variants.append(QueryVariant(query.id, text, f"translate_{target}", target, self.weight))
        return variants

