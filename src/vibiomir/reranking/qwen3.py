from __future__ import annotations

from dataclasses import dataclass, field

from vibiomir.schemas import SearchItem, SearchResult

from .base import Reranker


@dataclass(slots=True)
class Qwen3Reranker(Reranker):
    """Qwen3 yes/no causal-LM reranker, matching the authors' released recipe."""

    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    instruction: str = (
        "Given a Vietnamese medical question, retrieve relevant biomedical passages "
        "in Vietnamese, English, or Chinese that answer the question"
    )
    batch_size: int = 8
    max_length: int = 2048
    device: str | None = None
    blend_retrieval: float = 0.0
    revision: str | None = None
    _model: object | None = field(default=None, init=False, repr=False)
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _prefix_tokens: list[int] = field(default_factory=list, init=False, repr=False)
    _suffix_tokens: list[int] = field(default_factory=list, init=False, repr=False)
    _true_id: int = field(default=-1, init=False, repr=False)
    _false_id: int = field(default=-1, init=False, repr=False)

    def _load(self) -> tuple[object, object]:
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Install `vibiomir-lab[transformers]` for Qwen3 reranking"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left", revision=self.revision
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, revision=self.revision
            )
            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(target)
            self._model.eval()
            prefix = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on "
                'the Query and the Instruct provided. Note that the answer can only be "yes" or '
                '"no".<|im_end|>\n<|im_start|>user\n'
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
            self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)
            self._true_id = self._tokenizer.convert_tokens_to_ids("yes")
            self._false_id = self._tokenizer.convert_tokens_to_ids("no")
        return self._tokenizer, self._model

    def _score(self, query: str, documents: list[str]) -> list[float]:
        import torch

        tokenizer, model = self._load()
        target = next(model.parameters()).device
        output: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            pairs = [
                f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
                for document in documents[start : start + self.batch_size]
            ]
            encoded = tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens),
            )
            for index, ids in enumerate(encoded["input_ids"]):
                encoded["input_ids"][index] = self._prefix_tokens + ids + self._suffix_tokens
            encoded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            encoded = {key: value.to(target) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits[:, -1, :].float()
                yes_no = torch.stack([logits[:, self._false_id], logits[:, self._true_id]], dim=1)
                scores = torch.softmax(yes_no, dim=1)[:, 1]
            output.extend(float(value) for value in scores.cpu())
        return output

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        items: dict[str, SearchItem],
        k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        scores = self._score(query, [items[row.item_id].text for row in candidates])
        retrieval = [row.score for row in candidates]
        low, high = min(retrieval), max(retrieval)
        output: list[SearchResult] = []
        for row, score in zip(candidates, scores):
            normalized = 0.0 if high == low else (row.score - low) / (high - low)
            output.append(
                SearchResult(
                    item_id=row.item_id,
                    score=score + self.blend_retrieval * normalized,
                    source="qwen3_reranker",
                    payload={**row.payload, "first_stage_score": row.score},
                )
            )
        output.sort(key=lambda row: (-row.score, row.item_id))
        for rank, row in enumerate(output[:k], 1):
            row.rank = rank
        return output[:k]
