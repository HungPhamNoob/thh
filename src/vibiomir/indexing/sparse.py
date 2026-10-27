from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vibiomir.models import resolve_hf_model

from .store import ArtifactStore


class SparseTextEncoder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[dict[str, int]]:
        raise NotImplementedError


class SpladeSparseEncoder(SparseTextEncoder):
    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        batch_size: int = 8,
        max_length: int = 256,
        device: str | None = None,
        min_weight: float = 0.0,
        quantization_factor: int = 100,
        revision: str | None = None,
        **_kwargs,
    ):
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install `vibiomir-lab[transformers]` for SPLADE") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(target)
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_weight = min_weight
        self.quantization_factor = quantization_factor

    def encode(self, texts: list[str]) -> list[dict[str, int]]:
        output: list[dict[str, int]] = []
        target = next(self.model.parameters()).device
        for start in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target) for key, value in encoded.items()}
            with self.torch.no_grad():
                logits = self.model(**encoded).logits
                weights = self.torch.log1p(self.torch.relu(logits))
                weights *= encoded["attention_mask"].unsqueeze(-1)
                vectors = weights.max(dim=1).values
            for vector in vectors.cpu():
                indices = (vector > self.min_weight).nonzero(as_tuple=False).flatten().tolist()
                tokens = self.tokenizer.convert_ids_to_tokens(indices)
                sparse = {
                    token: int(round(float(vector[index]) * self.quantization_factor))
                    for token, index in zip(tokens, indices)
                    if token not in self.tokenizer.all_special_tokens
                    and round(float(vector[index]) * self.quantization_factor) > 0
                }
                if not sparse:
                    # Same non-empty guard as naver/splade's Anserini exporter.
                    sparse[self.tokenizer.convert_ids_to_tokens(998)] = 1
                output.append(sparse)
        return output


class BGEM3SparseEncoder(SparseTextEncoder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 16,
        max_length: int = 1024,
        use_fp16: bool = True,
        quantization_factor: int = 100,
        revision: str | None = None,
        **_kwargs,
    ):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError("Install `vibiomir-lab[bge]` for BGE-M3 sparse indexing") from exc
        self.model = BGEM3FlagModel(resolve_hf_model(model_name, revision), use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length
        self.quantization_factor = quantization_factor

    def encode(self, texts: list[str]) -> list[dict[str, int]]:
        encoded = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        converted = self.model.convert_id_to_token(encoded["lexical_weights"])
        return [
            {
                str(token): int(round(float(weight) * self.quantization_factor))
                for token, weight in vector.items()
                if round(float(weight) * self.quantization_factor) > 0
            }
            for vector in converted
        ]


def build_sparse_encoder(config: dict[str, Any]) -> SparseTextEncoder:
    kind = config.get("type")
    params = {key: value for key, value in config.items() if key != "type"}
    if kind == "splade":
        return SpladeSparseEncoder(**params)
    if kind == "bge_m3_sparse":
        return BGEM3SparseEncoder(**params)
    raise ValueError(f"Unknown sparse encoder type: {kind}")


def export_impact_vectors(
    store_path: str | Path,
    stage: str,
    encoder_config: dict[str, Any],
    output_path: str | Path,
    batch_size: int = 128,
    languages: Iterable[str] | None = None,
) -> int:
    """Export quantized sparse vectors, resuming from a sidecar state file."""
    encoder = build_sparse_encoder(encoder_config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    language_list = list(languages or [])
    state_path = output_path.with_suffix(output_path.suffix + ".state.json")
    expected = {
        "stage": stage,
        "encoder": encoder_config,
        "languages": language_list,
    }
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state:
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Cannot resume impact export: {key} changed")
        count = int(state["offset"])
        _truncate_jsonl(output_path, count)
        mode = "a"
    else:
        count = 0
        mode = "w"
    seen = 0
    with ArtifactStore(store_path) as store, output_path.open(mode, encoding="utf-8") as handle:
        for items in store.iter_items(stage, batch_size=batch_size, languages=language_list):
            batch_end = seen + len(items)
            if batch_end <= count:
                seen = batch_end
                continue
            items = items[max(0, count - seen) :]
            vectors = encoder.encode([item.text for item in items])
            for item, vector in zip(items, vectors):
                handle.write(
                    json.dumps({"id": item.id, "vector": vector}, ensure_ascii=False) + "\n"
                )
                count += 1
            handle.flush()
            _write_state(state_path, {**expected, "offset": count})
            seen = batch_end
    state_path.unlink(missing_ok=True)
    return count


def _truncate_jsonl(path: Path, lines: int) -> None:
    temporary = path.with_suffix(path.suffix + ".resume")
    with (
        path.open("r", encoding="utf-8") as source,
        temporary.open("w", encoding="utf-8") as target,
    ):
        for index, line in enumerate(source):
            if index >= lines:
                break
            target.write(line)
    temporary.replace(path)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
