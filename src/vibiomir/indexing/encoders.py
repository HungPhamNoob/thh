from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from vibiomir.gpu import apply_gpu_memory_limit, is_cuda_oom, record_cuda_oom
from vibiomir.models import repair_1d_position_id_buffers, resolve_hf_model

log = logging.getLogger(__name__)


class TextEncoder(ABC):
    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashTextEncoder(TextEncoder):
    """Dependency-free deterministic encoder for smoke tests, never for competition runs."""

    def __init__(self, dimension: int = 64, normalize: bool = True, **_kwargs):
        self.dimension = dimension
        self.normalize = normalize

    def _encode(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.casefold().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                output[row, index] += sign
        if self.normalize:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            output /= np.maximum(norms, 1e-12)
        return output

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class SentenceTransformerTextEncoder(TextEncoder):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        normalize: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        query_prompt_name: str | None = None,
        device: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        code_revision: str | None = None,
        max_length: int | None = None,
        use_fp16: bool = True,
        **_kwargs,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install `vibiomir-lab[dense]` for dense indexing") from exc
        apply_gpu_memory_limit()
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_kwargs={"code_revision": code_revision} if code_revision else None,
            config_kwargs={"code_revision": code_revision} if code_revision else None,
        )
        if max_length is not None:
            self.model.max_seq_length = int(max_length)
        target = str(self.model.device)
        if use_fp16 and target.startswith("cuda"):
            self.model.half()
        repair_1d_position_id_buffers(self.model)
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.query_prompt_name = query_prompt_name

    def _encode(self, texts: list[str], query: bool) -> np.ndarray:
        values = [(self.query_prefix if query else self.document_prefix) + text for text in texts]
        kwargs: dict[str, Any] = {}
        if query and self.query_prompt_name:
            kwargs["prompt_name"] = self.query_prompt_name
        batch_size = self.batch_size
        while True:
            try:
                encoded = self.model.encode(
                    values,
                    batch_size=batch_size,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                    **kwargs,
                )
                self.batch_size = batch_size
                return np.asarray(encoded, dtype=np.float32)
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                record_cuda_oom()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    log.warning("CUDA OOM in sentence encoder; retrying with batch_size=%d", batch_size)
                    continue
                if os.getenv("VIBIOMIR_GPU_CPU_FALLBACK", "1") == "1" and str(
                    self.model.device
                ).startswith("cuda"):
                    log.warning("CUDA OOM at batch_size=1; moving sentence encoder to CPU")
                    self.model.to("cpu")
                    continue
                raise RuntimeError("UNSUPPORTED_4GB_VRAM: sentence encoder OOM at batch_size=1") from exc

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, True)


class TransformersTextEncoder(TextEncoder):
    """HF encoder matching MedCPT, BMRetriever and Qwen3 official inference recipes."""

    def __init__(
        self,
        model_name: str,
        document_model_name: str | None = None,
        pooling: str = "mean",
        batch_size: int = 32,
        query_max_length: int = 512,
        document_max_length: int = 512,
        normalize: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        padding_side: str = "right",
        add_eos: bool = False,
        document_as_pair: bool = False,
        dimension: int | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        document_revision: str | None = None,
        device: str | None = None,
        use_fp16: bool = True,
        **_kwargs,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install `vibiomir-lab[transformers]` for HF indexing") from exc
        apply_gpu_memory_limit()
        self.torch = torch
        self.auto_model = AutoModel
        self.auto_tokenizer = AutoTokenizer
        self.model_name = model_name
        self.document_model_name = document_model_name or model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.padding_side = padding_side
        self.add_eos = add_eos
        self.document_as_pair = document_as_pair
        self.dimension = dimension
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.document_revision = document_revision or revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        self._query_tower: tuple[Any, Any] | None = None
        self._document_tower: tuple[Any, Any] | None = None

    def _load(self, name: str, revision: str | None) -> tuple[Any, Any]:
        tokenizer = self.auto_tokenizer.from_pretrained(
            name,
            trust_remote_code=self.trust_remote_code,
            padding_side=self.padding_side,
            revision=revision,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = self.auto_model.from_pretrained(
            name, trust_remote_code=self.trust_remote_code, revision=revision
        )
        if self.use_fp16 and str(self.device).startswith("cuda"):
            model.half()
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _tower(self, query: bool) -> tuple[Any, Any]:
        if query:
            if self._query_tower is None:
                self._query_tower = self._load(self.model_name, self.revision)
            return self._query_tower
        if self._document_tower is None:
            self._document_tower = self._load(self.document_model_name, self.document_revision)
        return self._document_tower

    def _tokenize(self, tokenizer, texts: list[str], max_length: int, pair: bool):
        kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": max_length - (1 if self.add_eos else 0),
            "return_tensors": "pt",
        }
        if pair:
            # MedCPT's article tower was trained on (title, abstract) pairs.  A
            # chunk has no independent title, so keep title empty and place the
            # complete, unmodified chunk in the article/abstract segment.
            encoded = tokenizer([""] * len(texts), texts, **kwargs)
        elif not self.add_eos:
            encoded = tokenizer(texts, **kwargs)
        else:
            unpadded = {
                key: value
                for key, value in kwargs.items()
                if key not in {"padding", "return_tensors"}
            }
            encoded = tokenizer(
                texts,
                padding=False,
                return_attention_mask=False,
                **unpadded,
            )
            eos = tokenizer.eos_token_id
            if eos is None:
                raise ValueError(f"Tokenizer for {self.model_name} has no eos_token_id")
            encoded["input_ids"] = [ids + [eos] for ids in encoded["input_ids"]]
            encoded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _encode(self, texts: list[str], query: bool) -> np.ndarray:
        tokenizer, model = self._tower(query)
        prefix = self.query_prefix if query else self.document_prefix
        max_length = self.query_max_length if query else self.document_max_length
        output: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            values = [prefix + text for text in texts[start : start + self.batch_size]]
            encoded = self._tokenize(
                tokenizer, values, max_length, self.document_as_pair and not query
            )
            with self.torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"]
            if self.pooling == "cls":
                embeddings = hidden[:, 0]
            elif self.pooling == "mean":
                embeddings = (hidden * mask[..., None]).sum(1) / mask.sum(
                    1, keepdim=True
                ).clamp_min(1)
            elif self.pooling == "last_token":
                if bool((mask[:, -1].sum() == mask.shape[0]).item()):
                    embeddings = hidden[:, -1]
                else:
                    indices = mask.sum(dim=1) - 1
                    embeddings = hidden[
                        self.torch.arange(hidden.shape[0], device=hidden.device), indices
                    ]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
            if self.dimension is not None:
                embeddings = embeddings[:, : self.dimension]
            if self.normalize:
                embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
            output.append(embeddings.float().cpu().numpy())
        return np.concatenate(output).astype(np.float32) if output else np.empty((0, 0), np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, True)


class BGEM3DenseTextEncoder(TextEncoder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 16,
        max_length: int = 1024,
        use_fp16: bool = True,
        revision: str | None = None,
        **_kwargs,
    ):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError("Install `vibiomir-lab[bge]` for BGE-M3 indexing") from exc
        apply_gpu_memory_limit()
        self.model = BGEM3FlagModel(resolve_hf_model(model_name, revision), use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )["dense_vecs"],
            dtype=np.float32,
        )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


def build_text_encoder(config: dict[str, Any]) -> TextEncoder:
    kind = config.get("type", "sentence_transformer")
    params = {key: value for key, value in config.items() if key != "type"}
    if kind == "hash":
        return HashTextEncoder(**params)
    if kind == "sentence_transformer":
        return SentenceTransformerTextEncoder(**params)
    if kind == "transformers":
        return TransformersTextEncoder(**params)
    if kind == "bge_m3_dense":
        return BGEM3DenseTextEncoder(**params)
    raise ValueError(f"Unknown encoder type: {kind}")
