from __future__ import annotations

from dataclasses import dataclass

from vibiomir.schemas import Chunk, Document
from vibiomir.text import whitespace_token_spans

from .base import Chunker, make_chunk


@dataclass(slots=True)
class FixedTokenChunker(Chunker):
    size: int = 180
    overlap: int = 40
    min_tokens: int = 12

    def __post_init__(self) -> None:
        if self.size <= 0 or self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("Require size > overlap >= 0")

    def split(self, document: Document) -> list[Chunk]:
        spans = whitespace_token_spans(document.text)
        if not spans:
            return []
        chunks: list[Chunk] = []
        step = self.size - self.overlap
        for token_start in range(0, len(spans), step):
            token_end = min(token_start + self.size, len(spans))
            if token_end - token_start < self.min_tokens and chunks:
                break
            start = spans[token_start][0]
            end = spans[token_end - 1][1]
            chunks.append(make_chunk(document, len(chunks), start, end))
            if token_end == len(spans):
                break
        return chunks

