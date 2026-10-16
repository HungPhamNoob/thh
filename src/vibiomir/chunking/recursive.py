from __future__ import annotations

from dataclasses import dataclass, field

from vibiomir.schemas import Chunk, Document
from vibiomir.text import whitespace_token_spans

from .base import Chunker, make_chunk


@dataclass(slots=True)
class RecursiveChunker(Chunker):
    """Split at semantic boundaries, then fall back to fixed token windows."""

    max_tokens: int = 220
    overlap_tokens: int = 30
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", "; ", ", "])

    def _split_span(self, text: str, start: int, end: int, level: int = 0) -> list[tuple[int, int]]:
        segment = text[start:end]
        if len(segment.split()) <= self.max_tokens:
            return [(start, end)]
        if level < len(self.separators):
            separator = self.separators[level]
            pieces: list[tuple[int, int]] = []
            cursor = start
            while True:
                found = text.find(separator, cursor, end)
                if found < 0:
                    pieces.append((cursor, end))
                    break
                boundary = found + len(separator)
                pieces.append((cursor, boundary))
                cursor = boundary
            if len(pieces) > 1:
                result: list[tuple[int, int]] = []
                for left, right in pieces:
                    if text[left:right].strip():
                        result.extend(self._split_span(text, left, right, level + 1))
                return result
            return self._split_span(text, start, end, level + 1)

        tokens = whitespace_token_spans(segment)
        step = self.max_tokens - self.overlap_tokens
        output: list[tuple[int, int]] = []
        for offset in range(0, len(tokens), step):
            stop = min(offset + self.max_tokens, len(tokens))
            output.append((start + tokens[offset][0], start + tokens[stop - 1][1]))
            if stop == len(tokens):
                break
        return output

    def split(self, document: Document) -> list[Chunk]:
        leaves = self._split_span(document.text, 0, len(document.text))
        packed: list[tuple[int, int]] = []
        for start, end in leaves:
            start += len(document.text[start:end]) - len(document.text[start:end].lstrip())
            end -= len(document.text[start:end]) - len(document.text[start:end].rstrip())
            if start >= end:
                continue
            if packed and len(document.text[packed[-1][0] : end].split()) <= self.max_tokens:
                packed[-1] = (packed[-1][0], end)
            else:
                packed.append((start, end))
        return [make_chunk(document, i, start, end) for i, (start, end) in enumerate(packed)]

