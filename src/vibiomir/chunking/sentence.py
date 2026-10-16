from __future__ import annotations

import re
from dataclasses import dataclass

from vibiomir.schemas import Chunk, Document

from .base import Chunker, make_chunk

_SENTENCE_RE = re.compile(r".*?(?:[.!?。！？]+(?=\s|$)|\n+|$)", re.DOTALL)


def sentence_spans(text: str) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group(0)
        if not raw.strip():
            continue
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start, end = match.start() + left, match.start() + right
        spans.append((start, end, len(text[start:end].split())))
    return spans


@dataclass(slots=True)
class SentenceChunker(Chunker):
    max_tokens: int = 180
    overlap_sentences: int = 1
    min_tokens: int = 12

    def split(self, document: Document) -> list[Chunk]:
        sentences = sentence_spans(document.text)
        if not sentences:
            return []
        groups: list[tuple[int, int]] = []
        start_idx = 0
        while start_idx < len(sentences):
            end_idx, count = start_idx, 0
            while end_idx < len(sentences):
                next_count = count + sentences[end_idx][2]
                if end_idx > start_idx and next_count > self.max_tokens:
                    break
                count = next_count
                end_idx += 1
            if count >= self.min_tokens or not groups:
                groups.append((sentences[start_idx][0], sentences[end_idx - 1][1]))
            if end_idx >= len(sentences):
                break
            start_idx = max(start_idx + 1, end_idx - self.overlap_sentences)
        return [make_chunk(document, i, start, end) for i, (start, end) in enumerate(groups)]

