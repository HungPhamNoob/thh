from __future__ import annotations

from dataclasses import dataclass

from vibiomir.schemas import Chunk, Document

from .base import Chunker


@dataclass(slots=True)
class ContextualWindowChunker(Chunker):
    """Add document context to embeddings while retaining the exact source chunk for submission."""

    base: Chunker
    surrounding_chars: int = 500
    include_title: bool = True

    def split(self, document: Document) -> list[Chunk]:
        chunks = self.base.split(document)
        for chunk in chunks:
            left = max(0, chunk.start_char - self.surrounding_chars)
            right = min(len(document.text), chunk.end_char + self.surrounding_chars)
            context = document.text[left:right]
            title = f"Title: {document.title}\n" if self.include_title and document.title else ""
            chunk.index_text = f"{title}Document context: {context}\nTarget passage: {chunk.text}"
            chunk.metadata["context_span"] = [left, right]
        return chunks

