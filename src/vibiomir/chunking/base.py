from __future__ import annotations

from abc import ABC, abstractmethod

from vibiomir.schemas import Chunk, Document


class Chunker(ABC):
    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        raise NotImplementedError


def make_chunk(document: Document, order: int, start: int, end: int) -> Chunk:
    text = document.text[start:end]
    return Chunk(
        id=f"{document.id}::c{order:05d}",
        doc_id=document.id,
        text=text,
        index_text=text,
        order=order,
        start_char=start,
        end_char=end,
        language=document.language,
        metadata={"url": document.url, "title": document.title},
    )

