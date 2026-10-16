from __future__ import annotations

from abc import ABC, abstractmethod

from vibiomir.schemas import Query, QueryVariant


class QueryProcessor(ABC):
    @abstractmethod
    def transform(self, query: Query) -> list[QueryVariant]:
        raise NotImplementedError

