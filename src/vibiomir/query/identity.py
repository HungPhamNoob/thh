from __future__ import annotations

from vibiomir.schemas import Query, QueryVariant

from .base import QueryProcessor


class IdentityQueryProcessor(QueryProcessor):
    def transform(self, query: Query) -> list[QueryVariant]:
        return [
            QueryVariant(
                query_id=query.id,
                text=query.text,
                language=query.language,
                variant_id="original",
                weight=1.0,
            )
        ]

