from __future__ import annotations

from collections import defaultdict
from typing import Any

from vibiomir.aggregation import DocumentAggregator
from vibiomir.chunking import build_chunker
from vibiomir.fusion import build_fusion
from vibiomir.query import build_query_processor
from vibiomir.reranking import build_reranker
from vibiomir.retrieval import build_retriever
from vibiomir.schemas import (
    Chunk,
    Document,
    Prediction,
    Query,
    QueryVariant,
    SearchItem,
    SearchResult,
)
from vibiomir.selection import build_selector


class SearchPipeline:
    """Configurable document/chunk cascade with independent components."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.chunker = build_chunker(config["chunking"])
        self.query_processor = build_query_processor(config.get("query", {"type": "identity"}))
        self.doc_configs = config.get("document_retrievers", [])
        self.chunk_configs = config.get("chunk_retrievers", [])
        self.doc_retrievers = [build_retriever(item) for item in self.doc_configs]
        self.chunk_retrievers = [build_retriever(item) for item in self.chunk_configs]
        self.fusion = build_fusion(config.get("fusion", {"type": "rrf"}))
        self.reranker = build_reranker(config.get("reranker", {"type": "none"}))
        self.aggregator = DocumentAggregator(**config.get("document_aggregation", {}))
        self.doc_selector = build_selector(config.get("document_selection", {"max_k": 10}))
        self.chunk_selector = build_selector(config.get("chunk_selection", {"max_k": 10}))
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.doc_items: dict[str, SearchItem] = {}
        self.chunk_items: dict[str, SearchItem] = {}

    def fit(self, documents: list[Document]) -> None:
        self.documents = {document.id: document for document in documents}
        chunk_list = [chunk for document in documents for chunk in self.chunker.split(document)]
        self.chunks = {chunk.id: chunk for chunk in chunk_list}
        self.doc_items = {
            document.id: SearchItem(
                id=document.id,
                text=f"{document.title}\n{document.text}" if document.title else document.text,
                doc_id=document.id,
                kind="document",
                payload={"doc_id": document.id, "language": document.language},
            )
            for document in documents
        }
        self.chunk_items = {
            chunk.id: SearchItem(
                id=chunk.id,
                text=chunk.index_text,
                doc_id=chunk.doc_id,
                kind="chunk",
                payload={
                    "doc_id": chunk.doc_id,
                    "order": chunk.order,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "language": chunk.language,
                    "document_text": self.documents[chunk.doc_id].text,
                },
            )
            for chunk in chunk_list
        }
        for retriever in self.doc_retrievers:
            retriever.fit(list(self.doc_items.values()))
        for retriever in self.chunk_retrievers:
            retriever.fit(list(self.chunk_items.values()))

    def _retrieve(
        self,
        variants: list[QueryVariant],
        retrievers: list,
        retriever_configs: list[dict[str, Any]],
        items: dict[str, SearchItem],
        default_k: int,
        allowed_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        runs: dict[str, list[SearchResult]] = {}
        weights: dict[str, float] = {}
        for retriever_index, (retriever, retriever_config) in enumerate(
            zip(retrievers, retriever_configs)
        ):
            enabled_variants = set(retriever_config.get("variant_ids", []))
            enabled_languages = set(retriever_config.get("languages", []))
            fetch_k = int(retriever_config.get("k", default_k))
            base_weight = float(retriever_config.get("weight", 1.0))
            for variant in variants:
                if enabled_variants and variant.variant_id not in enabled_variants:
                    continue
                source = f"r{retriever_index}:{retriever.name}:{variant.variant_id}"
                results = retriever.search(variant.text, fetch_k, allowed_ids=allowed_ids)
                filtered: list[SearchResult] = []
                for result in results:
                    item = items.get(result.item_id)
                    if item is None:
                        continue
                    if enabled_languages and item.payload.get("language") not in enabled_languages:
                        continue
                    result.payload = {**item.payload, **result.payload}
                    result.source = source
                    filtered.append(result)
                for rank, result in enumerate(filtered, 1):
                    result.rank = rank
                runs[source] = filtered
                weights[source] = base_weight * variant.weight
        return self.fusion.fuse(runs, weights=weights, k=default_k) if runs else []

    def _select_chunks(
        self, chunk_results: list[SearchResult], selected_doc_ids: set[str]
    ) -> list[SearchResult]:
        config = self.config.get("chunk_selection", {})
        filtered = [row for row in chunk_results if str(row.payload["doc_id"]) in selected_doc_ids]
        if not config.get("per_document", True):
            return self.chunk_selector.select(filtered, self.chunk_items)
        grouped: dict[str, list[SearchResult]] = defaultdict(list)
        for row in filtered:
            grouped[str(row.payload["doc_id"])].append(row)
        selected: list[SearchResult] = []
        for rows in grouped.values():
            selected.extend(self.chunk_selector.select(rows, self.chunk_items))
        selected.sort(key=lambda row: (-row.score, row.item_id))
        global_max = int(config.get("global_max", len(selected)))
        return selected[:global_max]

    def search(self, query: Query) -> Prediction:
        if not self.documents:
            raise RuntimeError("Pipeline must be fitted before search")
        variants = self.query_processor.transform(query)
        retrieval = self.config.get("retrieval", {})
        doc_pool = int(retrieval.get("document_pool", 200))
        chunk_pool = int(retrieval.get("chunk_pool", 500))
        doc_results = self._retrieve(
            variants,
            self.doc_retrievers,
            self.doc_configs,
            self.doc_items,
            doc_pool,
        )

        allowed_chunks: set[str] | None = None
        if doc_results and retrieval.get("cascade", True):
            allowed_docs = {result.item_id for result in doc_results[:doc_pool]}
            allowed_chunks = {
                chunk_id for chunk_id, chunk in self.chunks.items() if chunk.doc_id in allowed_docs
            }
        chunk_results = self._retrieve(
            variants,
            self.chunk_retrievers,
            self.chunk_configs,
            self.chunk_items,
            chunk_pool,
            allowed_ids=allowed_chunks,
        )
        rerank_pool = int(self.config.get("reranker", {}).get("candidate_k", min(200, chunk_pool)))
        rerank_k = int(self.config.get("reranker", {}).get("output_k", rerank_pool))
        rerank_variant = self.config.get("reranker", {}).get("query_variant_id")
        rerank_query = query.text
        if rerank_variant:
            try:
                rerank_query = next(
                    variant.text for variant in variants if variant.variant_id == rerank_variant
                )
            except StopIteration as exc:
                raise ValueError(f"Missing reranker query variant: {rerank_variant}") from exc
        chunk_results = self.reranker.rerank(
            rerank_query,
            chunk_results[:rerank_pool],
            self.chunk_items,
            rerank_k,
        )

        chunk_documents = self.aggregator.aggregate(chunk_results) if chunk_results else []
        document_runs: dict[str, list[SearchResult]] = {}
        document_weights: dict[str, float] = {}
        if doc_results:
            document_runs["document_stage"] = doc_results
            document_weights["document_stage"] = float(retrieval.get("document_stage_weight", 1.0))
        if chunk_documents:
            document_runs["chunk_stage"] = chunk_documents
            document_weights["chunk_stage"] = float(retrieval.get("chunk_stage_weight", 1.0))
        merged_documents = self.fusion.fuse(
            document_runs,
            weights=document_weights,
            k=max(doc_pool, len(chunk_documents)),
        )
        selected_docs = self.doc_selector.select(merged_documents, self.doc_items)
        selected_doc_ids = {result.item_id for result in selected_docs}
        selected_chunks = self._select_chunks(chunk_results, selected_doc_ids)
        return Prediction(
            query_id=query.id,
            relevant_docs=[result.item_id for result in selected_docs],
            relevant_chunks=[self.chunks[result.item_id] for result in selected_chunks],
            diagnostics={
                "query_variants": [variant.variant_id for variant in variants],
                "document_scores": {result.item_id: result.score for result in selected_docs},
                "chunk_scores": {result.item_id: result.score for result in selected_chunks},
            },
        )

    def run(self, queries: list[Query]) -> list[Prediction]:
        return [self.search(query) for query in queries]
