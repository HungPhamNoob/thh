from __future__ import annotations

from collections import defaultdict
from typing import Any

from vibiomir.aggregation import DocumentAggregator
from vibiomir.fusion import build_fusion
from vibiomir.indexing import ArtifactStore
from vibiomir.query import build_query_processor
from vibiomir.reranking import build_reranker
from vibiomir.retrieval import build_retriever
from vibiomir.schemas import Prediction, Query, QueryVariant, SearchItem, SearchResult
from vibiomir.selection import build_selector


class IndexedSearchPipeline:
    """Full-corpus pipeline backed by SQLite + Qdrant/Lucene/PLAID indexes."""

    def __init__(self, config: dict[str, Any], store_path: str):
        self.config = config
        self.store = ArtifactStore(store_path)
        self.query_processor = build_query_processor(config.get("query", {"type": "identity"}))
        self.doc_configs = config.get("document_retrievers", [])
        self.chunk_configs = config.get("chunk_retrievers", [])
        self.doc_retrievers = [build_retriever(item) for item in self.doc_configs]
        self.chunk_retrievers = [build_retriever(item) for item in self.chunk_configs]
        for retriever in [*self.doc_retrievers, *self.chunk_retrievers]:
            retriever.fit([])
        self.fusion = build_fusion(config.get("fusion", {"type": "rrf"}))
        self.reranker = build_reranker(config.get("reranker", {"type": "none"}))
        self.aggregator = DocumentAggregator(**config.get("document_aggregation", {}))
        self.doc_selector = build_selector(config.get("document_selection", {"max_k": 10}))
        self.chunk_selector = build_selector(config.get("chunk_selection", {"max_k": 10}))

    def close(self) -> None:
        for retriever in [*self.doc_retrievers, *self.chunk_retrievers]:
            close = getattr(retriever, "close", None)
            if close is not None:
                close()
        self.store.close()

    def _retrieve(
        self,
        stage: str,
        variants: list[QueryVariant],
        retrievers: list,
        configs: list[dict[str, Any]],
        default_k: int,
        allowed_doc_ids: set[str] | None = None,
    ) -> tuple[list[SearchResult], dict[str, SearchItem]]:
        runs: dict[str, list[SearchResult]] = {}
        weights: dict[str, float] = {}
        known_items: dict[str, SearchItem] = {}
        for retriever_index, (retriever, retriever_config) in enumerate(zip(retrievers, configs)):
            enabled_variants = set(retriever_config.get("variant_ids", []))
            enabled_languages = set(retriever_config.get("languages", []))
            requested_k = int(retriever_config.get("k", default_k))
            overfetch = int(retriever_config.get("cascade_overfetch_factor", 10))
            fetch_k = requested_k * overfetch if allowed_doc_ids is not None else requested_k
            base_weight = float(retriever_config.get("weight", 1.0))
            for variant in variants:
                if enabled_variants and variant.variant_id not in enabled_variants:
                    continue
                source = f"r{retriever_index}:{retriever.name}:{variant.variant_id}"
                results = retriever.search(variant.text, fetch_k)
                items = self.store.get_items(stage, [row.item_id for row in results])
                known_items.update(items)
                filtered: list[SearchResult] = []
                for row in results:
                    item = items.get(row.item_id)
                    if item is None:
                        continue
                    if enabled_languages and item.payload.get("language") not in enabled_languages:
                        continue
                    row.payload = {**item.payload, **row.payload}
                    row.source = source
                    if allowed_doc_ids is not None and item.doc_id not in allowed_doc_ids:
                        continue
                    filtered.append(row)
                    if len(filtered) >= requested_k:
                        break
                for rank, row in enumerate(filtered, 1):
                    row.rank = rank
                runs[source] = filtered
                weights[source] = base_weight * variant.weight
        fused = self.fusion.fuse(runs, weights=weights, k=default_k) if runs else []
        return fused, known_items

    def _select_chunks(
        self,
        results: list[SearchResult],
        selected_doc_ids: set[str],
        items: dict[str, SearchItem],
    ) -> list[SearchResult]:
        config = self.config.get("chunk_selection", {})
        filtered = [row for row in results if str(row.payload["doc_id"]) in selected_doc_ids]
        if not config.get("per_document", True):
            return self.chunk_selector.select(filtered, items)
        grouped: dict[str, list[SearchResult]] = defaultdict(list)
        for row in filtered:
            grouped[str(row.payload["doc_id"])].append(row)
        selected: list[SearchResult] = []
        for rows in grouped.values():
            selected.extend(self.chunk_selector.select(rows, items))
        selected.sort(key=lambda row: (-row.score, row.item_id))
        return selected[: int(config.get("global_max", len(selected)))]

    def search(self, query: Query) -> Prediction:
        variants = self.query_processor.transform(query)
        retrieval = self.config.get("retrieval", {})
        doc_pool = int(retrieval.get("document_pool", 200))
        chunk_pool = int(retrieval.get("chunk_pool", 500))
        doc_results, doc_items = self._retrieve(
            "document", variants, self.doc_retrievers, self.doc_configs, doc_pool
        )
        allowed_docs: set[str] | None = None
        if retrieval.get("cascade", True) and doc_results:
            allowed_docs = {row.item_id for row in doc_results[:doc_pool]}
        chunk_results, chunk_items = self._retrieve(
            "chunk",
            variants,
            self.chunk_retrievers,
            self.chunk_configs,
            chunk_pool,
            allowed_doc_ids=allowed_docs,
        )
        reranker_config = self.config.get("reranker", {})
        rerank_pool = int(reranker_config.get("candidate_k", min(200, chunk_pool)))
        rerank_k = int(reranker_config.get("output_k", rerank_pool))
        rerank_variant = reranker_config.get("query_variant_id")
        rerank_query = query.text
        if rerank_variant:
            try:
                rerank_query = next(
                    variant.text for variant in variants if variant.variant_id == rerank_variant
                )
            except StopIteration as exc:
                raise ValueError(f"Missing reranker query variant: {rerank_variant}") from exc
        rerank_items = self.store.get_items(
            "chunk", [row.item_id for row in chunk_results[:rerank_pool]]
        )
        chunk_items.update(rerank_items)
        chunk_results = self.reranker.rerank(
            rerank_query, chunk_results[:rerank_pool], rerank_items, rerank_k
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
        missing_doc_items = self.store.get_items(
            "document", [row.item_id for row in merged_documents if row.item_id not in doc_items]
        )
        doc_items.update(missing_doc_items)
        selected_docs = self.doc_selector.select(merged_documents, doc_items)
        selected_doc_ids = {row.item_id for row in selected_docs}
        selected_chunks = self._select_chunks(chunk_results, selected_doc_ids, chunk_items)
        chunks = self.store.get_chunks([row.item_id for row in selected_chunks])
        return Prediction(
            query_id=query.id,
            relevant_docs=[row.item_id for row in selected_docs],
            relevant_chunks=[
                chunks[row.item_id] for row in selected_chunks if row.item_id in chunks
            ],
            diagnostics={
                "query_variants": [variant.variant_id for variant in variants],
                "document_scores": {row.item_id: row.score for row in selected_docs},
                "chunk_scores": {row.item_id: row.score for row in selected_chunks},
            },
        )

    def run(self, queries: list[Query]) -> list[Prediction]:
        return [self.search(query) for query in queries]
