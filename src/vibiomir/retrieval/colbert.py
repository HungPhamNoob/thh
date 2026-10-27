from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from vibiomir.models import resolve_hf_model
from vibiomir.schemas import SearchItem, SearchResult

from .base import Retriever


@dataclass(slots=True)
class ColBERTv2Retriever(Retriever):
    """Adapter for the authors' ColBERTv2/PLAID index and search implementation."""

    index_name: str
    root: str
    id_map: str
    checkpoint: str | None = None
    checkpoint_revision: str | None = None
    experiment: str = "vibiomir"
    name: str = "colbertv2_plaid"
    ncells: int | None = None
    centroid_score_threshold: float | None = None
    ndocs: int | None = None
    overfetch_factor: int = 5
    _searcher: object | None = field(default=None, init=False, repr=False)
    _ids: list[str] = field(default_factory=list, init=False, repr=False)

    def _load(self) -> object:
        if self._searcher is None:
            try:
                from colbert import Searcher
                from colbert.infra import ColBERTConfig, Run, RunConfig
            except ImportError as exc:
                raise ImportError(
                    "Install the authors' `colbert-ai` package for ColBERTv2/PLAID"
                ) from exc
            params = {
                key: value
                for key, value in {
                    "root": self.root,
                    "ncells": self.ncells,
                    "centroid_score_threshold": self.centroid_score_threshold,
                    "ndocs": self.ndocs,
                }.items()
                if value is not None
            }
            config = ColBERTConfig(**params)
            kwargs = {"index": self.index_name, "config": config}
            if self.checkpoint:
                kwargs["checkpoint"] = resolve_hf_model(
                    self.checkpoint,
                    self.checkpoint_revision,
                    ["*.json", "*.txt", "*.metadata", "pytorch_model.bin"],
                )
            with Run().context(RunConfig(nranks=1, experiment=self.experiment)):
                self._searcher = Searcher(**kwargs)
            with open(self.id_map, "r", encoding="utf-8") as handle:
                self._ids = [str(json.loads(line)) for line in handle if line.strip()]
        return self._searcher

    def fit(self, items: Sequence[SearchItem]) -> None:
        self._load()

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        searcher = self._load()
        fetch = k if allowed_ids is None else max(k, k * self.overfetch_factor)
        pids, _ranks, scores = searcher.search(query, k=fetch)
        output: list[SearchResult] = []
        for pid, score in zip(pids, scores):
            item_id = self._ids[int(pid)]
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            output.append(
                SearchResult(
                    item_id=item_id,
                    score=float(score),
                    rank=len(output) + 1,
                    source=self.name,
                )
            )
            if len(output) >= k:
                break
        return output
