#!/usr/bin/env python3
"""Build every persistent retriever declared by one full-corpus config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from vibiomir.config import load_config
from vibiomir.indexing.late_chunking import build_late_chunking_qdrant_index
from vibiomir.indexing.qdrant import build_qdrant_index
from vibiomir.indexing.sparse import export_impact_vectors
from vibiomir.indexing.store import ArtifactStore
from vibiomir.models import resolve_hf_model


def _lucene_index(collection: Path, index_dir: Path, impact: bool, threads: int) -> None:
    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonVectorCollection" if impact else "JsonCollection",
        "--input",
        str(collection.parent),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
    ]
    if impact:
        command.extend(["--impact", "--pretokenized"])
    else:
        command.extend(["--storePositions", "--storeDocvectors", "--storeRaw"])
    subprocess.run(command, check=True)


def _build_colbert(store_path: Path, stage: str, config: dict, nranks: int) -> None:
    try:
        from colbert import Indexer
        from colbert.infra import ColBERTConfig, Run, RunConfig
    except ImportError as exc:
        raise ImportError("Install the authors' `colbert-ai` package for ColBERTv2") from exc
    root = Path(config["root"])
    collection = root / "collections" / f"{stage}.tsv"
    id_map = Path(config["id_map"])
    with ArtifactStore(store_path) as store:
        count = store.export_colbert(stage, collection, id_map, languages=config.get("languages"))
    print(f"Exported {count} rows for ColBERT: {collection}")
    experiment = config.get("experiment", "vibiomir")
    with Run().context(RunConfig(nranks=nranks, experiment=experiment)):
        index_config = ColBERTConfig(root=str(root), nbits=int(config.get("nbits", 2)))
        checkpoint = resolve_hf_model(
            config["checkpoint"],
            config.get("checkpoint_revision"),
            ["*.json", "*.txt", "*.metadata", "pytorch_model.bin"],
        )
        indexer = Indexer(checkpoint=checkpoint, config=index_config)
        indexer.index(name=config["index_name"], collection=str(collection), overwrite=True)


def _build_one(store_path: Path, stage: str, config: dict, nranks: int, threads: int) -> None:
    kind = config["type"]
    languages = config.get("languages")
    if kind == "qdrant":
        indexing = config.get("indexing", {})
        common = {
            "languages": languages,
            "distance": config.get("distance", "cosine"),
            "on_disk": config.get("on_disk", True),
            "store_text_payload": config.get("store_text_payload", True),
            "url_env": config.get("url_env", "QDRANT_URL"),
            "api_key_env": config.get("api_key_env", "QDRANT_API_KEY"),
            "prefix_env": config.get("prefix_env", "QDRANT_COLLECTION_PREFIX"),
            "local_path": config.get("local_path"),
            "timeout": config.get("timeout", 60.0),
            "prefer_grpc": config.get("prefer_grpc", False),
        }
        if indexing.get("type") == "late_chunking":
            model_config = {key: value for key, value in indexing.items() if key != "type"}
            manifest = build_late_chunking_qdrant_index(
                store_path,
                config["collection_name"],
                config["state_dir"],
                model_config,
                config["encoder"],
                **common,
            )
        else:
            manifest = build_qdrant_index(
                store_path,
                stage,
                config["encoder"],
                config["collection_name"],
                state_dir=config["state_dir"],
                **common,
            )
        print(f"Built Qdrant collection: {manifest}")
        return
    if kind == "pyserini":
        index_dir = Path(config["index_dir"])
        collection = index_dir.parent / f"{index_dir.name}_input" / "part-00000.jsonl"
        with ArtifactStore(store_path) as store:
            count = store.export_jsonl(stage, collection, languages=languages)
        print(f"Exported {count} BM25 rows: {collection}")
        _lucene_index(collection, index_dir, impact=False, threads=threads)
        return
    if kind == "lucene_impact":
        index_dir = Path(config["index_dir"])
        collection = index_dir.parent / f"{index_dir.name}_input" / "part-00000.jsonl"
        count = export_impact_vectors(
            store_path, stage, config["encoder"], collection, languages=languages
        )
        print(f"Exported {count} learned-sparse rows: {collection}")
        _lucene_index(collection, index_dir, impact=True, threads=threads)
        return
    if kind == "colbertv2":
        _build_colbert(store_path, stage, config, nranks)
        return
    raise ValueError(f"Retriever type {kind!r} has no persistent index builder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--stage", choices=["document", "chunk"])
    parser.add_argument("--retriever", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nranks", type=int, default=1, help="ColBERT indexing GPU ranks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    stages = [args.stage] if args.stage else ["document", "chunk"]
    for stage in stages:
        key = "document_retrievers" if stage == "document" else "chunk_retrievers"
        retrievers = config.get(key, [])
        for index, retriever in enumerate(retrievers):
            if args.retriever is not None and index != args.retriever:
                continue
            print(f"Building {stage} retriever #{index}: {retriever['type']}")
            _build_one(Path(args.store), stage, retriever, args.nranks, args.threads)
