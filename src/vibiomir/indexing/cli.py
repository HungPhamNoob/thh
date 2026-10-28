from __future__ import annotations

import argparse
import json
from pathlib import Path

from vibiomir.chunking import build_chunker
from vibiomir.config import apply_overrides, load_config
from vibiomir.evaluation import evaluate_submission
from vibiomir.io import iter_documents, load_queries, load_submission, write_submission
from vibiomir.pipeline import IndexedSearchPipeline

from .late_chunking import build_late_chunking_qdrant_index
from .numpy import build_numpy_index
from .qdrant import build_qdrant_index
from .sparse import export_impact_vectors
from .store import ArtifactStore, prepare_store


def _config(args: argparse.Namespace) -> dict:
    return apply_overrides(load_config(args.config), args.set or [])


def command_prepare(args: argparse.Namespace) -> int:
    config = _config(args)
    corpus = args.corpus
    if args.hf_dataset:
        if not args.hf_file:
            raise ValueError("--hf-file is required with --hf-dataset")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install `vibiomir-lab[ingest]` for Hugging Face input") from exc
        corpus = hf_hub_download(
            repo_id=args.hf_dataset,
            filename=args.hf_file,
            repo_type="dataset",
            revision=args.hf_revision,
        )
    documents, chunks = prepare_store(
        iter_documents(corpus), build_chunker(config["chunking"]), args.store
    )
    print(f"Prepared {documents} documents and {chunks} chunks in {args.store}")
    return 0


def _retriever_config(config: dict, stage: str, index: int) -> dict:
    key = "document_retrievers" if stage == "document" else "chunk_retrievers"
    try:
        return config[key][index]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"No {stage} retriever #{index} in config") from exc


def command_build_qdrant(args: argparse.Namespace) -> int:
    retriever = _retriever_config(_config(args), args.stage, args.retriever)
    if retriever.get("type") != "qdrant" or "encoder" not in retriever:
        raise ValueError("Selected retriever must have type=qdrant and an encoder block")
    state_dir = args.state_dir or retriever.get("state_dir")
    if not state_dir:
        raise ValueError("Set state_dir in the retriever config or pass --state-dir")
    common = {
        "languages": retriever.get("languages"),
        "distance": retriever.get("distance", "cosine"),
        "recreate": args.recreate,
        "on_disk": retriever.get("on_disk", True),
        "store_text_payload": retriever.get("store_text_payload", True),
        "url_env": retriever.get("url_env", "QDRANT_URL"),
        "api_key_env": retriever.get("api_key_env", "QDRANT_API_KEY"),
        "prefix_env": retriever.get("prefix_env", "QDRANT_COLLECTION_PREFIX"),
        "local_path": args.local_path or retriever.get("local_path"),
        "timeout": retriever.get("timeout", 60.0),
        "prefer_grpc": retriever.get("prefer_grpc", False),
    }
    indexing = retriever.get("indexing", {})
    if indexing.get("type") == "late_chunking":
        if args.stage != "chunk":
            raise ValueError("Late chunking can only build a chunk index")
        model_config = {key: value for key, value in indexing.items() if key != "type"}
        manifest = build_late_chunking_qdrant_index(
            args.store,
            retriever["collection_name"],
            state_dir,
            model_config,
            retriever["encoder"],
            **common,
        )
    else:
        manifest = build_qdrant_index(
            args.store,
            args.stage,
            retriever["encoder"],
            retriever["collection_name"],
            state_dir=state_dir,
            batch_size=args.batch_size,
            **common,
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def command_export_lucene(args: argparse.Namespace) -> int:
    with ArtifactStore(args.store) as store:
        count = store.export_jsonl(args.stage, args.output, languages=args.language)
    print(f"Exported {count} {args.stage} items to {args.output}")
    return 0


def command_build_numpy(args: argparse.Namespace) -> int:
    retriever = _retriever_config(_config(args), args.stage, args.retriever)
    if retriever.get("type") != "numpy_index" or "encoder" not in retriever:
        raise ValueError("Selected retriever must have type=numpy_index and an encoder block")
    manifest = build_numpy_index(
        args.store,
        args.stage,
        retriever["encoder"],
        args.output,
        batch_size=args.batch_size,
        languages=retriever.get("languages"),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def command_export_colbert(args: argparse.Namespace) -> int:
    with ArtifactStore(args.store) as store:
        count = store.export_colbert(
            args.stage, args.collection, args.id_map, languages=args.language
        )
    print(f"Exported {count} {args.stage} items to {args.collection} and {args.id_map}")
    return 0


def command_build_impact(args: argparse.Namespace) -> int:
    retriever = _retriever_config(_config(args), args.stage, args.retriever)
    if retriever.get("type") != "lucene_impact" or "encoder" not in retriever:
        raise ValueError("Selected retriever must have type=lucene_impact and an encoder block")
    count = export_impact_vectors(
        args.store,
        args.stage,
        retriever["encoder"],
        args.output,
        batch_size=args.batch_size,
        languages=retriever.get("languages"),
    )
    print(f"Exported {count} learned-sparse vectors to {args.output}")
    return 0


def command_indexed_run(args: argparse.Namespace) -> int:
    pipeline = IndexedSearchPipeline(_config(args), args.store)
    try:
        predictions = pipeline.run(load_queries(args.queries))
    finally:
        pipeline.close()
    write_submission(predictions, args.output, debug=args.debug)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    if args.gold:
        metrics = evaluate_submission(load_submission(args.output), load_submission(args.gold))
        serialized = json.dumps(metrics, indent=2, ensure_ascii=False)
        print(serialized)
        metrics_path = Path(args.metrics_output or Path(args.output).with_name("metrics.json"))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")
    return 0


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")


def add_index_parsers(subparsers) -> None:
    group = subparsers.add_parser("index", help="Prepare local stores and full-corpus indexes")
    commands = group.add_subparsers(dest="index_command", required=True)

    prepare = commands.add_parser("prepare", help="Stream corpus into a local SQLite store")
    _add_config_flags(prepare)
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", help="JSONL, parquet file, or parquet directory")
    source.add_argument("--hf-dataset", help="Hugging Face dataset repository id")
    prepare.add_argument("--hf-file", help="file in --hf-dataset, usually a parquet shard")
    prepare.add_argument("--hf-revision", help="dataset commit/tag; pin this for real runs")
    prepare.add_argument("--store", required=True)
    prepare.set_defaults(handler=command_prepare)

    qdrant = commands.add_parser("build-qdrant", help="Stream a dense index to Qdrant")
    _add_config_flags(qdrant)
    qdrant.add_argument("--store", required=True)
    qdrant.add_argument("--stage", required=True, choices=["document", "chunk"])
    qdrant.add_argument("--retriever", type=int, default=0)
    qdrant.add_argument("--state-dir")
    qdrant.add_argument("--batch-size", type=int, default=256)
    qdrant.add_argument("--local-path", help="local Qdrant path for development/smoke tests")
    qdrant.add_argument("--recreate", action="store_true")
    qdrant.set_defaults(handler=command_build_qdrant)

    numpy_index = commands.add_parser(
        "build-numpy", help="Build dependency-free exact index for smoke tests"
    )
    _add_config_flags(numpy_index)
    numpy_index.add_argument("--store", required=True)
    numpy_index.add_argument("--stage", required=True, choices=["document", "chunk"])
    numpy_index.add_argument("--retriever", type=int, default=0)
    numpy_index.add_argument("--output", required=True)
    numpy_index.add_argument("--batch-size", type=int, default=256)
    numpy_index.set_defaults(handler=command_build_numpy)

    lucene = commands.add_parser("export-lucene", help="Export JsonCollection input for Pyserini")
    lucene.add_argument("--store", required=True)
    lucene.add_argument("--stage", required=True, choices=["document", "chunk"])
    lucene.add_argument("--output", required=True)
    lucene.add_argument("--language", action="append", help="keep only this language")
    lucene.set_defaults(handler=command_export_lucene)

    impact = commands.add_parser(
        "export-impact", help="Encode SPLADE/BGE sparse vectors for a Lucene impact index"
    )
    _add_config_flags(impact)
    impact.add_argument("--store", required=True)
    impact.add_argument("--stage", required=True, choices=["document", "chunk"])
    impact.add_argument("--retriever", type=int, default=0)
    impact.add_argument("--output", required=True)
    impact.add_argument("--batch-size", type=int, default=128)
    impact.set_defaults(handler=command_build_impact)

    colbert = commands.add_parser("export-colbert", help="Export collection.tsv and pid map")
    colbert.add_argument("--store", required=True)
    colbert.add_argument("--stage", required=True, choices=["document", "chunk"])
    colbert.add_argument("--collection", required=True)
    colbert.add_argument("--id-map", required=True)
    colbert.add_argument("--language", action="append", help="keep only this language")
    colbert.set_defaults(handler=command_export_colbert)

    run = subparsers.add_parser("indexed-run", help="Search persistent full-corpus indexes")
    _add_config_flags(run)
    run.add_argument("--store", required=True)
    run.add_argument("--queries", required=True, help="official query parquet or JSONL")
    run.add_argument("--output", required=True)
    run.add_argument("--gold")
    run.add_argument("--metrics-output")
    run.add_argument("--debug", action="store_true")
    run.set_defaults(handler=command_indexed_run)
