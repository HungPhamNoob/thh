from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from vibiomir.chunking import build_chunker
from vibiomir.config import apply_overrides, load_config
from vibiomir.evaluation import evaluate_submission
from vibiomir.io import (
    iter_documents,
    load_queries,
    load_submission,
    validate_submission,
    write_submission,
    write_submission_zip,
)
from vibiomir.pipeline import IndexedSearchPipeline

from .late_chunking import build_late_chunking_qdrant_index
from .numpy import build_numpy_index
from .qdrant import build_qdrant_index
from .sparse import export_impact_vectors
from .store import ArtifactStore, prepare_store

VIBIOMIR_CORPUS_DATASET = "vibiomir-corpus/vibiomir-corpus"
VIBIOMIR_CORPUS_REVISION = "13d60924d53e720a7f84c78e3b08b7e17b670b5d"
VIBIOMIR_CORPUS_PATTERN = "corpus/v1/*.parquet"
VIBIOMIR_CORPUS_SHARDS = [f"corpus/v1/part-{index:05d}.parquet" for index in range(13)]
VIBIOMIR_SOURCE_ROWS = 1_220_598
VIBIOMIR_DOCUMENTS = 1_047_415
VIBIOMIR_EMPTY_ROWS = 173_183


def _config(args: argparse.Namespace) -> dict:
    return apply_overrides(load_config(args.config), args.set or [])


def _git_commit() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _model_records(value: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if isinstance(value, dict):
        if "model_name" in value:
            records.append(
                {
                    "model_name": value["model_name"],
                    "revision": value.get("revision"),
                    "type": value.get("type"),
                }
            )
        if "checkpoint" in value:
            records.append(
                {
                    "model_name": value["checkpoint"],
                    "revision": value.get("checkpoint_revision"),
                    "type": value.get("type"),
                }
            )
        for child in value.values():
            records.extend(_model_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_model_records(child))
    unique: dict[tuple[object, object, object], dict[str, object]] = {}
    for record in records:
        key = (record["model_name"], record["revision"], record["type"])
        unique[key] = record
    return list(unique.values())


def _verify_parquet_corpus(
    paths: list[Path],
    *,
    revision: str,
    required_text_field: str,
    expected_source_rows: int | None,
    expected_documents: int | None,
    expected_empty_rows: int | None,
) -> tuple[int, int, int]:
    """Validate schema and coverage before creating/replacing the SQLite store."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Install `vibiomir-lab[ingest]` to verify parquet rows") from exc
    source_rows = non_empty = empty = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if "id" not in names:
            raise ValueError(f"Corpus schema error at {path}: required field 'id' is absent")
        if required_text_field not in names:
            raise ValueError(
                f"Corpus schema error at {path}: required field {required_text_field!r} is absent"
            )
        source_rows += parquet.metadata.num_rows
        for batch in parquet.iter_batches(columns=[required_text_field], batch_size=4096):
            for value in batch.column(0).to_pylist():
                if value is not None and str(value).strip():
                    non_empty += 1
                else:
                    empty += 1
    actual = {
        "source rows": source_rows,
        "non-empty documents": non_empty,
        "empty rows": empty,
    }
    expected = {
        "source rows": expected_source_rows,
        "non-empty documents": expected_documents,
        "empty rows": expected_empty_rows,
    }
    mismatches = [
        f"{name}: expected {expected[name]}, found {actual[name]}"
        for name in actual
        if expected[name] is not None and expected[name] != actual[name]
    ]
    if mismatches:
        shard_list = "\n".join(f"- {path}" for path in paths)
        raise ValueError(
            f"Corpus verification failed at revision {revision}:\n"
            + "\n".join(mismatches)
            + f"\nMatched shards:\n{shard_list}"
        )
    print(
        f"Verified revision {revision}: {source_rows} source rows, "
        f"{non_empty} non-empty documents, {empty} empty rows across {len(paths)} shards"
    )
    return source_rows, non_empty, empty


def command_prepare(args: argparse.Namespace) -> int:
    config = _config(args)
    official_corpus = bool(getattr(args, "official_vibiomir_corpus", False))
    if official_corpus:
        if args.hf_dataset != VIBIOMIR_CORPUS_DATASET:
            raise ValueError(f"Official corpus requires --hf-dataset {VIBIOMIR_CORPUS_DATASET}")
        if args.hf_revision != VIBIOMIR_CORPUS_REVISION:
            raise ValueError(f"Official corpus requires revision {VIBIOMIR_CORPUS_REVISION}")
        if args.hf_pattern != [VIBIOMIR_CORPUS_PATTERN]:
            raise ValueError(f"Official corpus requires pattern {VIBIOMIR_CORPUS_PATTERN}")
        args.expected_shards = len(VIBIOMIR_CORPUS_SHARDS)
        args.expected_source_rows = VIBIOMIR_SOURCE_ROWS
        args.expected_documents = VIBIOMIR_DOCUMENTS
        args.expected_empty_rows = VIBIOMIR_EMPTY_ROWS
        args.required_text_field = "text"
    corpus = args.corpus
    parquet_files: list[Path] = []
    shard_names: list[str] = []
    resolved_revision = args.hf_revision or "local"
    if args.hf_dataset:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise ImportError("Install `vibiomir-lab[ingest]` for Hugging Face input") from exc
        if not args.hf_revision:
            raise ValueError("--hf-revision is required for reproducible Hugging Face input")
        if args.hf_file:
            corpus = hf_hub_download(
                repo_id=args.hf_dataset,
                filename=args.hf_file,
                repo_type="dataset",
                revision=args.hf_revision,
            )
            if str(corpus).endswith(".parquet"):
                parquet_files = [Path(corpus)]
        else:
            if not args.hf_pattern:
                raise ValueError("--hf-pattern is required when --hf-file is omitted")
            patterns = args.hf_pattern
            snapshot = Path(
                snapshot_download(
                    repo_id=args.hf_dataset,
                    repo_type="dataset",
                    revision=args.hf_revision,
                    allow_patterns=patterns,
                    max_workers=args.hf_max_workers,
                )
            )
            resolved_revision = snapshot.resolve().name
            parquet_files = sorted(
                {
                    path
                    for pattern in patterns
                    for path in snapshot.glob(pattern)
                    if path.is_file() and path.suffix == ".parquet"
                }
            )
            shard_names = [path.relative_to(snapshot).as_posix() for path in parquet_files]
            if not parquet_files:
                raise ValueError(f"No parquet files matched {patterns} in {args.hf_dataset}")
            corpus = parquet_files
    elif Path(corpus).is_dir():
        parquet_files = sorted(Path(corpus).rglob("*.parquet"))
    elif str(corpus).endswith(".parquet"):
        parquet_files = [Path(corpus)]
    if args.expected_shards is not None:
        actual_shards = len(parquet_files)
        if actual_shards != args.expected_shards:
            raise ValueError(
                f"Expected {args.expected_shards} parquet shards, found {actual_shards}; "
                f"revision={resolved_revision}; shards={shard_names or parquet_files}"
            )
    if official_corpus and shard_names != VIBIOMIR_CORPUS_SHARDS:
        raise ValueError(
            f"Official corpus shard list mismatch at revision {resolved_revision}: {shard_names}"
        )
    source_rows = non_empty = empty_rows = None
    if parquet_files:
        source_rows, non_empty, empty_rows = _verify_parquet_corpus(
            parquet_files,
            revision=resolved_revision,
            required_text_field=getattr(args, "required_text_field", "text"),
            expected_source_rows=args.expected_source_rows,
            expected_documents=args.expected_documents,
            expected_empty_rows=getattr(args, "expected_empty_rows", None),
        )
    metadata = {
        "method": Path(args.config).stem,
        "config_path": str(args.config),
        "resolved_config": config,
        "chunker_config": config["chunking"],
        "corpus_dataset": args.hf_dataset or "local",
        "corpus_revision": resolved_revision,
        "corpus_shards": shard_names or [str(path) for path in parquet_files],
        "source_rows": source_rows,
        "expected_documents": non_empty,
        "empty_rows": empty_rows,
        "code_commit": _git_commit(),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    documents, chunks = prepare_store(
        iter_documents(corpus),
        build_chunker(config["chunking"]),
        args.store,
        store_metadata=metadata,
    )
    if args.expected_documents is not None and documents != args.expected_documents:
        raise ValueError(
            f"Expected {args.expected_documents} non-empty documents, prepared {documents}"
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
    queries = load_queries(args.queries)
    query_ids = [query.id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Query input contains duplicate ids")
    if any(not query.text.strip() for query in queries):
        raise ValueError("Query input contains null/empty query text")
    if args.official and len(queries) != 1200:
        raise ValueError(f"Official run requires exactly 1200 queries, found {len(queries)}")
    config = _config(args)
    pipeline = IndexedSearchPipeline(config, args.store)
    try:
        predictions = pipeline.run(queries)
    finally:
        pipeline.close()
    write_submission(predictions, args.output, debug=args.debug)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    rows = load_submission(args.output)
    cited_ids = {
        str(doc_id)
        for row in rows
        for doc_id in row.get("relevant_docs", [])
    } | {
        str(chunk["doc_id"])
        for row in rows
        for chunk in row.get("relevant_chunks", [])
        if isinstance(chunk, dict) and "doc_id" in chunk
    }
    with ArtifactStore(args.store) as store:
        documents = store.get_documents(cited_ids).values()
        store_metadata = store.metadata()
    errors = validate_submission(
        rows,
        documents,
        expected_query_ids=set(query_ids),
        require_numeric_ids=args.official,
        strict_fields=args.official,
    )
    if errors:
        raise ValueError("Submission validation failed before packaging:\n- " + "\n- ".join(errors))
    if args.zip_output:
        write_submission_zip(args.output, args.zip_output)
        print(f"Wrote competition ZIP to {args.zip_output}")
    metrics_path: Path | None = None
    if args.gold:
        metrics = evaluate_submission(load_submission(args.output), load_submission(args.gold))
        payload = {
            "metric_version": "BTC-2026-10-25-macro-f2-v1",
            "gold_sha256": _sha256(args.gold),
            "submission_sha256": _sha256(args.output),
            "query_count": len(queries),
            "aggregate": metrics,
        }
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        print(serialized)
        metrics_path = Path(args.metrics_output or Path(args.output).with_name("metrics.json"))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")
    run_dir = Path(args.output).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    qdrant_manifests: list[dict] = []
    for retriever in [
        *config.get("document_retrievers", []),
        *config.get("chunk_retrievers", []),
    ]:
        state_dir = retriever.get("state_dir")
        if state_dir and Path(state_dir, "manifest.json").exists():
            qdrant_manifests.append(
                json.loads(Path(state_dir, "manifest.json").read_text(encoding="utf-8"))
            )
    manifest = {
        "format_version": 1,
        "method": Path(args.config).stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "config_path": str(args.config),
        "query_path": str(Path(args.queries).resolve()),
        "query_sha256": _sha256(args.queries),
        "query_count": len(queries),
        "store_path": str(Path(args.store).resolve()),
        "store_metadata": store_metadata,
        "models": _model_records(config),
        "indexes": qdrant_manifests,
        "submission_json_sha256": _sha256(args.output),
        "submission_zip_sha256": _sha256(args.zip_output) if args.zip_output else None,
        "metrics_status": "available" if metrics_path else "unavailable",
        "metrics_path": str(metrics_path.resolve()) if metrics_path else None,
        "prediction_rows": len(rows),
        "retrieved_document_references": sum(len(row["relevant_docs"]) for row in rows),
        "unique_retrieved_documents": len(cited_ids),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
    prepare.add_argument("--hf-file", help="one file in --hf-dataset (legacy/small runs)")
    prepare.add_argument(
        "--hf-pattern",
        action="append",
        help="snapshot pattern; repeatable, e.g. corpus/v1/*.parquet",
    )
    prepare.add_argument("--hf-revision", help="dataset commit SHA; required with --hf-dataset")
    prepare.add_argument("--hf-max-workers", type=int, default=4)
    prepare.add_argument("--expected-shards", type=int)
    prepare.add_argument("--expected-source-rows", type=int)
    prepare.add_argument("--expected-documents", type=int, help="non-empty documents after filter")
    prepare.add_argument("--expected-empty-rows", type=int)
    prepare.add_argument("--required-text-field", default="text")
    prepare.add_argument(
        "--official-vibiomir-corpus",
        action="store_true",
        help="enforce the pinned ViBioMIR v1 dataset, revision, shard names and counts",
    )
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
    run.add_argument("--zip-output", help="write a root-level submission.json in this ZIP")
    run.add_argument("--gold")
    run.add_argument("--metrics-output")
    run.add_argument("--debug", action="store_true")
    run.add_argument("--official", action="store_true", help="enforce 1200-query BTC output")
    run.set_defaults(handler=command_indexed_run)
