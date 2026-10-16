from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vibiomir.chunking import build_chunker
from vibiomir.config import apply_overrides, load_config
from vibiomir.evaluation import evaluate_submission
from vibiomir.io import (
    load_documents,
    load_queries,
    load_submission,
    validate_submission,
    write_chunks_jsonl,
    write_submission,
)
from vibiomir.pipeline import SearchPipeline


def _config(args: argparse.Namespace) -> dict:
    return apply_overrides(load_config(args.config), args.set or [])


def command_run(args: argparse.Namespace) -> int:
    config = _config(args)
    documents = load_documents(args.corpus)
    queries = load_queries(args.queries)
    pipeline = SearchPipeline(config)
    pipeline.fit(documents)
    predictions = pipeline.run(queries)
    write_submission(predictions, args.output, debug=args.debug)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    return 0


def command_chunk(args: argparse.Namespace) -> int:
    config = _config(args)
    documents = load_documents(args.corpus)
    chunker = build_chunker(config["chunking"])
    chunks = [chunk for document in documents for chunk in chunker.split(document)]
    write_chunks_jsonl(chunks, args.output)
    print(f"Wrote {len(chunks)} chunks to {args.output}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    metrics = evaluate_submission(load_submission(args.predictions), load_submission(args.gold))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    documents = load_documents(args.corpus) if args.corpus else None
    errors = validate_submission(load_submission(args.submission), documents)
    if errors:
        print("Submission validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Submission is valid")
    return 0


def command_show_config(args: argparse.Namespace) -> int:
    print(yaml.safe_dump(_config(args), sort_keys=False, allow_unicode=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vibiomir")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    run_parser = subparsers.add_parser("run", help="Run retrieval and write a submission")
    add_config_flags(run_parser)
    run_parser.add_argument("--corpus", required=True)
    run_parser.add_argument("--queries", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--debug", action="store_true")
    run_parser.set_defaults(handler=command_run)

    chunk_parser = subparsers.add_parser("chunk", help="Export exact and index-time chunk text")
    add_config_flags(chunk_parser)
    chunk_parser.add_argument("--corpus", required=True)
    chunk_parser.add_argument("--output", required=True)
    chunk_parser.set_defaults(handler=command_chunk)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compute macro document/chunk F2")
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--gold", required=True)
    evaluate_parser.set_defaults(handler=command_evaluate)

    validate_parser = subparsers.add_parser("validate", help="Validate submission schema and text provenance")
    validate_parser.add_argument("--submission", required=True)
    validate_parser.add_argument("--corpus")
    validate_parser.set_defaults(handler=command_validate)

    show_parser = subparsers.add_parser("show-config", help="Render inherited config plus overrides")
    add_config_flags(show_parser)
    show_parser.set_defaults(handler=command_show_config)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()

