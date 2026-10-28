#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

python -m unittest discover -s tests -v

for script in scripts/*.sh; do
  bash -n "$script"
done

python - <<'PY'
from pathlib import Path
from vibiomir.config import load_config
from vibiomir.query import build_query_processor
from vibiomir.reranking import build_reranker
from vibiomir.retrieval import build_retriever

for path in sorted(Path("configs/full").glob("*.yaml")):
    config = load_config(path)
    build_query_processor(config["query"])
    for retriever in config.get("document_retrievers", []):
        build_retriever(retriever)
    for retriever in config.get("chunk_retrievers", []):
        build_retriever(retriever)
    build_reranker(config["reranker"])
    print(f"Config/factory smoke: {path}")
PY

smoke_dir=runs/smoke
store="$smoke_dir/store.sqlite"
config=configs/smoke/qdrant_hash.yaml
python -m vibiomir index prepare \
  --config "$config" --corpus examples/corpus.jsonl --store "$store"
python -m vibiomir index build-qdrant \
  --config "$config" --store "$store" --stage chunk --retriever 0 --recreate
python -m vibiomir indexed-run \
  --config "$config" --store "$store" --queries examples/queries.jsonl \
  --output "$smoke_dir/submission.json" --gold examples/gold.json \
  --metrics-output "$smoke_dir/metrics.json"
python -m vibiomir validate \
  --submission "$smoke_dir/submission.json" --corpus examples/corpus.jsonl
