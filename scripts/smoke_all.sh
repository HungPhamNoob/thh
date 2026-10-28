#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin=$PYTHON
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
else
  python_bin=python3
fi

"$python_bin" -m unittest discover -s tests -v

for script in scripts/*.sh; do
  bash -n "$script"
done

"$python_bin" - <<'PY'
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
"$python_bin" -m vibiomir index prepare \
  --config "$config" --corpus examples/corpus.jsonl --store "$store"
"$python_bin" -m vibiomir index build-qdrant \
  --config "$config" --store "$store" --stage chunk --retriever 0 --recreate
"$python_bin" -m vibiomir indexed-run \
  --config "$config" --store "$store" --queries examples/queries.jsonl \
  --output "$smoke_dir/submission.json" --gold examples/gold.json \
  --metrics-output "$smoke_dir/metrics.json" --zip-output "$smoke_dir/submission.zip"
"$python_bin" -m vibiomir validate \
  --submission "$smoke_dir/submission.json" --corpus examples/corpus.jsonl \
  --queries examples/queries.jsonl --official
"$python_bin" - <<'PY'
from zipfile import ZipFile

with ZipFile("runs/smoke/submission.zip") as archive:
    assert archive.namelist() == ["submission.json"]
PY
