#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Resolve interpreter: explicit $PYTHON, repo venv, then system python3.
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
else
  python_bin="python3"
fi

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <jsonl-directory-or-file> <index-directory>" >&2
  exit 2
fi

input_path=$1
index_path=$2

cleanup_dir=""
if [[ -f "$input_path" ]]; then
  cleanup_dir=$(mktemp -d)
  ln -s "$(realpath "$input_path")" "$cleanup_dir/collection.jsonl"
  input_path=$cleanup_dir
  trap 'rm -rf "$cleanup_dir"' EXIT
fi

"$python_bin" -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$input_path" \
  --index "$index_path" \
  --generator DefaultLuceneDocumentGenerator \
  --threads "${VIBIOMIR_THREADS:-8}" \
  --storePositions --storeDocvectors --storeRaw

