#!/usr/bin/env bash
set -euo pipefail

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

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$input_path" \
  --index "$index_path" \
  --generator DefaultLuceneDocumentGenerator \
  --threads "${VIBIOMIR_THREADS:-8}" \
  --storePositions --storeDocvectors --storeRaw

