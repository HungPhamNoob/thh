#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <method-name> [store.sqlite]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

method=$1
store=${2:-"artifacts/full/${method}/store.sqlite"}
config="configs/full/${method}.yaml"
dataset="vibiomir-corpus/vibiomir-corpus"
revision="13d60924d53e720a7f84c78e3b08b7e17b670b5d"

if [[ ! -f "$config" ]]; then
  echo "Config not found: $config" >&2
  exit 2
fi

python -m vibiomir index prepare \
  --config "$config" \
  --hf-dataset "$dataset" \
  --hf-pattern 'corpus/v1/*.parquet' \
  --hf-revision "$revision" \
  --expected-shards 13 \
  --expected-source-rows 1220598 \
  --expected-documents 1047415 \
  --store "$store"

python scripts/build_method_indexes.py \
  --config "$config" --store "$store" \
  --threads "${VIBIOMIR_THREADS:-8}" \
  --nranks "${VIBIOMIR_COLBERT_NRANKS:-1}"
