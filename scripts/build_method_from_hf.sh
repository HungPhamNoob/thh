#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 <method-name> <hf-dataset-repo> <parquet-file> <revision> [store.sqlite]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

method=$1
dataset=$2
file=$3
revision=$4
store=${5:-"artifacts/full/${method}/store.sqlite"}
config="configs/full/${method}.yaml"

python -m vibiomir index prepare \
  --config "$config" --hf-dataset "$dataset" --hf-file "$file" \
  --hf-revision "$revision" --store "$store"
python scripts/build_method_indexes.py \
  --config "$config" --store "$store" \
  --threads "${VIBIOMIR_THREADS:-8}" --nranks "${VIBIOMIR_COLBERT_NRANKS:-1}"
