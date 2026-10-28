#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 <method-name> <hf-dataset-repo> <revision> [parquet-pattern] [store.sqlite]" >&2
  exit 2
fi

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
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

method=$1
dataset=$2
revision=$3
pattern=${4:-"corpus/**/*.parquet"}
store=${5:-"artifacts/full/${method}/store.sqlite"}
config="configs/full/${method}.yaml"

"$python_bin" -m vibiomir index prepare \
  --config "$config" --hf-dataset "$dataset" --hf-pattern "$pattern" \
  --hf-revision "$revision" --store "$store"
"$python_bin" scripts/build_method_indexes.py \
  --config "$config" --store "$store" \
  --threads "${VIBIOMIR_THREADS:-8}" --nranks "${VIBIOMIR_COLBERT_NRANKS:-1}"
