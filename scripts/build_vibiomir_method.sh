#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <method-name> [store.sqlite]" >&2
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
store=${2:-"artifacts/full/${method}/store.sqlite"}
config="configs/full/${method}.yaml"
dataset="vibiomir-corpus/vibiomir-corpus"
revision="13d60924d53e720a7f84c78e3b08b7e17b670b5d"

if [[ ! -f "$config" ]]; then
  echo "Config not found: $config" >&2
  exit 2
fi
mkdir -p "artifacts/full/${method}"
export VIBIOMIR_GPU_MAX_GIB=${VIBIOMIR_GPU_MAX_GIB:-3.0}
export VIBIOMIR_GPU_REPORT="artifacts/full/${method}/gpu_report.json"
exec > >(tee -a "artifacts/full/${method}/build.log") 2>&1

"$python_bin" -m vibiomir index prepare \
  --config "$config" \
  --hf-dataset "$dataset" \
  --hf-pattern 'corpus/v1/*.parquet' \
  --hf-revision "$revision" \
  --expected-shards 13 \
  --expected-source-rows 1220598 \
  --expected-documents 1047384 \
  --expected-empty-rows 173214 \
  --official-vibiomir-corpus \
  --store "$store"

"$python_bin" scripts/run_with_gpu_monitor.py \
  --report "artifacts/full/${method}/gpu_peak_index.json" -- \
  "$python_bin" scripts/build_method_indexes.py \
    --config "$config" --store "$store" \
    --threads "${VIBIOMIR_THREADS:-8}" \
    --nranks "${VIBIOMIR_COLBERT_NRANKS:-1}"
