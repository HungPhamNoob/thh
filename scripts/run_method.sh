#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <method-name> <queries.parquet|jsonl> [gold.json] [store.sqlite]" >&2
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
queries=$2
gold=${3:-}
store=${4:-"artifacts/full/${method}/store.sqlite"}
config="configs/full/${method}.yaml"
run_dir="runs/${method}"
mkdir -p "$run_dir"
export VIBIOMIR_GPU_MAX_GIB=${VIBIOMIR_GPU_MAX_GIB:-3.0}
export VIBIOMIR_GPU_REPORT="$run_dir/gpu_report.json"
exec > >(tee -a "$run_dir/run.log") 2>&1

command=("$python_bin" -m vibiomir indexed-run
  --config "$config" --store "$store" --queries "$queries"
  --output "$run_dir/submission.json" --zip-output "$run_dir/submission.zip"
  --metrics-output "$run_dir/metrics.json" --official)
if [[ -n "$gold" ]]; then
  command+=(--gold "$gold")
fi
"$python_bin" scripts/run_with_gpu_monitor.py \
  --report "$run_dir/gpu_peak_retrieval.json" -- "${command[@]}"

"$python_bin" -m vibiomir validate \
  --submission "$run_dir/submission.json" --queries "$queries" --official --store "$store"
