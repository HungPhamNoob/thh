#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <method-name> <queries.parquet|jsonl> [gold.json] [store.sqlite]" >&2
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
queries=$2
gold=${3:-}
store=${4:-"artifacts/full/${method}/store.sqlite"}
config="configs/full/${method}.yaml"
run_dir="runs/${method}"
mkdir -p "$run_dir"

command=(python -m vibiomir indexed-run
  --config "$config" --store "$store" --queries "$queries"
  --output "$run_dir/submission.json" --metrics-output "$run_dir/metrics.json")
if [[ -n "$gold" ]]; then
  command+=(--gold "$gold")
fi
"${command[@]}"

python -m vibiomir validate --submission "$run_dir/submission.json"
