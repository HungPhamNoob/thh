#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <method-name|config.yaml> <corpus-parquet-or-jsonl> [store.sqlite]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

# Resolve interpreter: explicit $PYTHON, repo venv, then system python3.
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
else
  python_bin="python3"
fi
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

method_or_config=$1
corpus=$2
if [[ "$method_or_config" == *.yaml ]]; then
  config=$method_or_config
  method=$(basename "$config" .yaml)
else
  method=$method_or_config
  config="configs/full/${method}.yaml"
fi
store=${3:-"artifacts/full/${method}/store.sqlite"}

if [[ ! -f "$config" ]]; then
  echo "Config not found: $config" >&2
  exit 2
fi

"$python_bin" -m vibiomir index prepare \
  --config "$config" --corpus "$corpus" --store "$store"
"$python_bin" scripts/build_method_indexes.py \
  --config "$config" --store "$store" \
  --threads "${VIBIOMIR_THREADS:-8}" --nranks "${VIBIOMIR_COLBERT_NRANKS:-1}"
