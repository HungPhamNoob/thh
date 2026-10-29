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

# Reuse an existing store when it verifiably matches the pinned corpus;
# preparing again writes a full temporary copy that may not fit on disk.
# Set FORCE_PREPARE=1 to rebuild the store from the shards anyway.
need_prepare=1
if [[ -f "$store" && "${FORCE_PREPARE:-0}" != "1" ]]; then
  if "$python_bin" - "$store" "$revision" <<'PY'
import json
import sqlite3
import sys

store, revision = sys.argv[1], sys.argv[2]
try:
    con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM store_metadata")}
    docs = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    con.close()
except sqlite3.Error:
    sys.exit(1)
ok = (
    meta.get("store_format_version") == 3
    and meta.get("corpus_revision") == revision
    and meta.get("source_rows") == 1_220_598
    and docs == meta.get("expected_documents")
    and docs > 0
)
sys.exit(0 if ok else 1)
PY
  then
    need_prepare=0
    echo "Reusing valid store $store for corpus revision $revision (FORCE_PREPARE=1 to rebuild)"
  fi
fi

if [[ "$need_prepare" == "1" ]]; then
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
fi

"$python_bin" scripts/run_with_gpu_monitor.py \
  --report "artifacts/full/${method}/gpu_peak_index.json" -- \
  "$python_bin" scripts/build_method_indexes.py \
    --config "$config" --store "$store" \
    --threads "${VIBIOMIR_THREADS:-8}" \
    --nranks "${VIBIOMIR_COLBERT_NRANKS:-1}"
