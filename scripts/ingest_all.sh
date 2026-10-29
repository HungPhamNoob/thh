#!/usr/bin/env bash
set -euo pipefail

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

stage=${1:-prepare}
official_revision=${VIBIOMIR_DATASET_REVISION:-71be0e1239a19dfd7a717e2bc13e3bec21bd0334}
case "$stage" in
  prepare)
    "$python_bin" -m vibiomir ingest download-official --revision "$official_revision"
    "$python_bin" -m vibiomir ingest manifest
    ;;
  pilot)
    "$python_bin" -m vibiomir ingest download-official --revision "$official_revision"
    "$python_bin" -m vibiomir ingest manifest
    "$python_bin" -m vibiomir ingest crawl \
      --state data/interim/pilot/crawl_state.sqlite \
      --out data/interim/pilot/parts \
      --samples data/raw/html_samples_pilot \
      --log-file data/interim/pilot/crawl.log \
      --skip-domain pubmed.ncbi.nlm.nih.gov \
      --sample-per-domain "${VIBIOMIR_PILOT_PER_DOMAIN:-5}" \
      --max-attempts 1 --rps 0.5 --max-rps 1.0 --concurrency 32
    "$python_bin" -m vibiomir ingest audit-html \
      --samples data/raw/html_samples_pilot \
      --output data/interim/pilot/html_audit.csv
    ;;
  pubmed)
    "$python_bin" -m vibiomir ingest pubmed
    ;;
  crawl)
    "$python_bin" -m vibiomir ingest crawl \
      --skip-domain pubmed.ncbi.nlm.nih.gov --skip-domain vov.vn
    ;;
  wayback)
    "$python_bin" -m vibiomir ingest wayback \
      --skip-domain pubmed.ncbi.nlm.nih.gov --skip-domain vov.vn
    ;;
  merge)
    "$python_bin" -m vibiomir ingest merge --version "${VIBIOMIR_CORPUS_VERSION:-v1}"
    ;;
  audit)
    "$python_bin" -m vibiomir ingest audit-html
    ;;
  status)
    "$python_bin" -m vibiomir ingest status
    ;;
  *)
    echo "Usage: $0 {prepare|pilot|pubmed|crawl|wayback|audit|status|merge}" >&2
    exit 2
    ;;
esac
