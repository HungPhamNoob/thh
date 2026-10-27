#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

stage=${1:-prepare}
case "$stage" in
  prepare)
    python -m vibiomir ingest download-official
    python -m vibiomir ingest manifest
    ;;
  pubmed)
    python -m vibiomir ingest pubmed
    ;;
  crawl)
    python -m vibiomir ingest crawl \
      --skip-domain pubmed.ncbi.nlm.nih.gov --skip-domain vov.vn
    ;;
  wayback)
    python -m vibiomir ingest wayback \
      --skip-domain pubmed.ncbi.nlm.nih.gov --skip-domain vov.vn
    ;;
  merge)
    python -m vibiomir ingest merge --version "${VIBIOMIR_CORPUS_VERSION:-v1}"
    ;;
  *)
    echo "Usage: $0 {prepare|pubmed|crawl|wayback|merge}" >&2
    exit 2
    ;;
esac
