"""Download and read the official ViBioMIR files from Hugging Face."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from vibiomir.ingest.domains import domain_language, url_domain

OFFICIAL_REPO = "AIGuruTinix/ViBioMIR"
# Dataset revision inspected on 2026-10-27. Pinning keeps ids and URLs reproducible even if the
# Hugging Face repository changes later; callers can still pass another explicit revision.
OFFICIAL_REVISION = "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"
OFFICIAL_FILES = ("query.parquet", "links_corpus_vi_en_zh.parquet")
LINKS_FILE = "links_corpus_vi_en_zh.parquet"


def download_official(dest: str | Path, revision: str | None = OFFICIAL_REVISION) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the ingest extra: pip install -e '.[ingest]'") from exc
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return [
        Path(
            hf_hub_download(
                OFFICIAL_REPO, name, repo_type="dataset", revision=revision, local_dir=dest
            )
        )
        for name in OFFICIAL_FILES
    ]


def read_links(path: str | Path) -> list[tuple[int, str]]:
    """Return ``(id, url)`` pairs from the official links parquet, sorted by id."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the ingest extra: pip install -e '.[ingest]'") from exc
    table = pq.read_table(path, columns=["id", "url"])
    ids = table.column("id").to_pylist()
    urls = table.column("url").to_pylist()
    return sorted(zip((int(value) for value in ids), urls))


def manifest_rows(links: list[tuple[int, str]]) -> list[dict]:
    rows = []
    for doc_id, url in links:
        host = url_domain(url)
        rows.append({"id": doc_id, "url": url, "domain": host, "language": domain_language(host)})
    return rows


def sample_links_by_domain(
    links: list[tuple[int, str]], per_domain: int, seed: int = 20261027
) -> list[tuple[int, str]]:
    """Return a deterministic random sample from every host in the official URL list."""
    if per_domain < 1:
        raise ValueError("per_domain must be at least 1")
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for item in links:
        grouped[url_domain(item[1])].append(item)
    rng = random.Random(seed)
    sampled = []
    for domain in sorted(grouped):
        rows = grouped[domain]
        sampled.extend(rng.sample(rows, min(per_domain, len(rows))))
    return sorted(sampled)


def read_id_file(path: str | Path) -> set[int]:
    """Read ids from a parquet file with an ``id`` column or a text file with one id per line."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        column = pq.read_table(path, columns=["id"]).column("id")
        return {int(value) for value in column.to_pylist()}
    with path.open("r", encoding="utf-8") as handle:
        return {int(line) for line in (raw.strip() for raw in handle) if line}
