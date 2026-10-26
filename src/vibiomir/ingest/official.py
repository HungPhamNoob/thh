"""Download and read the official ViBioMIR files from Hugging Face."""

from __future__ import annotations

from pathlib import Path

from vibiomir.ingest.domains import domain_language, url_domain

OFFICIAL_REPO = "AIGuruTinix/ViBioMIR"
OFFICIAL_FILES = ("query.parquet", "links_corpus_vi_en_zh.parquet")
LINKS_FILE = "links_corpus_vi_en_zh.parquet"


def download_official(dest: str | Path, revision: str | None = None) -> list[Path]:
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


def read_id_file(path: str | Path) -> set[int]:
    """Read ids from a parquet file with an ``id`` column or a text file with one id per line."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        column = pq.read_table(path, columns=["id"]).column("id")
        return {int(value) for value in column.to_pylist()}
    with path.open("r", encoding="utf-8") as handle:
        return {int(line) for line in (raw.strip() for raw in handle) if line}
