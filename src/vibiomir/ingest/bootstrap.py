"""Convert public crawls of the ViBioMIR corpus into the shared part schema (ticket D3).

Only rows whose ``id`` exists in the official links file with the same URL are kept, so a
bootstrap source can never introduce an id or swap the text of two ids.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from pathlib import Path

from vibiomir.ingest.domains import domain_language, url_domain

log = logging.getLogger("vibiomir.ingest")

# Column names of each supported dataset, mapped onto the part schema.
DATASETS: dict[str, dict[str, str]] = {
    # lighth0406/R2AI_STAGE3_ds corpus.parquet (no dataset card or license as of 2026-10-25)
    "lighth0406": {
        "id": "id",
        "url": "url",
        "final_url": "final_url",
        "title": "title",
        "text": "text",
        "extractor": "extractor",
        "published_at": "published_at",
        "raw_sha256": "raw_sha256",
    },
}


def convert_dataset(
    name: str,
    source_path: str | Path,
    official_links: dict[int, str],
    out_dir: str | Path,
) -> Counter[str]:
    """Stream ``source_path`` row group by row group into ``out_dir/part-<name>-NNNNN``."""
    import pyarrow.parquet as pq

    from vibiomir.ingest.fetch import write_part

    columns = DATASETS[name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen: set[int] = set()
    parquet = pq.ParquetFile(source_path)
    for group in range(parquet.num_row_groups):
        part = out_dir / f"part-{name}-{group:05d}.parquet"
        table = parquet.read_row_group(group, columns=list(columns.values()))
        rows = []
        for record in table.to_pylist():
            doc_id = int(record[columns["id"]])
            url = record[columns["url"]]
            if official_links.get(doc_id) != url:
                counts["id_url_mismatch"] += 1
                continue
            if doc_id in seen:
                counts["duplicate_id"] += 1
                continue
            seen.add(doc_id)
            text = record[columns["text"]] or ""
            title = record[columns["title"]] or ""
            if not text.strip():
                counts["empty_text"] += 1
                continue
            host = url_domain(url)
            body_chars = len(text) - len(title) if text.startswith(title) else len(text)
            counts["ok" if body_chars >= 200 else "short"] += 1
            rows.append({
                "id": doc_id,
                "url": url,
                "final_url": record.get(columns["final_url"]),
                "domain": host,
                "language": domain_language(host),
                "title": title,
                "text": text,
                "source": f"hf_{name}",
                "extractor": f"{name}:{record.get(columns['extractor']) or ''}",
                "http_status": None,
                "fetched_at": None,
                "published_at": record.get(columns["published_at"]),
                "encoding": None,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "raw_sha256": record.get(columns["raw_sha256"]),
            })
        write_part(rows, part)
        if (group + 1) % 50 == 0 or group + 1 == parquet.num_row_groups:
            log.info("%s: %d/%d row groups %s", name, group + 1, parquet.num_row_groups,
                     dict(counts))
    return counts
