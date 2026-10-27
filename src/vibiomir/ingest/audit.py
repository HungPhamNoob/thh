"""Offline quality audit for the raw HTML samples kept by the crawler.

The crawler intentionally stores only a small number of successful pages plus problem pages
for each domain.  Re-running extraction on those files is cheap and makes selector regressions,
short pages, decoding damage and challenge pages visible without touching the websites again.
"""

from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

from vibiomir.ingest.extract import MIN_OK_CHARS, extract_page
from vibiomir.ingest.fetch import classify_response

AUDIT_COLUMNS = (
    "domain",
    "id",
    "path",
    "status",
    "extractor",
    "encoding",
    "title_chars",
    "body_chars",
    "replacement_chars",
    "error",
)


def inspect_html_sample(path: str | Path, samples_root: str | Path) -> dict:
    """Inspect one ``<domain>/<id>.html.gz`` sample without network access."""
    path = Path(path)
    root = Path(samples_root)
    domain = path.parent.name
    try:
        doc_id = int(path.name.removesuffix(".html.gz"))
    except ValueError:
        doc_id = -1

    row = {
        "domain": domain,
        "id": doc_id,
        "path": str(path.relative_to(root)),
        "status": "failed",
        "extractor": "",
        "encoding": "",
        "title_chars": 0,
        "body_chars": 0,
        "replacement_chars": 0,
        "error": "",
    }
    try:
        with gzip.open(path, "rb") as handle:
            body = handle.read()
        verdict = classify_response(200, body)
        if verdict is not None:
            row["status"] = verdict
            row["error"] = "challenge or error page"
            return row

        extracted = extract_page(body, None, domain)
        body_chars = max(0, len(extracted.text) - len(extracted.title))
        row.update(
            status="gone" if extracted.soft_404 else (
                "ok" if body_chars >= MIN_OK_CHARS else "short"
            ),
            extractor=extracted.extractor,
            encoding=extracted.encoding,
            title_chars=len(extracted.title),
            body_chars=body_chars,
            replacement_chars=extracted.text.count("\ufffd"),
        )
    except Exception as exc:  # noqa: BLE001 - the report must include unreadable samples
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def audit_html_samples(samples_dir: str | Path, output: str | Path) -> tuple[list[dict], dict]:
    """Audit every saved sample, write CSV and return rows plus an aggregate summary."""
    samples_dir = Path(samples_dir)
    paths = sorted(samples_dir.glob("*/*.html.gz"))
    if not paths:
        raise FileNotFoundError(f"No <domain>/<id>.html.gz samples found in {samples_dir}")

    rows = [inspect_html_sample(path, samples_dir) for path in paths]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    by_domain: dict[str, Counter] = defaultdict(Counter)
    extractors: Counter = Counter()
    for row in rows:
        by_domain[row["domain"]][row["status"]] += 1
        if row["extractor"]:
            extractors[row["extractor"]] += 1
    summary = {
        "samples": len(rows),
        "domains": len(by_domain),
        "status": dict(Counter(row["status"] for row in rows)),
        "extractors": dict(extractors),
        "by_domain": {domain: dict(counts) for domain, counts in sorted(by_domain.items())},
    }
    return rows, summary
