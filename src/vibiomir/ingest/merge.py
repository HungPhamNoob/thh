"""Merge crawl, PubMed and fallback parts into one corpus row per official id (D6).

Parts may hold several rows for one id (a page re-crawled after ``reset``, a crash between
writing a part and committing state, a Wayback copy of a page that was crawled later). The
winner per id is the first by: text long enough (``ok``) > source priority > newest
``fetched_at``. Ids without any row are kept with empty text and ``source=missing``, so the
corpus always has exactly one row per id in ``links_corpus``.

Rows whose page redirected to the site's front page (a removed article) never win. Before a
row is written, ``clean.clean_text`` drops boilerplate paragraphs learned per domain.

Two passes keep memory low: the first reads only what ranking needs, the second copies the
winning rows into id buckets, and each bucket is then sorted, joined with the manifest and
written as one final part.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vibiomir.ingest.clean import BoilerplateModel, clean_text
from vibiomir.ingest.extract import MIN_OK_CHARS
from vibiomir.ingest.fetch import PART_COLUMNS, redirected_home

log = logging.getLogger("vibiomir.ingest")

SOURCE_PRIORITY = ("crawl", "pubmed_eutils", "wayback")
MISSING = "missing"
BUCKET_ROWS = 100_000


@dataclass(slots=True)
class MergeReport:
    rows: int
    unique_ids: int
    by_language: dict[str, Counter]
    by_domain: dict[str, Counter]
    url_mismatches: int
    duplicate_rows: int
    parts: int
    home_redirects: int = 0
    cleaned_docs: int = 0
    cleaned_chars: int = 0


def part_schema():
    import pyarrow as pa

    return pa.schema([(name, getattr(pa, kind)()) for name, kind in PART_COLUMNS])


def source_rank(source: str | None) -> int:
    """Lower is better; unknown sources (e.g. ``hf_*`` passed explicitly) come last."""
    try:
        return SOURCE_PRIORITY.index(source or "")
    except ValueError:
        return len(SOURCE_PRIORITY)


def is_ok(title: str | None, text: str | None) -> bool:
    return len(text or "") - len(title or "") >= MIN_OK_CHARS


def find_parts(dirs: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for folder in dirs:
        paths.extend(sorted(Path(folder).glob("part-*.parquet")))
    return paths


def select_winners(parts: Sequence[Path]):
    """Return ``(winners, total_rows, home_redirects)``.

    ``winners`` maps part index -> sorted row indices; rows that redirected to the front page
    are dropped before ranking and counted in ``home_redirects``.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    keys = []
    home_redirects = 0
    for index, path in enumerate(parts):
        table = pq.read_table(
            path, columns=["id", "url", "final_url", "source", "fetched_at", "title", "text"]
        )
        body = pc.subtract(
            pc.utf8_length(pc.fill_null(table["text"], "")),
            pc.utf8_length(pc.fill_null(table["title"], "")),
        )
        ranks = [source_rank(s) for s in table["source"].to_pylist()]
        keys.append(
            pa.table(
                {
                    "id": table["id"].cast(pa.int64()),
                    "bad": pc.less(body, MIN_OK_CHARS),
                    "rank": pa.array(ranks, pa.int8()),
                    "fetched_at": pc.fill_null(table["fetched_at"].cast(pa.string()), ""),
                    "part": pa.array(np.full(table.num_rows, index, np.int32)),
                    "row": pa.array(np.arange(table.num_rows, dtype=np.int32)),
                }
            )
        )
        home = [redirected_home(u or "", f) for u, f in
                zip(table["url"].to_pylist(), table["final_url"].to_pylist())]
        if any(home):
            home_redirects += sum(home)
            keys[-1] = keys[-1].filter(pc.invert(pa.array(home)))
    if not keys:
        return {}, 0, 0
    combined = pa.concat_tables(keys)
    order = pc.sort_indices(
        combined,
        sort_keys=[("id", "ascending"), ("bad", "ascending"), ("rank", "ascending"),
                   ("fetched_at", "descending")],
    )
    ranked = combined.take(order)
    ids = ranked["id"].to_numpy()
    first = np.ones(len(ids), dtype=bool)
    first[1:] = ids[1:] != ids[:-1]
    part_index = ranked["part"].to_numpy()[first]
    row_index = ranked["row"].to_numpy()[first]
    winners: dict[int, np.ndarray] = {}
    for index in np.unique(part_index):
        winners[int(index)] = np.sort(row_index[part_index == index])
    return winners, combined.num_rows + home_redirects, home_redirects


def merge_corpus(
    manifest_path: str | Path,
    part_dirs: Sequence[str | Path],
    out_dir: str | Path,
    coverage_path: str | Path | None = None,
    version: str = "v0",
    bucket_rows: int = BUCKET_ROWS,
    clean: bool = True,
) -> MergeReport:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    manifest = pq.read_table(manifest_path, columns=["id", "url", "domain", "language"])
    manifest = manifest.sort_by("id")
    manifest_ids = manifest["id"].to_numpy()
    if len(np.unique(manifest_ids)) != len(manifest_ids):
        raise ValueError("manifest has duplicate ids")
    # Bucket boundaries by position, because official ids are sparse (583 .. 7.4M).
    starts = manifest_ids[::bucket_rows]

    parts = find_parts(part_dirs)
    log.info("merging %d parts into %d buckets", len(parts), len(starts))
    winners, total_rows, home_redirects = select_winners(parts)
    model = BoilerplateModel()

    staging = out_dir.with_name(out_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    schema = part_schema()
    writers: dict[int, pq.ParquetWriter] = {}
    try:
        for index, rows in winners.items():
            table = pq.read_table(parts[index]).select([n for n, _ in PART_COLUMNS])
            table = table.cast(schema).take(pa.array(rows))
            if clean:
                for domain, text in zip(table["domain"].to_pylist(), table["text"].to_pylist()):
                    model.observe(domain, text or "")
            bucket = np.searchsorted(starts, table["id"].to_numpy(), side="right") - 1
            for b in np.unique(bucket):
                if int(b) not in writers:
                    writers[int(b)] = pq.ParquetWriter(staging / f"bucket-{b:03d}.parquet", schema)
                writers[int(b)].write_table(table.filter(pa.array(bucket == b)))
    finally:
        for writer in writers.values():
            writer.close()

    boilerplate = model.boilerplate() if clean else None
    cleaned_docs = cleaned_chars = 0
    tmp_out = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True)
    by_language: dict[str, Counter] = defaultdict(Counter)
    by_domain: dict[str, Counter] = defaultdict(Counter)
    url_mismatches = rows_written = unique_ids = 0
    for b in range(len(starts)):
        expected = manifest.slice(b * bucket_rows, bucket_rows)
        path = staging / f"bucket-{b:03d}.parquet"
        found = pq.read_table(path) if path.exists() else schema.empty_table()
        if boilerplate is not None:
            found, docs, chars = _clean_table(found, boilerplate)
            cleaned_docs += docs
            cleaned_chars += chars
        merged, mismatches = _fill_bucket(expected, found, schema)
        url_mismatches += mismatches
        rows_written += merged.num_rows
        unique_ids += len(pc.unique(merged["id"]))
        _count(merged, by_language, by_domain)
        pq.write_table(merged, tmp_out / f"part-{b:05d}.parquet", compression="zstd")
    shutil.rmtree(staging)

    report = MergeReport(
        rows=rows_written,
        unique_ids=unique_ids,
        by_language=dict(by_language),
        by_domain=dict(by_domain),
        url_mismatches=url_mismatches,
        duplicate_rows=total_rows - home_redirects - sum(len(r) for r in winners.values()),
        parts=len(parts),
        home_redirects=home_redirects,
        cleaned_docs=cleaned_docs,
        cleaned_chars=cleaned_chars,
    )
    if not report.rows == report.unique_ids == len(manifest_ids):
        raise RuntimeError(
            f"merged {report.rows} rows / {report.unique_ids} ids, manifest has {len(manifest_ids)}"
        )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp_out, out_dir)
    # Next to the folder, not inside it, so pq.read_table(out_dir) sees only parquet files.
    _write_info(out_dir.with_name(out_dir.name + "_info.json"), report, part_dirs, version)
    if coverage_path is not None:
        write_coverage(report, coverage_path)
    return report


def _clean_table(table, boilerplate: dict[str, frozenset[str]], batch_rows: int = 10_000):
    """Apply ``clean_text`` per row; return ``(table, changed_docs, removed_chars)``.

    Rows are converted to Python strings ``batch_rows`` at a time, so a bucket of long pages does
    not need several copies of all its text in memory at once.
    """
    import pyarrow as pa

    text_chunks, hash_chunks = [], []
    changed = removed = 0
    for offset in range(0, table.num_rows, batch_rows):
        batch = table.slice(offset, batch_rows)
        texts, hashes = [], []
        for domain, text, digest in zip(batch["domain"].to_pylist(), batch["text"].to_pylist(),
                                        batch["text_sha256"].to_pylist()):
            cleaned = clean_text(text or "", boilerplate.get(domain, frozenset()))
            if cleaned != (text or ""):
                changed += 1
                removed += len(text or "") - len(cleaned)
                digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            texts.append(cleaned)
            hashes.append(digest)
        text_chunks.append(pa.array(texts, pa.string()))
        hash_chunks.append(pa.array(hashes, pa.string()))
    names = table.schema.names
    table = table.set_column(names.index("text"), "text",
                             pa.chunked_array(text_chunks, pa.string()))
    table = table.set_column(names.index("text_sha256"), "text_sha256",
                             pa.chunked_array(hash_chunks, pa.string()))
    return table, changed, removed


def _fill_bucket(expected, found, schema):
    """Left-join the manifest slice with the winning rows; return ``(table, url_mismatches)``."""
    import pyarrow as pa
    import pyarrow.compute as pc

    found = found.sort_by("id")
    position = pc.index_in(expected["id"], value_set=found["id"])
    present = pc.is_valid(position)
    taken = found.take(position)  # null rows where the id has no part row
    columns = {}
    for name, _ in PART_COLUMNS:
        if name in ("id", "url", "domain", "language"):
            # The official file and the domain table decide these, never the part.
            columns[name] = expected[name]
        else:
            columns[name] = taken[name]
    columns["source"] = pc.if_else(present, taken["source"], pa.scalar(MISSING))
    columns["title"] = pc.fill_null(taken["title"], "")
    columns["text"] = pc.fill_null(taken["text"], "")
    mismatches = pc.sum(
        pc.and_(present, pc.not_equal(pc.fill_null(taken["url"], ""), expected["url"]))
    ).as_py() or 0
    table = pa.table(columns).cast(schema)
    return table, mismatches


def _count(table, by_language: dict[str, Counter], by_domain: dict[str, Counter]) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc

    body = pc.subtract(pc.utf8_length(table["text"]), pc.utf8_length(table["title"]))
    quality = pc.if_else(
        pc.equal(table["source"], MISSING),
        pa.scalar("missing"),
        pc.if_else(pc.greater_equal(body, MIN_OK_CHARS), pa.scalar("ok"), pa.scalar("short")),
    )
    keys = pa.table({
        "language": table["language"], "domain": table["domain"],
        "source": table["source"], "quality": quality, "id": table["id"],
    })
    grouped = keys.group_by(["language", "domain", "source", "quality"]).aggregate(
        [("id", "count")]
    )
    for row in grouped.to_pylist():
        n = row["id_count"]
        for counter in (by_language[row["language"]], by_domain[row["domain"]]):
            counter["total"] += n
            counter[row["quality"]] += n
            counter[f"source:{row['source']}"] += n
        by_domain[row["domain"]]["lang:" + row["language"]] += n


def write_coverage(report: MergeReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted({k for c in report.by_domain.values() for k in c if k.startswith("source:")})
    fields = ["domain", "language", "total", "ok", "short", "missing", "ok_rate",
              *(s.split(":", 1)[1] for s in sources)]
    ranked = sorted(report.by_domain.items(), key=lambda item: -item[1]["total"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for domain, counts in ranked:
            language = next((k[5:] for k in counts if k.startswith("lang:")), "")
            writer.writerow([
                domain, language, counts["total"], counts["ok"], counts["short"],
                counts["missing"], f"{counts['ok'] / counts['total']:.4f}",
                *(counts[s] for s in sources),
            ])


def _write_info(path: Path, report: MergeReport, part_dirs, version: str) -> None:
    info = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": report.rows,
        "unique_ids": report.unique_ids,
        "input_parts": report.parts,
        "part_dirs": [str(p) for p in part_dirs],
        "duplicate_rows_dropped": report.duplicate_rows,
        "home_redirect_rows_dropped": report.home_redirects,
        "cleaned_docs": report.cleaned_docs,
        "cleaned_chars": report.cleaned_chars,
        "url_mismatches": report.url_mismatches,
        "by_language": {k: dict(v) for k, v in sorted(report.by_language.items())},
    }
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_corpus(corpus_dir: str | Path, ids: set[int] | None = None, languages=None,
                include_missing: bool = False):
    """Yield corpus rows as dicts, optionally restricted to ids and languages."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    for path in sorted(Path(corpus_dir).glob("part-*.parquet")):
        table = pq.read_table(path)
        mask = None
        if ids is not None:
            mask = pc.is_in(table["id"], value_set=pa.array(sorted(ids), pa.int64()))
        if languages:
            lang_mask = pc.is_in(table["language"], value_set=pa.array(list(languages)))
            mask = lang_mask if mask is None else pc.and_(mask, lang_mask)
        if not include_missing:
            keep = pc.not_equal(table["source"], MISSING)
            mask = keep if mask is None else pc.and_(mask, keep)
        if mask is not None:
            table = table.filter(mask)
        yield from table.to_pylist()


def export_jsonl(rows: Iterable[dict], path: str | Path, limit: int | None = None) -> int:
    """Write rows in the format ``vibiomir.io.load_documents`` reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if limit is not None and written >= limit:
                break
            record = {
                "id": row["id"],
                "url": row["url"],
                "title": row["title"],
                "language": row["language"],
                "text": row["text"],
                "domain": row["domain"],
                "source": row["source"],
                "final_url": row["final_url"],
                "published_at": row["published_at"],
                "text_sha256": row["text_sha256"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written
