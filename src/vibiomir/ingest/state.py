"""SQLite crawl state so that an interrupted crawl resumes without refetching finished pages."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from vibiomir.ingest.domains import domain_language, url_domain

# pending: not fetched yet · ok: text extracted · short: text < min chars · failed: transient
# error, retried on the next run while attempts remain · blocked: anti-bot page or unreachable
# host · gone: 404/410 or soft 404 · disallowed: robots.txt forbids the URL · unsupported: not
# HTML (pdf, pptx, images, ...)
STATUSES = ("pending", "ok", "short", "failed", "blocked", "gone", "disallowed", "unsupported")
FINAL_STATUSES = ("ok", "short", "gone", "disallowed", "unsupported")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    lang TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    http_status INTEGER,
    final_url TEXT,
    error TEXT,
    fetched_at TEXT,
    text_sha256 TEXT,
    source TEXT
);
CREATE INDEX IF NOT EXISTS pages_domain_status ON pages(domain, status, id);
"""


@dataclass(slots=True)
class PageUpdate:
    id: int
    status: str
    http_status: int | None = None
    final_url: str | None = None
    error: str | None = None
    fetched_at: str | None = None
    text_sha256: str | None = None
    source: str = "crawl"


class CrawlState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The timeout lets several crawl processes share one state file.
        self.conn = sqlite3.connect(self.path, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS excluded (id INTEGER PRIMARY KEY)")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> CrawlState:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

    def seed(self, links: Iterable[tuple[int, str]]) -> int:
        """Insert unseen links as ``pending``; existing rows keep their status."""
        rows = []
        for doc_id, url in links:
            host = url_domain(url)
            rows.append((doc_id, url, host, domain_language(host)))
        before = self.count()
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO pages(id, url, domain, lang) VALUES (?, ?, ?, ?)", rows
            )
        return self.count() - before

    def exclude(self, ids: Iterable[int]) -> None:
        """Hide ids for this connection only, e.g. pages already covered by a bootstrap source."""
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO excluded(id) VALUES (?)", ((int(i),) for i in ids)
            )

    def domains_todo(self, max_attempts: int, skip: Iterable[str] = ()) -> dict[str, int]:
        skip = set(skip)
        rows = self.conn.execute(
            """SELECT domain, COUNT(*) FROM pages
               WHERE (status = 'pending' OR (status IN ('failed', 'blocked') AND attempts < ?))
                 AND id NOT IN (SELECT id FROM excluded)
               GROUP BY domain ORDER BY COUNT(*) DESC""",
            (max_attempts,),
        ).fetchall()
        return {domain: n for domain, n in rows if domain not in skip}

    def iter_todo(
        self, domain: str, max_attempts: int, batch_size: int = 1000, limit: int | None = None
    ) -> Iterator[tuple[int, str]]:
        """Yield ``(id, url)`` of work for one domain in id order, reading in small batches."""
        last_id = -1
        emitted = 0
        while limit is None or emitted < limit:
            size = batch_size if limit is None else min(batch_size, limit - emitted)
            rows = self.conn.execute(
                """SELECT id, url FROM pages
                   WHERE domain = ? AND id > ?
                     AND (status = 'pending' OR (status IN ('failed', 'blocked') AND attempts < ?))
                     AND id NOT IN (SELECT id FROM excluded)
                   ORDER BY id LIMIT ?""",
                (domain, last_id, max_attempts, size),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                yield row
            emitted += len(rows)
            last_id = rows[-1][0]

    def update(self, updates: Iterable[PageUpdate]) -> None:
        with self.conn:
            self.conn.executemany(
                """UPDATE pages SET status = ?, attempts = attempts + 1, http_status = ?,
                       final_url = ?, error = ?, fetched_at = ?, text_sha256 = ?, source = ?
                   WHERE id = ?""",
                [
                    (
                        u.status,
                        u.http_status,
                        u.final_url,
                        u.error,
                        u.fetched_at,
                        u.text_sha256,
                        u.source,
                        u.id,
                    )
                    for u in updates
                ],
            )

    def reset(self, domains: Iterable[str], statuses: Iterable[str]) -> int:
        """Set matching pages back to ``pending`` with zero attempts, e.g. after an extractor fix.

        A domain also matches its subdomains (``39.net`` matches ``care.39.net``). Rows already
        written to parquet parts stay there; the merge keeps the newest ``fetched_at`` per id.
        """
        statuses = list(statuses)
        marks = ",".join("?" * len(statuses))
        changed = 0
        with self.conn:
            for domain in domains:
                cursor = self.conn.execute(
                    f"""UPDATE pages SET status = 'pending', attempts = 0
                        WHERE (domain = ? OR domain LIKE '%.' || ?) AND status IN ({marks})""",
                    (domain, domain, *statuses),
                )
                changed += cursor.rowcount
        return changed

    def summary(self) -> list[tuple[str, str, str, int]]:
        """Return ``(lang, domain, status, count)`` rows."""
        return self.conn.execute(
            "SELECT lang, domain, status, COUNT(*) FROM pages GROUP BY lang, domain, status"
        ).fetchall()
