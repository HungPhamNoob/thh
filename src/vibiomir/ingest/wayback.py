"""Wayback Machine fallback for pages the crawl could not fetch (D7).

Pages that stayed ``pending``/``failed``/``blocked`` (or ``gone``: a dead page may have been
archived while alive) in the crawl state are copied into a separate state file, then fetched
as ``https://web.archive.org/web/<timestamp>id_/<url>``. ``id_`` returns the archived bytes
without the Wayback toolbar, so the normal per-site extractors apply. The crawl state is left
untouched; rows go to their own part folder with ``source=wayback`` and the merge (D6) ranks
them below a direct crawl.

In the Wayback state ``gone`` means "no capture" (or a captured 404) and ``blocked`` means the
capture itself is a bot-check page.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from vibiomir.ingest.state import CrawlState

ARCHIVE_HOST = "web.archive.org"
# ``2026`` asks for the capture closest to 2026; Wayback redirects to the exact timestamp.
URL_TEMPLATE = "https://web.archive.org/web/{timestamp}id_/{{url}}"
SOURCE = "wayback"
FALLBACK_STATUSES = ("pending", "failed", "blocked", "gone")


def url_template(timestamp: str = "2026") -> str:
    return URL_TEMPLATE.format(timestamp=timestamp)


def seed_from_crawl(
    crawl_state: CrawlState,
    fallback_state: CrawlState,
    domains: Iterable[str] | None = None,
    skip_domains: Iterable[str] = (),
    statuses: Iterable[str] = FALLBACK_STATUSES,
) -> int:
    """Copy unfinished crawl pages into ``fallback_state``; return how many were new.

    Pages already in the fallback state keep their status, so reseeding after a later crawl
    run only adds pages that have failed since. Pending fallback pages that the crawl has
    fetched in the meantime are dropped.
    """
    statuses = list(statuses)
    marks = ",".join("?" * len(statuses))
    rows = crawl_state.conn.execute(
        f"SELECT id, url, domain FROM pages WHERE status IN ({marks}) ORDER BY id", statuses
    ).fetchall()
    wanted = set(domains) if domains else None
    skip = set(skip_domains)
    links = [
        (doc_id, url)
        for doc_id, url, domain in rows
        if domain not in skip and (wanted is None or domain in wanted)
    ]
    done = crawl_state.conn.execute(
        "SELECT id FROM pages WHERE status IN ('ok', 'short')"
    ).fetchall()
    with fallback_state.conn:
        fallback_state.conn.executemany(
            "DELETE FROM pages WHERE id = ? AND status = 'pending'", done
        )
    return fallback_state.seed(links)


def default_state_path(crawl_state_path: str | Path) -> Path:
    return Path(crawl_state_path).with_name("wayback_state.sqlite")
