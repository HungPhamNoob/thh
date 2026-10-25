"""PubMed title + abstract through NCBI E-utilities (efetch) instead of crawling pubmed HTML.

The PubMed website answers crawlers with a proof-of-work page; E-utilities is the documented
programmatic access. Without an API key NCBI allows 3 requests/s, with ``NCBI_API_KEY`` 10.
Work is split into fixed batches of the id-sorted PMID list, so a rerun skips every batch
whose parquet part already exists.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import logging
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vibiomir.ingest.domains import DomainPolicy
from vibiomir.ingest.extract import MIN_OK_CHARS, clean_text
from vibiomir.ingest.state import CrawlState, PageUpdate

log = logging.getLogger("vibiomir.ingest")

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_DOMAIN = "pubmed.ncbi.nlm.nih.gov"
_PMID_IN_URL = re.compile(r"/(\d+)/?(?:[?#].*)?$")
_MONTHS = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}


def pmid_from_url(url: str) -> str | None:
    match = _PMID_IN_URL.search(url)
    return match.group(1) if match else None


def _text(element: ET.Element | None) -> str:
    """All text inside an element, inline markup such as <i> or <sup> flattened."""
    if element is None:
        return ""
    return clean_text("".join(element.itertext())).replace("\n", " ")


def _published_at(article: ET.Element) -> str | None:
    date = article.find(".//Article/Journal/JournalIssue/PubDate")
    if date is None:
        date = article.find(".//PubDate")
    if date is None:
        return None
    year = date.findtext("Year")
    if not year:
        medline = date.findtext("MedlineDate") or ""
        found = re.search(r"\d{4}", medline)
        return found.group(0) if found else None
    month = (date.findtext("Month") or "").strip()
    month_number = int(month) if month.isdigit() else _MONTHS.get(month[:3].lower())
    if not month_number:
        return year
    day = (date.findtext("Day") or "").strip()
    return f"{year}-{month_number:02d}" + (f"-{int(day):02d}" if day.isdigit() else "")


def article_record(article: ET.Element) -> dict | None:
    """Title + abstract of one ``PubmedArticle`` or ``PubmedBookArticle``.

    ``text`` = title, then one paragraph per AbstractText; labelled sections keep their label
    ("BACKGROUND: ..."). OtherAbstract (translations) and copyright lines are left out.
    """
    pmid = article.findtext(".//PMID")
    if not pmid:
        return None
    title = _text(article.find(".//ArticleTitle")) or _text(article.find(".//BookTitle"))
    paragraphs = []
    for section in article.findall(".//Abstract/AbstractText"):
        body = _text(section)
        if not body:
            continue
        label = (section.get("Label") or "").strip()
        paragraphs.append(f"{label}: {body}" if label else body)
    text = "\n\n".join([title, *paragraphs]) if title else "\n\n".join(paragraphs)
    return {
        "pmid": pmid.strip(),
        "title": title,
        "text": text,
        "abstract_chars": sum(len(p) for p in paragraphs),
        "published_at": _published_at(article),
    }


def parse_efetch(data: bytes) -> dict[str, dict]:
    root = ET.fromstring(data)
    if root.tag != "PubmedArticleSet":
        raise ValueError(f"unexpected efetch root <{root.tag}>")
    records = {}
    for article in root:
        if article.tag in ("PubmedArticle", "PubmedBookArticle"):
            record = article_record(article)
            if record:
                records[record["pmid"]] = record
    return records


@dataclass(slots=True)
class PubMedConfig:
    out_dir: Path
    raw_dir: Path | None
    state_path: Path | None
    api_key: str | None = None
    email: str | None = None
    tool: str = "vibiomir-lab"
    batch_size: int = 200
    concurrency: int = 3
    timeout: float = 120.0
    retries: int = 5
    limit_batches: int | None = None


def make_batches(links: list[tuple[int, str]], size: int) -> list[list[tuple[int, str]]]:
    """Split ``(doc_id, pmid)`` pairs, sorted by doc id, into fixed-size batches."""
    pairs = sorted(
        (doc_id, pmid)
        for doc_id, url in links
        if url and (pmid := pmid_from_url(url)) is not None
    )
    return [pairs[i : i + size] for i in range(0, len(pairs), size)]


class PubMedFetcher:
    def __init__(self, config: PubMedConfig, transport=None) -> None:
        self.config = config
        self.transport = transport
        self.counts: Counter[str] = Counter()

    def run(self, links: list[tuple[int, str]]) -> Counter[str]:
        batches = make_batches(links, self.config.batch_size)
        asyncio.run(self._run(batches))
        return self.counts

    def _part_path(self, index: int) -> Path:
        return self.config.out_dir / f"part-pubmed-{index:05d}.parquet"

    async def _run(self, batches: list[list[tuple[int, str]]]) -> None:
        import httpx

        from vibiomir.ingest.fetch import DomainLimiter

        cfg = self.config
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.raw_dir is not None:
            cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        todo = [i for i in range(len(batches)) if not self._part_path(i).exists()]
        if cfg.limit_batches is not None:
            todo = todo[: cfg.limit_batches]
        log.info("pubmed: %d of %d batches to fetch", len(todo), len(batches))
        rps = 9.0 if cfg.api_key else 2.8  # stay just under NCBI's 10 / 3 requests per second
        limiter = DomainLimiter(DomainPolicy(rps=rps, max_rps=rps, concurrency=cfg.concurrency))
        state = CrawlState(cfg.state_path) if cfg.state_path else None
        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in todo:
            queue.put_nowait(index)
        done = 0

        async with httpx.AsyncClient(timeout=cfg.timeout, transport=self.transport) as client:

            async def worker() -> None:
                nonlocal done
                while not queue.empty():
                    index = queue.get_nowait()
                    await self._batch(client, limiter, state, index, batches[index])
                    done += 1
                    if done % 50 == 0 or done == len(todo):
                        log.info("pubmed: %d/%d batches %s", done, len(todo), dict(self.counts))

            try:
                await asyncio.gather(*(worker() for _ in range(cfg.concurrency)))
            finally:
                if state is not None:
                    state.close()

    async def _batch(self, client, limiter, state, index: int, batch) -> None:
        cfg = self.config
        params = {"db": "pubmed", "retmode": "xml", "tool": cfg.tool}
        if cfg.email:
            params["email"] = cfg.email
        if cfg.api_key:
            params["api_key"] = cfg.api_key
        params["id"] = ",".join(pmid for _, pmid in batch)
        data, records, error = b"", {}, None
        for attempt in range(cfg.retries):
            if attempt:
                await asyncio.sleep(2.0 * 2 ** (attempt - 1) * random.uniform(0.5, 1.5))
            await limiter.wait()
            try:
                response = await client.post(EFETCH_URL, data=params)
                if response.status_code in (429, 500, 502, 503, 504):
                    limiter.throttle(5.0)
                    error = f"http {response.status_code}"
                    continue
                response.raise_for_status()
                data = response.content
                records = parse_efetch(data)  # truncated or error XML raises → retry
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - retried, then reported per batch
                error = f"{type(exc).__name__}: {exc}"[:200]
        if error is not None:
            log.error("pubmed batch %d failed: %s", index, error)
            self.counts["failed_batches"] += 1
            return

        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows, updates = [], []
        for doc_id, pmid in batch:
            record = records.get(pmid)
            if record is None:
                updates.append(PageUpdate(doc_id, "gone", error="not in efetch response",
                                          fetched_at=fetched_at, source="pubmed_eutils"))
                self.counts["missing"] += 1
                continue
            status = "ok" if record["abstract_chars"] >= MIN_OK_CHARS else "short"
            text_sha = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            url = f"https://{PUBMED_DOMAIN}/{pmid}/"
            rows.append({
                "id": doc_id, "url": url, "final_url": url, "domain": PUBMED_DOMAIN,
                "language": "en", "title": record["title"], "text": record["text"],
                "source": "pubmed_eutils", "extractor": "pubmed_xml", "http_status": 200,
                "fetched_at": fetched_at, "published_at": record["published_at"],
                "encoding": "utf-8", "text_sha256": text_sha, "raw_sha256": None,
            })
            updates.append(PageUpdate(doc_id, status, http_status=200, final_url=url,
                                      fetched_at=fetched_at, text_sha256=text_sha,
                                      source="pubmed_eutils"))
            self.counts["with_abstract" if status == "ok" else "title_only"] += 1

        from vibiomir.ingest.fetch import write_part

        if cfg.raw_dir is not None:
            with gzip.open(cfg.raw_dir / f"batch-{index:05d}.xml.gz", "wb") as handle:
                handle.write(data)
        # Written even when empty: the part file marks the batch as done for reruns.
        write_part(rows, self._part_path(index))
        if state is not None:
            state.update(updates)
