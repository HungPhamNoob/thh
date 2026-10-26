"""Polite asynchronous crawler: per-domain rate limits, retries, robots.txt, resumable state.

Throughput comes from crawling many domains at once, not from pushing one domain hard. Each
domain gets its own request-rate limiter (raised slowly while the site answers, halved on
429/503) and a circuit breaker that stops the domain for this run when it only returns
anti-bot pages or never connects. Stopped pages stay retryable for a later run.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import logging
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.robotparser import RobotFileParser

from vibiomir.ingest.domains import (
    DEFAULT_SKIP_DOMAINS,
    DomainPolicy,
    domain_language,
    domain_policy,
    rate_group,
)
from vibiomir.ingest.extract import MIN_OK_CHARS, extract_page
from vibiomir.ingest.state import FINAL_STATUSES, CrawlState, PageUpdate

log = logging.getLogger("vibiomir.ingest")

ROBOTS_TOKEN = "vibiomir-research-crawler"
# A plain bot token: on 2026-10-25 youmed.vn answered 403 to "Mozilla/5.0 (compatible; ...)".
DEFAULT_USER_AGENT = (
    f"{ROBOTS_TOKEN}/0.1 (+https://github.com/HungPhamNoob/thh; R2AI2026 ViBioMIR research)"
)

# Bodies that mean "a bot check, not the article", matched on small pages only. Seen on
# 2026-10-25: Cloudflare block pages, a JS cookie-and-reload page, PubMed's proof-of-work page
# and plain "Access Denied" pages. Generic words such as "captcha" are left out on purpose:
# ordinary pages embed reCAPTCHA scripts.
CHALLENGE_PATTERNS = tuple(
    re.compile(pattern, re.I | re.S)
    for pattern in (
        rb"attention required! \| cloudflare",
        rb"cf-error-details|cf-chl-|/cdn-cgi/challenge-platform",
        rb"<title>just a moment\.\.\.</title>",
        rb"<title>access denied</title>",
        rb"blocked by our security system",
        rb"document\.cookie\s*=.{0,200}location\.reload",
        rb"cookies must be enabled",
        rb"const challengeid\s*=",
        "滑动拼图验证".encode(),  # 39.net slider captcha
    )
)
CHALLENGE_URL = re.compile(r"/(?:verify|captcha|challenge)(?:\.html?|\.php|/)?$", re.I)
# The same short text on this many pages of one domain is a block page, not content.
REPEATED_SHORT_LIMIT = 5
CHALLENGE_MAX_BYTES = 20_000
HTML_TYPES = ("text/html", "application/xhtml", "text/plain")
TRANSIENT_STATUSES = {408, 425, 500, 502, 504, 520, 521, 522, 523, 524}
THROTTLE_STATUSES = {429, 503}


@dataclass(slots=True)
class CrawlConfig:
    state_path: Path
    out_dir: Path
    samples_dir: Path | None = None
    user_agent: str = DEFAULT_USER_AGENT
    domains: list[str] | None = None
    skip_domains: tuple[str, ...] = DEFAULT_SKIP_DOMAINS
    limit_per_domain: int | None = None
    max_attempts: int = 3
    retries: int = 2
    global_concurrency: int = 128
    timeout: float = 20.0
    max_bytes: int = 5_000_000
    flush_every: int = 2000
    keep_samples: int = 50
    max_problem_samples: int = 200
    block_threshold: int = 10
    cooldown: float = 900.0
    max_pauses: int = 2
    extract_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))
    default_policy: DomainPolicy = field(default_factory=DomainPolicy)
    respect_robots: bool = True
    log_every: float = 30.0
    backoff_base: float = 2.0
    # Fetch through a mirror such as the Wayback Machine: ``{url}`` is the official URL, which
    # rows keep in ``url``; ``final_url`` records the mirror copy actually read.
    url_template: str | None = None
    source: str = "crawl"
    # One rate limit for every domain, because with a mirror all requests go to one host.
    shared_group: str | None = None


@dataclass(slots=True)
class FetchResult:
    status: int | None
    final_url: str | None
    content_type: str
    body: bytes
    error: str | None = None
    retry_after: str | None = None


def classify_response(status: int, body: bytes, final_url: str = "") -> str | None:
    """Return a terminal status for responses that carry no article, else ``None``."""
    if final_url and CHALLENGE_URL.search(final_url.split("?")[0].split("#")[0]):
        return "blocked"
    small = len(body) <= CHALLENGE_MAX_BYTES
    if small and any(pattern.search(body) for pattern in CHALLENGE_PATTERNS):
        return "blocked"
    if status in (404, 410):
        return "gone"
    if status in (401, 403, 451):
        return "blocked"
    if 200 <= status < 300:
        return None
    return "failed"


class DomainLimiter:
    """Spaces requests to one domain at ``rps``; additive increase, multiplicative decrease."""

    def __init__(self, policy: DomainPolicy) -> None:
        self.rps = policy.rps
        self.max_rps = policy.max_rps
        self._next = 0.0
        self._lock = asyncio.Lock()
        self._streak = 0

    def cap(self, max_rps: float) -> None:
        self.max_rps = min(self.max_rps, max_rps)
        self.rps = min(self.rps, self.max_rps)

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            start = max(now, self._next)
            self._next = start + 1.0 / self.rps
        if start > now:
            await asyncio.sleep(start - now)

    def success(self) -> None:
        self._streak += 1
        if self._streak >= 50:
            self._streak = 0
            self.rps = min(self.max_rps, self.rps + 0.5)

    def throttle(self, retry_after: float = 0.0) -> None:
        self._streak = 0
        self.rps = max(0.1, self.rps / 2)
        if retry_after > 0:
            self._next = max(self._next, asyncio.get_running_loop().time() + retry_after)


@dataclass(slots=True)
class DomainStats:
    remaining: int
    done: int = 0
    successes: int = 0
    consecutive_bad: int = 0
    stopped: str | None = None
    samples: int = 0
    problem_samples: int = 0
    group: str = ""
    short_texts: Counter = field(default_factory=Counter)
    paused_until: float = 0.0
    pauses: int = 0


@dataclass(slots=True)
class RateGroup:
    """Limiter and connection cap shared by all hosts of one registrable domain."""

    policy: DomainPolicy
    limiter: DomainLimiter
    semaphore: asyncio.Semaphore


class ResultSink:
    """Buffers rows and state updates; each flush writes one parquet part, then commits state."""

    def __init__(self, state: CrawlState, out_dir: Path, flush_every: int) -> None:
        self.state = state
        self.out_dir = out_dir
        self.flush_every = flush_every
        self.rows: list[dict] = []
        self.updates: list[PageUpdate] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        # Run-specific names let several crawl processes (one per domain shard) share a folder.
        self.run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{os.getpid()}"
        self.next_part = 0

    def add(self, update: PageUpdate, row: dict | None = None) -> None:
        self.updates.append(update)
        if row is not None:
            self.rows.append(row)
        if len(self.updates) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            name = f"part-{self.run_id}-{self.next_part:05d}.parquet"
            write_part(self.rows, self.out_dir / name)
            self.next_part += 1
        # State is committed after the part exists: a crash in between only duplicates rows,
        # which the merge step (D6) removes by id.
        if self.updates:
            self.state.update(self.updates)
        self.rows, self.updates = [], []


PART_COLUMNS = (
    ("id", "int64"),
    ("url", "string"),
    ("final_url", "string"),
    ("domain", "string"),
    ("language", "string"),
    ("title", "string"),
    ("text", "string"),
    ("source", "string"),
    ("extractor", "string"),
    ("http_status", "int32"),
    ("fetched_at", "string"),
    ("published_at", "string"),
    ("encoding", "string"),
    ("text_sha256", "string"),
    ("raw_sha256", "string"),
)


def write_part(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([(name, getattr(pa, kind)()) for name, kind in PART_COLUMNS])
    table = pa.Table.from_pylist([{n: row.get(n) for n, _ in PART_COLUMNS} for row in rows],
                                 schema=schema)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _retry_after(value: str | None) -> float:
    try:
        return min(300.0, float(value)) if value else 0.0
    except ValueError:
        return 0.0


class Crawler:
    def __init__(self, config: CrawlConfig, transport=None, executor: Executor | None = None):
        self.config = config
        self.transport = transport
        self.executor = executor
        self._own_executor: ProcessPoolExecutor | None = None
        self.stats: dict[str, DomainStats] = {}
        self.groups: dict[str, RateGroup] = {}
        self.counts: Counter[str] = Counter()
        self.robots: dict[str, RobotFileParser | None] = {}
        self.started = time.monotonic()

    # ------------------------------------------------------------------ public
    def run(self, links: list[tuple[int, str]] | None = None, exclude_ids=()) -> Counter[str]:
        state = CrawlState(self.config.state_path)
        try:
            if links:
                added = state.seed(links)
                log.info("seeded %d new pages (%d total)", added, state.count())
            if exclude_ids:
                state.exclude(exclude_ids)
            sink = ResultSink(state, self.config.out_dir, self.config.flush_every)
            try:
                asyncio.run(self._run(state, sink))
            finally:
                sink.flush()
        finally:
            state.close()
        return self.counts

    # ----------------------------------------------------------------- crawling
    async def _run(self, state: CrawlState, sink: ResultSink) -> None:
        import httpx

        cfg = self.config
        todo = state.domains_todo(cfg.max_attempts, skip=cfg.skip_domains)
        if cfg.domains:
            todo = {d: n for d, n in todo.items() if d in set(cfg.domains)}
        if cfg.limit_per_domain:
            todo = {d: min(n, cfg.limit_per_domain) for d, n in todo.items()}
        log.info("%d pages to fetch across %d domains", sum(todo.values()), len(todo))
        if not todo:
            return

        if self.executor is None and cfg.extract_workers > 0:
            self._own_executor = self.executor = ProcessPoolExecutor(cfg.extract_workers)
        semaphore = asyncio.Semaphore(cfg.global_concurrency)
        headers = {
            "User-Agent": cfg.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
            "Accept-Language": "vi,zh-CN;q=0.8,en;q=0.5",
        }
        client = httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(cfg.timeout, connect=min(10.0, cfg.timeout)),
            limits=httpx.Limits(
                max_connections=cfg.global_concurrency,
                max_keepalive_connections=cfg.global_concurrency,
            ),
            transport=self.transport,
        )
        reporter = asyncio.create_task(self._report())
        try:
            async with client:
                await asyncio.gather(
                    *(self._crawl_domain(client, semaphore, state, sink, d, n)
                      for d, n in todo.items())
                )
        finally:
            reporter.cancel()
            if self._own_executor is not None:
                self._own_executor.shutdown(cancel_futures=True)
                self._own_executor = self.executor = None
            self._log_progress(final=True)

    def _group_name(self, domain: str) -> str:
        return self.config.shared_group or rate_group(domain)

    def _group(self, domain: str) -> RateGroup:
        name = self._group_name(domain)
        if name not in self.groups:
            policy = self.config.default_policy
            if not self.config.shared_group:
                policy = domain_policy(name, policy)
            self.groups[name] = RateGroup(
                policy, DomainLimiter(policy), asyncio.Semaphore(policy.concurrency)
            )
        return self.groups[name]

    async def _crawl_domain(self, client, semaphore, state, sink, domain: str, n: int) -> None:
        cfg = self.config
        group = self._group(domain)
        policy = group.policy
        stats = self.stats[domain] = DomainStats(remaining=n, group=self._group_name(domain))
        if cfg.samples_dir is not None:
            stats.samples = len(list((cfg.samples_dir / domain).glob("*.html.gz")))
        robots = await self._robots(client, domain) if cfg.respect_robots else None
        # The site's own Crawl-delay protects the site; a mirror is limited by shared_group.
        if robots is not None and robots.crawl_delay(ROBOTS_TOKEN) and not cfg.url_template:
            group.limiter.cap(1.0 / float(robots.crawl_delay(ROBOTS_TOKEN)))

        queue: asyncio.Queue = asyncio.Queue(maxsize=policy.concurrency * 4)

        async def feed() -> None:
            for item in state.iter_todo(domain, cfg.max_attempts, limit=cfg.limit_per_domain):
                if stats.stopped:
                    break
                await queue.put(item)
            for _ in range(policy.concurrency):
                await queue.put(None)

        async def work() -> None:
            while (item := await queue.get()) is not None:
                if stats.stopped:
                    continue
                pause = stats.paused_until - time.monotonic()
                if pause > 0:
                    await asyncio.sleep(pause)
                await self._process(
                    client, semaphore, group, sink, robots, domain, stats, *item
                )

        await asyncio.gather(feed(), *(work() for _ in range(policy.concurrency)))
        if stats.stopped:
            log.warning("stopped %s after %d pages: %s", domain, stats.done, stats.stopped)

    async def _process(
        self, client, semaphore, group, sink, robots, domain, stats, doc_id, url
    ) -> None:
        if robots is not None and not robots.can_fetch(ROBOTS_TOKEN, url):
            self._finish(sink, stats, PageUpdate(doc_id, "disallowed", error="robots.txt",
                                                 source=self.config.source))
            return
        cfg = self.config
        fetch_url = cfg.url_template.format(url=url) if cfg.url_template else url
        result = await self._fetch_with_retries(client, semaphore, group, fetch_url)
        fetched_at = _utc_now()
        base = dict(http_status=result.status, final_url=result.final_url, fetched_at=fetched_at,
                    source=cfg.source)

        if result.error is not None:
            self._finish(sink, stats, PageUpdate(doc_id, "failed", error=result.error, **base))
            return
        verdict = classify_response(result.status or 0, result.body, result.final_url or "")
        if verdict is not None:
            self._save_sample(domain, doc_id, result.body, stats, problem=True)
            self._finish(sink, stats, PageUpdate(doc_id, verdict, error=f"http {result.status}",
                                                 **base))
            return
        mime = result.content_type.split(";")[0].strip().lower()
        if mime and not mime.startswith(HTML_TYPES):
            update = PageUpdate(doc_id, "unsupported", error=f"type {mime}", **base)
            self._finish(sink, stats, update)
            return

        try:
            extracted = await self._extract(result.body, result.content_type, domain)
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop the crawl
            self._save_sample(domain, doc_id, result.body, stats, problem=True)
            error = f"extract {type(exc).__name__}: {exc}"[:300]
            self._finish(sink, stats, PageUpdate(doc_id, "failed", error=error, **base))
            return

        if extracted.soft_404:
            self._finish(sink, stats, PageUpdate(doc_id, "gone", error="soft 404", **base))
            return
        body_chars = len(extracted.text) - len(extracted.title)
        status = "ok" if body_chars >= MIN_OK_CHARS else "short"
        self._save_sample(domain, doc_id, result.body, stats, problem=status != "ok")
        text_sha = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
        if status == "short":
            stats.short_texts[text_sha] += 1
            if stats.short_texts[text_sha] >= REPEATED_SHORT_LIMIT:
                error = f"repeated short page: {extracted.text[:60]!r}"
                self._finish(sink, stats, PageUpdate(doc_id, "blocked", error=error, **base))
                return
        row = {
            "id": doc_id,
            "url": url,
            "final_url": result.final_url,
            "domain": domain,
            "language": domain_language(domain),
            "title": extracted.title,
            "text": extracted.text,
            "source": cfg.source,
            "extractor": extracted.extractor,
            "http_status": result.status,
            "fetched_at": fetched_at,
            "published_at": extracted.published_at,
            "encoding": extracted.encoding,
            "text_sha256": text_sha,
            "raw_sha256": hashlib.sha256(result.body).hexdigest(),
        }
        update = PageUpdate(doc_id, status, text_sha256=text_sha, **base)
        self._finish(sink, stats, update, row)

    async def _extract(self, body: bytes, content_type: str, domain: str):
        if self.executor is None:
            return extract_page(body, content_type, domain)
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self.executor, extract_page, body, content_type, domain
            )
        except BrokenProcessPool:
            # A crashed worker breaks the whole pool; replace it so later pages still extract.
            if self._own_executor is self.executor:
                log.error("extraction pool broke; starting a new one")
                self._own_executor.shutdown(wait=False, cancel_futures=True)
                self._own_executor = self.executor = ProcessPoolExecutor(
                    max_workers=self.config.extract_workers
                )
            return extract_page(body, content_type, domain)

    def _finish(self, sink, stats: DomainStats, update: PageUpdate, row: dict | None = None):
        sink.add(update, row)
        self.counts[update.status] += 1
        stats.done += 1
        stats.remaining = max(0, stats.remaining - 1)
        if update.status in FINAL_STATUSES:
            stats.successes += 1
            stats.consecutive_bad = 0
            return
        now = time.monotonic()
        if now < stats.paused_until:
            return  # answers to requests already in flight when the pause started
        cfg = self.config
        if cfg.url_template and update.status == "blocked":
            return  # an archived bot-check page says nothing about the mirror or other pages
        stats.consecutive_bad += 1
        if stats.consecutive_bad < cfg.block_threshold:
            return
        reason = f"{stats.consecutive_bad} consecutive {update.status} ({update.error})"
        if stats.successes and stats.pauses < cfg.max_pauses:
            # The site served us before: back off, halve the group rate, then try again.
            stats.pauses += 1
            stats.consecutive_bad = 0
            stats.paused_until = now + cfg.cooldown
            self.groups[stats.group].limiter.throttle()
            log.warning("pausing %s for %.0fs: %s", stats.group, cfg.cooldown, reason)
            return
        stats.stopped = reason

    async def _fetch_with_retries(self, client, semaphore, group: RateGroup, url) -> FetchResult:
        cfg = self.config
        limiter = group.limiter
        result = FetchResult(None, None, "", b"", "not fetched")
        for attempt in range(cfg.retries + 1):
            if attempt:
                delay = cfg.backoff_base * 2 ** (attempt - 1)
                await asyncio.sleep(delay * random.uniform(0.5, 1.5))
            async with group.semaphore:
                await limiter.wait()
                async with semaphore:
                    result = await self._fetch(client, url)
            if result.error is None and result.status in THROTTLE_STATUSES:
                limiter.throttle(_retry_after(result.retry_after))
                continue
            if result.error is not None or result.status in TRANSIENT_STATUSES:
                continue
            limiter.success()
            return result
        if result.error is None and result.status in THROTTLE_STATUSES | TRANSIENT_STATUSES:
            result.error = f"http {result.status} after {cfg.retries + 1} tries"
        return result

    async def _fetch(self, client, url: str) -> FetchResult:
        import httpx

        try:
            async with client.stream("GET", url) as response:
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > self.config.max_bytes:
                        break
                return FetchResult(
                    status=response.status_code,
                    final_url=str(response.url),
                    content_type=response.headers.get("content-type", ""),
                    body=b"".join(chunks),
                    retry_after=response.headers.get("retry-after"),
                )
        except httpx.HTTPError as exc:
            return FetchResult(None, None, "", b"", type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - e.g. invalid URL; recorded per page
            return FetchResult(None, None, "", b"", f"{type(exc).__name__}: {exc}"[:200])

    async def _robots(self, client, domain: str) -> RobotFileParser | None:
        if domain in self.robots:
            return self.robots[domain]
        parser: RobotFileParser | None = RobotFileParser()
        try:
            response = await client.get(f"https://{domain}/robots.txt")
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser = None  # 4xx/5xx: no usable robots.txt, crawl with rate limits only
        except Exception:  # noqa: BLE001 - unreachable robots.txt must not stop the domain
            parser = None
        self.robots[domain] = parser
        return parser

    def _save_sample(self, domain, doc_id, body: bytes, stats: DomainStats, problem: bool):
        cfg = self.config
        if cfg.samples_dir is None or not body:
            return
        if problem:
            if stats.problem_samples >= cfg.max_problem_samples:
                return
            stats.problem_samples += 1
        else:
            if stats.samples >= cfg.keep_samples:
                return
            stats.samples += 1
        folder = cfg.samples_dir / domain
        folder.mkdir(parents=True, exist_ok=True)
        with gzip.open(folder / f"{doc_id}.html.gz", "wb") as handle:
            handle.write(body)

    # ---------------------------------------------------------------- progress
    async def _report(self) -> None:
        while True:
            await asyncio.sleep(self.config.log_every)
            self._log_progress()

    def _log_progress(self, final: bool = False) -> None:
        elapsed = max(1e-6, time.monotonic() - self.started)
        done = sum(self.counts.values())
        remaining = sum(s.remaining for s in self.stats.values() if not s.stopped)
        rate = done / elapsed
        active = [d for d, s in self.stats.items() if s.remaining and not s.stopped]
        left_by_group: Counter[str] = Counter()
        for domain in active:
            left_by_group[self.stats[domain].group] += self.stats[domain].remaining
        slowest = sorted(
            ((left / self.groups[g].limiter.rps, g) for g, left in left_by_group.items()),
            reverse=True,
        )[:3]
        eta = f"{slowest[0][0] / 3600:.1f}h" if slowest else "0h"
        log.info(
            "%s %d done (%.1f/s) %s | %d left in %d domains | bottleneck ETA %s: %s",
            "final:" if final else "progress:",
            done,
            rate,
            dict(self.counts),
            remaining,
            len(active),
            eta,
            ", ".join(f"{d} {sec / 3600:.1f}h" for sec, d in slowest),
        )

