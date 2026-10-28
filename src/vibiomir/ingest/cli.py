"""``vibiomir ingest`` subcommands."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_LINKS = "data/raw/official/links_corpus_vi_en_zh.parquet"
DEFAULT_STATE = "data/interim/crawl_state.sqlite"


def command_download_official(args: argparse.Namespace) -> int:
    from vibiomir.ingest.official import download_official

    for path in download_official(args.dest, revision=args.revision):
        print(path)
    return 0


def command_manifest(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vibiomir.ingest.official import manifest_rows, read_links

    rows = manifest_rows(read_links(args.links))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd")
    by_language = Counter(row["language"] for row in rows)
    domains = {row["domain"] for row in rows}
    print(f"Wrote {len(rows)} rows, {len(domains)} domains to {output}: {dict(by_language)}")
    return 0


def command_audit_html(args: argparse.Namespace) -> int:
    from vibiomir.ingest.audit import audit_html_samples

    _, summary = audit_html_samples(args.samples, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote HTML audit to {args.output}")
    return 0


def command_crawl(args: argparse.Namespace) -> int:
    from vibiomir.ingest.domains import DEFAULT_SKIP_DOMAINS, DomainPolicy
    from vibiomir.ingest.fetch import DEFAULT_USER_AGENT, CrawlConfig, Crawler
    from vibiomir.ingest.official import read_id_file, read_links, sample_links_by_domain

    _setup_logging(args.log_file)
    user_agent = DEFAULT_USER_AGENT
    contact = os.environ.get("VIBIOMIR_CRAWLER_CONTACT")
    if contact:
        user_agent = user_agent[:-1] + f"; contact: {contact})"
    config = CrawlConfig(
        state_path=Path(args.state),
        out_dir=Path(args.out),
        samples_dir=Path(args.samples) if args.samples else None,
        user_agent=user_agent,
        domains=args.domain or None,
        skip_domains=tuple(args.skip_domain) if args.skip_domain else DEFAULT_SKIP_DOMAINS,
        limit_per_domain=args.limit_per_domain,
        max_attempts=args.max_attempts,
        global_concurrency=args.concurrency,
        timeout=args.timeout,
        extract_workers=args.workers,
        default_policy=DomainPolicy(
            rps=args.rps, max_rps=args.max_rps, concurrency=args.domain_concurrency
        ),
        respect_robots=not args.ignore_robots,
    )
    links = read_links(args.links) if args.links and Path(args.links).exists() else None
    if links is not None and args.sample_per_domain:
        links = sample_links_by_domain(links, args.sample_per_domain, args.sample_seed)
    excluded: set[int] = set()
    for path in args.exclude_ids or []:
        excluded |= read_id_file(path)
    counts = Crawler(config).run(links, exclude_ids=excluded)
    print(f"Crawl finished: {dict(counts)}")
    return 0


def _file_handler(path: str | None) -> list[logging.Handler]:
    if not path:
        return []
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return [logging.FileHandler(path, encoding="utf-8")]


def _setup_logging(log_file: str | None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), *_file_handler(log_file)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def command_pubmed(args: argparse.Namespace) -> int:
    from vibiomir.ingest.official import read_links
    from vibiomir.ingest.pubmed import PUBMED_DOMAIN, PubMedConfig, PubMedFetcher

    _setup_logging(args.log_file)
    links = [(i, u) for i, u in read_links(args.links) if PUBMED_DOMAIN in u]
    config = PubMedConfig(
        out_dir=Path(args.out),
        raw_dir=Path(args.raw) if args.raw else None,
        state_path=Path(args.state) if args.state else None,
        api_key=os.environ.get("NCBI_API_KEY"),
        email=os.environ.get("NCBI_EMAIL"),
        batch_size=args.batch_size,
        limit_batches=args.limit_batches,
    )
    counts = PubMedFetcher(config).run(links)
    print(f"PubMed finished: {dict(counts)}")
    return 1 if counts["failed_batches"] else 0


def command_bootstrap(args: argparse.Namespace) -> int:
    from vibiomir.ingest.bootstrap import convert_dataset
    from vibiomir.ingest.official import read_links

    _setup_logging(args.log_file)
    official = dict(read_links(args.links))
    out = Path(args.out or f"data/interim/parts/hf_{args.dataset}")
    counts = convert_dataset(args.dataset, args.input, official, out)
    print(f"Bootstrap {args.dataset} finished: {dict(counts)}")
    return 0


def command_wayback(args: argparse.Namespace) -> int:
    from vibiomir.ingest.domains import DomainPolicy
    from vibiomir.ingest.fetch import DEFAULT_USER_AGENT, CrawlConfig, Crawler
    from vibiomir.ingest.state import CrawlState
    from vibiomir.ingest.wayback import (
        ARCHIVE_HOST,
        SOURCE,
        default_state_path,
        seed_from_crawl,
        url_template,
    )

    _setup_logging(args.log_file)
    fallback_path = Path(args.wayback_state) if args.wayback_state else default_state_path(
        args.state
    )
    skip = tuple(args.skip_domain or ("pubmed.ncbi.nlm.nih.gov", "vov.vn"))
    with CrawlState(args.state) as crawl_state, CrawlState(fallback_path) as fallback:
        added = seed_from_crawl(crawl_state, fallback, args.domain, skip)
    logging.getLogger("vibiomir.ingest").info("seeded %d new pages into %s", added, fallback_path)
    user_agent = DEFAULT_USER_AGENT
    contact = os.environ.get("VIBIOMIR_CRAWLER_CONTACT")
    if contact:
        user_agent = user_agent[:-1] + f"; contact: {contact})"
    config = CrawlConfig(
        state_path=fallback_path,
        out_dir=Path(args.out),
        samples_dir=Path(args.samples) if args.samples else None,
        user_agent=user_agent,
        domains=args.domain or None,
        skip_domains=skip,
        limit_per_domain=args.limit_per_domain,
        max_attempts=args.max_attempts,
        global_concurrency=args.concurrency,
        timeout=args.timeout,
        extract_workers=args.workers,
        default_policy=DomainPolicy(
            rps=args.rps, max_rps=args.max_rps, concurrency=args.concurrency
        ),
        url_template=url_template(args.timestamp),
        source=SOURCE,
        shared_group=ARCHIVE_HOST,
    )
    counts = Crawler(config).run()
    print(f"Wayback finished: {dict(counts)}")
    return 0


def command_merge(args: argparse.Namespace) -> int:
    from vibiomir.ingest.merge import merge_corpus

    _setup_logging(args.log_file)
    report = merge_corpus(
        args.manifest, args.parts, args.out, coverage_path=args.coverage, version=args.version,
        clean=not args.no_clean,
    )
    print(f"Merged {report.rows} rows ({report.unique_ids} unique ids) from {report.parts} parts; "
          f"dropped {report.duplicate_rows} duplicate rows and {report.home_redirects} home-page "
          f"redirects, {report.url_mismatches} url mismatches; cleaned {report.cleaned_docs} "
          f"docs (-{report.cleaned_chars} chars)")
    for language, counts in sorted(report.by_language.items()):
        print(f"  {language}: total {counts['total']} ok {counts['ok']} short {counts['short']} "
              f"missing {counts['missing']}")
    return 0


def command_export_jsonl(args: argparse.Namespace) -> int:
    from vibiomir.ingest.merge import export_jsonl, iter_corpus
    from vibiomir.ingest.official import read_id_file

    ids = read_id_file(args.ids) if args.ids else None
    rows = iter_corpus(args.corpus, ids=ids, languages=args.language,
                       include_missing=args.include_missing)
    written = export_jsonl(rows, args.output, limit=args.limit)
    print(f"Wrote {written} documents to {args.output}")
    return 0


def command_reset(args: argparse.Namespace) -> int:
    from vibiomir.ingest.state import CrawlState

    with CrawlState(args.state) as state:
        changed = state.reset(args.domain, args.status)
    print(f"Reset {changed} pages to pending")
    return 0


def command_status(args: argparse.Namespace) -> int:
    from vibiomir.ingest.state import CrawlState

    with CrawlState(args.state) as state:
        rows = state.summary()
    by_language: dict[str, Counter] = defaultdict(Counter)
    by_domain: dict[str, Counter] = defaultdict(Counter)
    for language, domain, status, count in rows:
        by_language[language][status] += count
        by_domain[domain][status] += count
    for language, counts in sorted(by_language.items()):
        print(f"{language}: total {sum(counts.values())} {dict(counts)}")
    ranked = sorted(by_domain.items(), key=lambda item: -sum(item[1].values()))
    for domain, counts in ranked[: args.top]:
        print(f"  {domain:40s} {sum(counts.values()):7d} {dict(counts)}")
    return 0


def add_ingest_parser(subparsers) -> None:
    ingest = subparsers.add_parser("ingest", help="Download, crawl and extract the corpus")
    commands = ingest.add_subparsers(dest="ingest_command", required=True)

    download = commands.add_parser("download-official", help="Fetch query + links parquet")
    download.add_argument("--dest", default="data/raw/official")
    from vibiomir.ingest.official import OFFICIAL_REVISION

    download.add_argument("--revision", default=OFFICIAL_REVISION)
    download.set_defaults(handler=command_download_official)

    manifest = commands.add_parser("manifest", help="Write id,url,domain,language parquet")
    manifest.add_argument("--links", default=DEFAULT_LINKS)
    manifest.add_argument("--output", default="data/interim/manifest.parquet")
    manifest.set_defaults(handler=command_manifest)

    crawl = commands.add_parser("crawl", help="Crawl pending pages; safe to stop and rerun")
    crawl.add_argument("--links", default=DEFAULT_LINKS, help="seeds the state on first run")
    crawl.add_argument("--state", default=DEFAULT_STATE)
    crawl.add_argument("--out", default="data/interim/parts/crawl")
    crawl.add_argument("--samples", default="data/raw/html_samples")
    crawl.add_argument("--log-file", default="data/interim/logs/crawl.log")
    crawl.add_argument("--domain", action="append", help="only these domains (repeatable)")
    crawl.add_argument(
        "--skip-domain", action="append", help="replaces the default skip list (pubmed)"
    )
    crawl.add_argument("--exclude-ids", action="append", help="parquet/txt of ids to leave out")
    crawl.add_argument("--limit-per-domain", type=int)
    crawl.add_argument(
        "--sample-per-domain",
        type=int,
        help="seed a deterministic random sample per host instead of every official URL",
    )
    crawl.add_argument("--sample-seed", type=int, default=20261027)
    crawl.add_argument("--max-attempts", type=int, default=3)
    crawl.add_argument("--concurrency", type=int, default=128, help="global open requests")
    crawl.add_argument("--domain-concurrency", type=int, default=4)
    crawl.add_argument("--rps", type=float, default=2.0, help="start rate per domain")
    crawl.add_argument("--max-rps", type=float, default=6.0, help="rate ceiling per domain")
    crawl.add_argument("--timeout", type=float, default=20.0)
    crawl.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    crawl.add_argument("--ignore-robots", action="store_true")
    crawl.set_defaults(handler=command_crawl)

    audit = commands.add_parser(
        "audit-html", help="Re-extract saved HTML samples and write a quality CSV"
    )
    audit.add_argument("--samples", default="data/raw/html_samples")
    audit.add_argument("--output", default="data/interim/html_audit.csv")
    audit.set_defaults(handler=command_audit_html)

    pubmed = commands.add_parser("pubmed", help="Title + abstract via NCBI E-utilities")
    pubmed.add_argument("--links", default=DEFAULT_LINKS)
    pubmed.add_argument("--out", default="data/interim/parts/pubmed")
    pubmed.add_argument("--raw", default="data/raw/pubmed", help="'' to skip raw XML")
    pubmed.add_argument("--state", default=DEFAULT_STATE, help="'' to leave state untouched")
    pubmed.add_argument("--log-file", default="data/interim/logs/pubmed.log")
    pubmed.add_argument("--batch-size", type=int, default=200)
    pubmed.add_argument("--limit-batches", type=int)
    pubmed.set_defaults(handler=command_pubmed)

    bootstrap = commands.add_parser("bootstrap", help="Convert a public crawl to part schema")
    bootstrap.add_argument("--dataset", required=True, choices=["lighth0406"])
    bootstrap.add_argument("--input", required=True, help="the dataset's corpus parquet")
    bootstrap.add_argument("--links", default=DEFAULT_LINKS)
    bootstrap.add_argument("--out", help="default data/interim/parts/hf_<dataset>")
    bootstrap.add_argument("--log-file", default="data/interim/logs/bootstrap.log")
    bootstrap.set_defaults(handler=command_bootstrap)

    wayback = commands.add_parser(
        "wayback", help="Fetch pages the crawl could not get from the Wayback Machine"
    )
    wayback.add_argument("--state", default=DEFAULT_STATE, help="crawl state to read from")
    wayback.add_argument("--wayback-state", help="default: wayback_state.sqlite next to --state")
    wayback.add_argument("--out", default="data/interim/parts/wayback")
    wayback.add_argument("--samples", default="data/raw/html_samples_wayback")
    wayback.add_argument("--log-file", default="data/interim/logs/wayback.log")
    wayback.add_argument("--domain", action="append", help="only these domains (repeatable)")
    wayback.add_argument(
        "--skip-domain", action="append", help="default: pubmed.ncbi.nlm.nih.gov and vov.vn"
    )
    wayback.add_argument("--timestamp", default="2026", help="prefer captures near this date")
    wayback.add_argument("--limit-per-domain", type=int)
    wayback.add_argument("--max-attempts", type=int, default=3)
    wayback.add_argument("--concurrency", type=int, default=6, help="open requests to Wayback")
    wayback.add_argument("--rps", type=float, default=1.0, help="start rate, all domains")
    wayback.add_argument("--max-rps", type=float, default=3.0, help="rate ceiling, all domains")
    wayback.add_argument("--timeout", type=float, default=60.0)
    wayback.add_argument("--workers", type=int, default=2)
    wayback.set_defaults(handler=command_wayback)

    merge = commands.add_parser("merge", help="One row per official id -> data/processed/corpus")
    merge.add_argument("--manifest", default="data/interim/manifest.parquet")
    merge.add_argument(
        "--parts", nargs="+",
        default=["data/interim/parts/crawl", "data/interim/parts/pubmed",
                 "data/interim/parts/wayback"],
    )
    merge.add_argument("--out", default="data/processed/corpus")
    merge.add_argument("--coverage", default="data/processed/coverage.csv")
    merge.add_argument("--version", default="v0")
    merge.add_argument("--no-clean", action="store_true", help="keep boilerplate paragraphs")
    merge.add_argument("--log-file", default="data/interim/logs/merge.log")
    merge.set_defaults(handler=command_merge)

    export = commands.add_parser("export-jsonl", help="Corpus subset as JSONL for vibiomir run")
    export.add_argument("--corpus", default="data/processed/corpus")
    export.add_argument("--output", required=True)
    export.add_argument("--ids", help="parquet/txt of ids to export")
    export.add_argument("--language", action="append", choices=["vi", "en", "zh"])
    export.add_argument("--limit", type=int)
    export.add_argument("--include-missing", action="store_true")
    export.set_defaults(handler=command_export_jsonl)

    reset = commands.add_parser("reset", help="Mark pages pending again, e.g. after a fix")
    reset.add_argument("--state", default=DEFAULT_STATE)
    reset.add_argument("--domain", action="append", required=True, help="also subdomains")
    reset.add_argument("--status", action="append", required=True)
    reset.set_defaults(handler=command_reset)

    status = commands.add_parser("status", help="Summarize crawl state by language and domain")
    status.add_argument("--state", default=DEFAULT_STATE)
    status.add_argument("--top", type=int, default=30)
    status.set_defaults(handler=command_status)
