"""Corpus ingestion: official links, domain table, crawl state, fetcher and extractors.

Heavy dependencies (httpx, pyarrow, lxml, trafilatura, huggingface_hub) are imported lazily;
install them with ``pip install -e '.[ingest]'``.
"""

from .domains import domain_language, url_domain
from .state import CrawlState, PageUpdate

__all__ = ["CrawlState", "PageUpdate", "domain_language", "url_domain"]
