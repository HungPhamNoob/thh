"""Domain → language table and per-domain crawl policy.

Language comes from the domain table, never from language detection: several Chinese sites
have neutral names (``test.pmphai.com``, ``www.wujue.com``, ``jb39.com``, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

EN_SUFFIXES: tuple[str, ...] = ("pubmed.ncbi.nlm.nih.gov",)

ZH_SUFFIXES: tuple[str, ...] = (
    "120ask.com",
    "39.net",
    "jb39.com",
    "cnkang.com",
    "familydoctor.com.cn",
    "familydoctor.cn",
    "zysjonline.com",
    "zhongyibaodian.net",
    "a-hospital.com",
    "youlai.cn",
    "msdmanuals.cn",
    "iiyi.com",
    "people.com.cn",
    "pmphai.com",
    "wujue.com",
    "zydcd.com",
    "qihuangzhishu.com",
    "baidu.com",
)

# Domains that are fetched through another channel instead of HTML crawling.
DEFAULT_SKIP_DOMAINS: tuple[str, ...] = ("pubmed.ncbi.nlm.nih.gov",)  # NCBI E-utilities (D2)


def url_domain(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _matches(host: str, suffixes: tuple[str, ...]) -> bool:
    return any(host == suffix or host.endswith("." + suffix) for suffix in suffixes)


def domain_language(host: str) -> str:
    host = host.lower()
    if _matches(host, EN_SUFFIXES):
        return "en"
    if _matches(host, ZH_SUFFIXES):
        return "zh"
    return "vi"


@dataclass(slots=True)
class DomainPolicy:
    """Politeness settings for one domain; the crawler adapts ``rps`` at run time."""

    rps: float = 2.0
    max_rps: float = 6.0
    concurrency: int = 4


_SECOND_LEVEL = {"com", "gov", "org", "net", "edu", "ac"}

# Hosts that get their own rate group although they share a registrable domain.
RATE_GROUP_OVERRIDES: dict[str, str] = {}


def rate_group(host: str) -> str:
    """Hosts sharing a registrable domain share one rate limit (``*.39.net`` -> ``39.net``).

    Limiting per host let ~20 subdomains of 39.net add up to ~40 requests/s against one
    backend during the 2026-10-25 pilot, which then answered with a slider captcha. The same
    captcha later hit ask.39.net at 2-4 requests/s, so it shares the 39.net group too.
    """
    host = host.lower()
    if host in RATE_GROUP_OVERRIDES:
        return RATE_GROUP_OVERRIDES[host]
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


# Keyed by rate group. Sites measured on 2026-10-25 with multi-second latency get more
# connections at the same request rate.
DOMAIN_POLICIES: dict[str, DomainPolicy] = {
    "39.net": DomainPolicy(rps=1.0, max_rps=2.0, concurrency=8),
    "cnkang.com": DomainPolicy(rps=2.0, max_rps=6.0, concurrency=8),
    "benhvienvietduc.org": DomainPolicy(rps=1.0, max_rps=2.0, concurrency=4),
}


def domain_policy(group: str, default: DomainPolicy | None = None) -> DomainPolicy:
    base = DOMAIN_POLICIES.get(group)
    if base is None:
        base = default or DomainPolicy()
    return DomainPolicy(rps=base.rps, max_rps=base.max_rps, concurrency=base.concurrency)
