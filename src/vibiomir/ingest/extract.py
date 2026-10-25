"""HTML decoding and main-content extraction.

Extraction order for one page: site rule for the host → generic CMS container rules →
trafilatura (if installed) → densest ``<p>`` container. Every block element becomes its own
paragraph and paragraphs are joined with ``\\n\\n`` so that ``<p>A</p><p>B</p>`` never turns
into ``AB``. Characters are kept as on the page; only whitespace runs inside a line are
collapsed, which the official normalizer does as well.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, field

BLOCK_TAGS = frozenset(
    "address article aside blockquote caption dd details dialog div dl dt fieldset figcaption "
    "figure footer h1 h2 h3 h4 h5 h6 header hr li main nav ol p pre section summary table "
    "tbody thead tfoot tr ul".split()
)
CELL_TAGS = frozenset({"td", "th"})
DROP_TAGS = frozenset(
    "script style noscript iframe svg button select option input textarea template object "
    "embed canvas video audio map head".split()
)
_INLINE_WS = re.compile(r"[ \t\r\f\v   -   　]+")
_PARAGRAPH_BREAK = re.compile(r"\n[^\S\n]*\n")
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.I)
_XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)
# A stray early "</html>" makes libxml2 put head/body outside the root element.
_CLOSING_ROOT_TAGS = re.compile(r"</(?:html|body)\s*>", re.I)
_BASE64_RUN = re.compile(r"^[A-Za-z0-9+/=]{200,}$")
_ENCODING_ALIASES = {
    "gb2312": "gb18030",
    "gbk": "gb18030",
    "x-gbk": "gb18030",
    "gb_2312-80": "gb18030",
}

MIN_OK_CHARS = 200


def _canonical_encoding(name: str | None) -> str | None:
    if not name:
        return None
    name = name.strip().strip("\"'").lower()
    name = _ENCODING_ALIASES.get(name, name)
    try:
        return codecs.lookup(name).name
    except LookupError:
        return None


def header_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    match = re.search(r"charset\s*=\s*([^;\s]+)", content_type, re.I)
    if not match:
        return None
    # Some servers send lists such as "utf-8,gbk": take the first entry.
    return _canonical_encoding(match.group(1).split(",")[0])


def decode_html(body: bytes, content_type: str | None = None) -> tuple[str, str]:
    """Decode a page using header charset, then ``<meta>`` charset, then sniffing."""
    if body.startswith(codecs.BOM_UTF8):
        return body[len(codecs.BOM_UTF8) :].decode("utf-8", errors="replace"), "utf-8"
    declared: list[str] = []
    for candidate in (header_charset(content_type), *_meta_charsets(body)):
        if candidate and candidate not in declared:
            declared.append(candidate)
    for encoding in [*declared, "utf-8"]:
        try:
            return body.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(body).best()
        if best is not None:
            encoding = _canonical_encoding(best.encoding) or "utf-8"
            return str(best), encoding
    except ImportError:
        pass
    encoding = declared[0] if declared else "utf-8"
    return body.decode(encoding, errors="replace"), encoding


def _meta_charsets(body: bytes) -> list[str]:
    found = []
    for raw in _META_CHARSET.findall(body[:8192]):
        name = _canonical_encoding(raw.decode("ascii", errors="ignore"))
        if name:
            found.append(name)
    return found


def element_text(element) -> str:
    """Render an lxml element as text with one paragraph per block element."""
    parts: list[str] = []
    # Explicit stack instead of recursion: some pages nest thousands of elements deep.
    stack: list[tuple[object, bool]] = [(element, False)]
    while stack:
        node, closing = stack.pop()
        tag = node.tag if isinstance(node.tag, str) else None
        name = tag.lower() if tag else None
        if closing:
            if name in BLOCK_TAGS:
                parts.append("\n\n")
            elif name in CELL_TAGS:
                parts.append(" ")
            if node is not element and node.tail:
                parts.append(node.tail)
            continue
        if name is None or name in DROP_TAGS:
            if node is not element and node.tail:
                parts.append(node.tail)
            continue
        if name in BLOCK_TAGS:
            parts.append("\n\n")
        elif name == "br":
            parts.append("\n")
        elif name in CELL_TAGS:
            parts.append(" ")
        if node.text:
            parts.append(node.text)
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(node))
    return clean_text("".join(parts))


def clean_text(raw: str) -> str:
    paragraphs = []
    for block in _PARAGRAPH_BREAK.split(raw.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [_INLINE_WS.sub(" ", line).strip() for line in block.split("\n")]
        lines = [line for line in lines if line]
        if lines:
            paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


@dataclass(frozen=True, slots=True)
class SiteRule:
    """CSS selectors for one site. Body selectors are read in order; each takes all matches."""

    name: str
    body: tuple[str, ...]
    title: tuple[str, ...] = ()  # empty: the h1 that matches og:title, see _page_title
    drop: tuple[str, ...] = ()
    min_chars: int = 1


_MEDIAWIKI_DROP = (
    "#toc", ".toc", ".mw-editsection", ".printfooter", "#catlinks", ".navbox", "#siteSub",
    "#contentSub", "#jump-to-nav", ".mw-jump-link",
)

# Matched by host suffix, most specific first.
SITE_RULES: dict[str, SiteRule] = {
    "bachmai.gov.vn": SiteRule(
        "site:bachmai.gov.vn",
        body=("#content-post .content-excerpt", "#content-post .content-editor"),
        title=("#content-post h3.h3", "meta[property='og:title']"),
    ),
    "vinmec.com": SiteRule(
        "site:vinmec.com", body=("#main-article",), drop=("#toc-container", ".table-of-contents")
    ),
    "medlatec.vn": SiteRule("site:medlatec.vn", body=(".block-posts-single .description",)),
    "giadinhonline.vn": SiteRule(
        "site:giadinhonline.vn", body=(".post-sapo", "#article-body", "#explus-editor")
    ),
    "youmed.vn": SiteRule("site:youmed.vn", body=("article .prose",)),
    "hellobacsi.com": SiteRule("site:hellobacsi.com", body=(".body-content",)),
    "vietnamnet.vn": SiteRule("site:vietnamnet.vn", body=(".content-detail-sapo", "#maincontent")),
    "ask.39.net": SiteRule(
        "site:ask.39.net",
        body=(".ask_hid .txt_ms", ".ask_hid .txt_bc", ".sele_txt"),
        title=("h1.ask_tit",),
    ),
    "39.net": SiteRule(
        "site:39.net", body=(".art_summary", ".art_content"), drop=(".art_rec", ".art_footer")
    ),
    "cnkang.com": SiteRule("site:cnkang.com", body=("#endText",), drop=("#jkpl",)),
    "familydoctor.com.cn": SiteRule("site:familydoctor.com.cn", body=("#viewContent",)),
    "a-hospital.com": SiteRule(
        "site:mediawiki", body=("#bodyContent",), drop=_MEDIAWIKI_DROP, min_chars=20
    ),
    "zhongyibaodian.net": SiteRule(
        "site:mediawiki", body=("#bodyContent",), drop=_MEDIAWIKI_DROP, min_chars=20
    ),
}

# Shared CMS layouts seen on many Vietnamese news sites; tried on every host without a site rule.
GENERIC_RULES: tuple[SiteRule, ...] = (
    SiteRule("cms:detail-content", body=(".detail-sapo", ".detail-content"), min_chars=100),
    SiteRule("cms:article-body", body=(".article__sapo", ".article__body"), min_chars=100),
    SiteRule("cms:content-fck", body=(".content-fck",), min_chars=100),
    SiteRule("cms:td-post-content", body=(".td-post-content",), min_chars=100),
    SiteRule("cms:entry-content", body=(".entry-content",), min_chars=100),
    SiteRule("cms:article-body-itemprop", body=("[itemprop='articleBody']",), min_chars=100),
)

SOFT_404_PATTERNS = (
    "trang không tồn tại",
    "không tìm thấy trang",
    "bài viết không tồn tại",
    "page not found",
    "页面不存在",
    "您访问的页面不存在",
    "页面找不到",
)


def site_rule(host: str) -> SiteRule | None:
    host = host.lower()
    best: tuple[int, SiteRule] | None = None
    for suffix, rule in SITE_RULES.items():
        if host == suffix or host.endswith("." + suffix):
            if best is None or len(suffix) > best[0]:
                best = (len(suffix), rule)
    return best[1] if best else None


@dataclass(slots=True)
class Extracted:
    title: str
    text: str
    extractor: str
    encoding: str
    published_at: str | None = None
    soft_404: bool = False
    metadata: dict = field(default_factory=dict)


def _select_text(doc, selectors: tuple[str, ...]) -> str:
    chosen = []
    for selector in selectors:
        for element in doc.cssselect(selector):
            if any(element in kept.iterdescendants() or element is kept for kept in chosen):
                continue
            chosen.append(element)
    return "\n\n".join(filter(None, (element_text(element) for element in chosen)))


def _first_title(doc, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        for element in doc.cssselect(selector):
            value = element.get("content") if element.tag == "meta" else element_text(element)
            value = clean_text(value or "").replace("\n", " ")
            if value:
                return value
    return ""


def _page_title(doc) -> str:
    """Prefer the h1 that agrees with og:title: the first h1 is often a logo or category."""
    og_title = _first_title(doc, ("meta[property='og:title']",))
    headings = [element_text(h).replace("\n", " ") for h in doc.cssselect("h1")]
    headings = [heading for heading in headings if heading]
    if og_title:
        for heading in headings:
            if heading in og_title or og_title in heading:
                return heading
        return og_title
    if headings:
        return headings[0]
    return _first_title(doc, ("title",))


def _published_at(doc) -> str | None:
    for selector in (
        "meta[property='article:published_time']",
        "meta[itemprop='datePublished']",
        "meta[name='pubdate']",
        "meta[name='publishdate']",
    ):
        for element in doc.cssselect(selector):
            if element.get("content"):
                return element.get("content").strip()
    return None


def _densest_container(doc) -> str:
    scores: dict = {}
    for paragraph in doc.iter("p"):
        parent = paragraph.getparent()
        if parent is not None:
            scores[parent] = scores.get(parent, 0) + len(paragraph.text_content().strip())
    if not scores:
        body = doc.find("body")
        return element_text(body) if body is not None else ""
    return element_text(max(scores, key=scores.get))


def _trafilatura(html: str) -> str | None:
    try:
        import trafilatura
    except ImportError:
        return None
    text = trafilatura.extract(
        html, include_comments=False, include_tables=True, favor_recall=True, deduplicate=False
    )
    return clean_text(text.replace("\n", "\n\n")) if text else None


def extract_html(html: str, host: str, encoding: str = "utf-8") -> Extracted:
    import lxml.html

    # lxml rejects str input that still carries an XML encoding declaration.
    html = _CLOSING_ROOT_TAGS.sub("", _XML_DECLARATION.sub("", html, count=1))
    doc = lxml.html.document_fromstring(html or "<html/>")
    for bad in doc.xpath("//script|//style|//noscript|//template"):
        bad.drop_tree()
    published_at = _published_at(doc)

    rule = site_rule(host)
    rules = ((rule,) if rule else ()) + GENERIC_RULES
    title, body, extractor = "", "", ""
    for candidate in rules:
        for selector in candidate.drop:
            for element in doc.cssselect(selector):
                element.drop_tree()
        text = _select_text(doc, candidate.body)
        if len(text) >= candidate.min_chars:
            title = _first_title(doc, candidate.title) if candidate.title else ""
            body, extractor = text, candidate.name
            break
    if not extractor:
        text = _trafilatura(html)
        if text:
            body, extractor = text, "trafilatura"
        else:
            body, extractor = _densest_container(doc), "densest_p"
    title = title or _page_title(doc)
    # Some sites embed base64 blobs that their JavaScript decodes; they are not page text.
    body = "\n\n".join(p for p in body.split("\n\n") if not _BASE64_RUN.match(p))

    text = body
    if title and not body.startswith(title):
        text = f"{title}\n\n{body}" if body else title
    lowered = text[:2000].lower()
    soft_404 = len(body) < MIN_OK_CHARS and any(p in lowered for p in SOFT_404_PATTERNS)
    return Extracted(
        title=title,
        text=text,
        extractor=extractor,
        encoding=encoding,
        published_at=published_at,
        soft_404=soft_404,
    )


def extract_page(body: bytes, content_type: str | None, host: str) -> Extracted:
    html, encoding = decode_html(body, content_type)
    return extract_html(html, host, encoding)
