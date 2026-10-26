"""Text-level cleanup applied when merging the corpus (D5/D6).

Raw HTML is not kept for most pages, so boilerplate that the extractors let through is removed
from the extracted text instead:

- paragraphs made only of table or list punctuation (``| |``, ``-``, ``›``);
- paragraphs that repeat across many pages of one domain, learned from a sample of that
  domain (hotline banners, copyright lines, share bars, template placeholders);
- everything after a "related articles" heading in the second half of a page.

The first paragraph is the title and is always kept. Paragraphs are only dropped whole, so the
remaining text is still the page's own wording.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

NOISE_PARAGRAPH = re.compile(r"^[\s|\-–—•·›»«‹,.:;*_=~#]*$")
# Headings that start the "more articles" tail of a page. Inline boxes such as Vietnamese
# "tin liên quan" are left out: news sites put them in the middle of the article.
TAIL_MARKERS = frozenset({
    "BÀI VIẾT CÙNG CHỦ ĐỀ",
    "Các tin khác",
    "CÁC TIN KHÁC",
    "Tin cùng chuyên mục",
    "ĐĂNG KÝ NHẬN TIN",
    "Xem thêm video đang được quan tâm:",
    "相关文章",
    "相关阅读",
    "其他类似疾病",
    "医院医生更多",
    "精彩推荐",
    "- 精彩推荐",
})
TAIL_PREFIXES = ("- 网友评论：", "网友评论：")
MAX_BOILERPLATE_CHARS = 400


@dataclass(slots=True)
class BoilerplateModel:
    """Counts short paragraphs per domain on up to ``sample`` pages each."""

    sample: int = 1000
    min_docs: int = 30
    min_share: float = 0.3
    docs: Counter = field(default_factory=Counter)
    counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def observe(self, domain: str, text: str) -> None:
        if self.docs[domain] >= self.sample or not text:
            return
        self.docs[domain] += 1
        paragraphs = {p.strip() for p in text.split("\n\n")[1:]}
        self.counts[domain].update(
            p for p in paragraphs if 0 < len(p) <= MAX_BOILERPLATE_CHARS
        )

    def boilerplate(self) -> dict[str, frozenset[str]]:
        result = {}
        for domain, counts in self.counts.items():
            n = self.docs[domain]
            if n < self.min_docs:
                continue
            frequent = frozenset(p for p, c in counts.items() if c >= self.min_share * n)
            if frequent:
                result[domain] = frequent
        return result


def clean_text(text: str, boilerplate: frozenset[str] = frozenset()) -> str:
    if not text:
        return text
    paragraphs = text.split("\n\n")
    kept = [paragraphs[0]]
    half = len(text) / 2
    position = len(paragraphs[0])
    for paragraph in paragraphs[1:]:
        stripped = paragraph.strip()
        position += len(paragraph) + 2
        if position > half and (stripped in TAIL_MARKERS or stripped.startswith(TAIL_PREFIXES)):
            break
        if NOISE_PARAGRAPH.match(stripped) or stripped in boilerplate:
            continue
        kept.append(paragraph)
    return "\n\n".join(kept)
