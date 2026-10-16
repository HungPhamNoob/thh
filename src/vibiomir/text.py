from __future__ import annotations

import html
import re
import unicodedata

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_SPACE_RE = re.compile(r"\s+")


def official_normalize(text: str) -> str:
    """Normalization specified by the ViBioMIR chunk-overlap rule."""
    return _SPACE_RE.sub(" ", html.unescape(unicodedata.normalize("NFKC", text)).lower()).strip()


def lexical_tokens(text: str, cjk_bigrams: bool = True) -> list[str]:
    normalized = official_normalize(text)
    tokens = _WORD_RE.findall(normalized)
    cjk_chars = _CJK_RE.findall(normalized)
    if cjk_bigrams and len(cjk_chars) > 1:
        tokens.extend(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
    return tokens


def whitespace_token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]

