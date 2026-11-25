"""Text normalisation applied between extraction and chunking.

Kept deliberately conservative: cleaning that "improves" text can destroy the
exact strings users search for (course codes, room numbers, fee amounts), so
this only fixes encoding artefacts and layout noise.
"""
from __future__ import annotations

import re
import unicodedata

from .models import Block

# Ligatures and typographic characters that break exact-match search.
_REPLACEMENTS = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u200b": "", "\ufeff": "",
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
}

_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?[-\u2013]?\s*\d{1,3}\s*[-\u2013]?\s*$", re.I)
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+(?=\n)")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for source, target in _REPLACEMENTS.items():
        text = text.replace(source, target)
    # U+FFFD means the extractor could not decode a glyph. Drop it rather than
    # leaving a replacement character inside a citable quote.
    text = text.replace("\ufffd", "")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def is_noise(text: str) -> bool:
    """Page numbers, rules and stray punctuation carry no answerable content."""
    stripped = text.strip()
    if not stripped or _PAGE_NUMBER_RE.match(stripped):
        return True
    if len(stripped) < 3:
        return True
    if not any(char.isalnum() for char in stripped):
        return True
    return False


def clean_blocks(blocks: list[Block]) -> list[Block]:
    """Clean, drop noise, and collapse headers repeated on every page."""
    cleaned: list[Block] = []
    for block in blocks:
        text = clean_text(block.text)
        if is_noise(text):
            continue
        block.text = text
        cleaned.append(block)

    # A short line appearing on most pages is a running header or footer.
    pages = {b.page_no for b in cleaned}
    if len(pages) >= 4:
        counts: dict[str, set[int]] = {}
        for block in cleaned:
            if block.kind in ("para", "heading") and len(block.text) < 90:
                counts.setdefault(block.text, set()).add(block.page_no)
        boilerplate = {
            text for text, seen in counts.items()
            if len(seen) >= max(3, int(len(pages) * 0.6))
        }
        if boilerplate:
            cleaned = [b for b in cleaned if b.text not in boilerplate]
    return cleaned
