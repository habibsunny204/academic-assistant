"""Text normalisation shared by grounding verification and evaluation.

Both need to decide whether "8:30 AM to 10:00 AM" and "08:30-10:00", or
"KT-318 (A)" and "KT-318(A)", say the same thing. Keeping one implementation
means the evaluator scores answers by exactly the rules the verifier applies.
"""
from __future__ import annotations

import re
import unicodedata

_MONTHS = ("january february march april may june july august september "
           "october november december").split()


def normalize(text: str) -> str:
    """Canonical lower-case form: times, ordinals, markers and punctuation unified."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"[\u2010-\u2015\u2212]", "-", t)
    t = re.sub(r"\[\d{1,2}\]", " ", t)          # citation markers
    t = t.replace("**", "").replace("/=", " ")
    t = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"(\d{1,2}:\d{2})\s*(?:a\.?m\.?|p\.?m\.?)(?![a-z])", r"\1", t)
    t = re.sub(r"\b0(\d:\d{2})", r"\1", t)       # 08:30 -> 8:30
    t = re.sub(r"(\d{1,2}:\d{2})\s*(?:-|to|until|till|and)\s*(\d{1,2}:\d{2})", r"\1-\2", t)
    t = re.sub(r"\s+\(", "(", t)                 # "kt-318 (a)" -> "kt-318(a)"
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)    # keep decimal points only
    t = re.sub(r"(?<!\d):|:(?!\d)", " ", t)      # keep colons inside times only
    t = re.sub(r"[,;!?\"'`]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def contains_term(normalized_text: str, term: str) -> bool:
    """Whole-token containment. "_" is part of a token, so "67" is not in "67_b"."""
    target = normalize(term)
    return bool(target) and re.search(
        rf"(?<![a-z0-9_]){re.escape(target)}(?![a-z0-9_])", normalized_text
    ) is not None


# Optional "kt-"/"anx1-" prefixes, then a digit-bearing core, then "-202", ":30", "(a)".
_TERM_RE = re.compile(
    r"(?:[a-z0-9_]+-)*[a-z0-9_]*\d[a-z0-9_]*(?:[-:/][a-z0-9_]+)*(?:\([a-z0-9]{1,3}\))?"
)


def checkable_terms(sentence: str) -> list[str]:
    """Terms carrying a digit: dates, fees, times, course codes, sections, rooms.

    These are the details a model most often gets wrong and a reader can least
    afford to have wrong, and whether a passage contains them is a yes/no
    question rather than a matter of similarity.
    """
    return list(dict.fromkeys(_TERM_RE.findall(normalize(sentence))))


def human_dates(iso_date: str) -> str:
    """'2026-09-09' -> '2026-09-09 9 september 2026', so either form can match."""
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", (iso_date or "").strip())
    if not match:
        return iso_date or ""
    year, month, day = match.groups()
    return f"{iso_date} {int(day)} {_MONTHS[int(month) - 1]} {year}"
