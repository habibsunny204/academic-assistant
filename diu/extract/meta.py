"""Deriving document-level metadata from filenames and first-page text.

Temporal filtering is only as good as the dates captured here. A notice without
an issue date cannot be ranked against a newer one, so when a date cannot be
found the document is flagged rather than silently defaulting to today.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from dateutil import parser as date_parser

SEMESTER_RE = re.compile(r"\b(Spring|Summer|Fall|Short)[\s\-_]*(20\d{2})\b", re.I)
VERSION_RE = re.compile(r"\bversion[\s\-_]*v?\s*([\d.]+)\b", re.I)
VERSION_FILE_RE = re.compile(r"[-_]v(\d+(?:[._]\d+)*)(?=[-_.]|$)", re.I)
EFFECTIVE_RE = re.compile(r"effective\s+from\s*:?\s*([^\n]{4,40})", re.I)
ISSUED_RE = re.compile(r"\bdate\s*:?\s*([0-3]?\d[^\n]{4,30})", re.I)
TRAILING_HASH_RE = re.compile(r"(?:[-_][0-9a-f]{8,})+$", re.I)

DOC_TYPE_KEYWORDS = [
    ("routine", ("routine", "schedule", "timetable")),
    ("notice", ("notice", "announcement", "circular")),
    ("form", ("form", "application")),
    ("curriculum", ("curriculum", "syllabus", "course-outline")),
    ("policy", ("policy", "policies", "guideline", "handbook", "rule", "regulation")),
    ("calendar", ("calendar",)),
]

# Mirrors the sidebar categories in the UI mock.
CATEGORY_KEYWORDS = [
    ("Admission", ("admission", "admit", "enrolment", "enrollment")),
    ("Registration", ("registration", "register", "add-drop", "semester drop")),
    ("Examination", ("exam", "examination", "improvement", "retake", "midterm", "final")),
    ("Curriculum", ("curriculum", "syllabus", "course", "credit", "prerequisite")),
    ("Graduation", ("graduation", "convocation", "clearance", "transcript")),
    ("Scholarships", ("scholarship", "waiver", "stipend", "financial aid")),
    ("Student Services", ("hostel", "transport", "library", "counsel", "club")),
    ("Academic Calendar", ("calendar", "holiday", "vacation")),
    ("Policies", ("policy", "policies", "rule", "regulation", "guideline")),
    ("Notices", ("notice", "announcement", "circular")),
]


def _parse_date(text: str) -> date | None:
    text = re.sub(r"[,\u2013\u2014]", " ", text).strip(" .:\t")
    text = re.split(r"\s{2,}", text)[0].strip()
    if not text:
        return None
    try:
        parsed = date_parser.parse(text, dayfirst=True, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    if not (1990 <= parsed.year <= 2100):
        return None
    return parsed.date()


# Letterhead lines: the institution's name and address head most notices, so
# they say nothing about which document this is.
LETTERHEAD_RE = re.compile(
    r"daffodil international university|daffodil smart city|birulia|savar|"
    r"dhaka\s*-?\s*\d{4}|^notice$|^office of the",
    re.I,
)
_SMALL_WORDS = {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}
_ACRONYMS = {"cse", "eee", "bba", "diu", "swe", "mct", "ict", "ph", "id", "gpa", "cgpa"}


def clean_filename_title(filename: str) -> str:
    """'notice-for-late-registration-fall-2026-1d709bde62.pdf'
    -> 'Notice for Late Registration Fall 2026'."""
    stem = Path(filename).stem
    stem = TRAILING_HASH_RE.sub("", stem)
    words = re.sub(r"[-_]+", " ", stem).split()
    if not words:
        return Path(filename).stem
    titled = []
    for i, word in enumerate(words):
        lower = word.lower()
        if lower in _ACRONYMS:
            titled.append(lower.upper())
        elif i > 0 and lower in _SMALL_WORDS:
            titled.append(lower)
        else:
            titled.append(word[:1].upper() + word[1:].lower())
    return " ".join(titled)


def _first_title_line(text: str) -> str | None:
    """The first heading-like line, unless it is letterhead.

    Only the first candidate is considered: when a document opens with the
    university's name, the lines after it are usually body text, which makes a
    worse title than the filename.
    """
    for raw in text.split("\n"):
        line = raw.strip()
        if 8 <= len(line) <= 110 and not line.lower().startswith(("page ", "date:")):
            if sum(c.isalpha() for c in line) >= len(line) * 0.5:
                return None if LETTERHEAD_RE.search(line) else line
    return None


def _classify(haystack: str, table: list[tuple[str, tuple[str, ...]]], default: str) -> str:
    lowered = haystack.lower()
    for label, keywords in table:
        if any(keyword in lowered for keyword in keywords):
            return label
    return default


def derive_metadata(filename: str, head_text: str) -> dict:
    """Combine filename signals with first-page text into document metadata."""
    head_text = head_text or ""
    haystack = f"{filename} {head_text[:2000]}"

    title = _first_title_line(head_text) or clean_filename_title(filename)

    semester = None
    match = SEMESTER_RE.search(haystack)
    if match:
        semester = f"{match.group(1).capitalize()} {match.group(2)}"

    version = None
    match = VERSION_RE.search(head_text) or VERSION_FILE_RE.search(Path(filename).stem)
    if match:
        version = "V" + match.group(1).replace("_", ".")

    effective_from = None
    match = EFFECTIVE_RE.search(head_text)
    if match:
        effective_from = _parse_date(match.group(1))

    issued_date = None
    match = ISSUED_RE.search(head_text)
    if match:
        issued_date = _parse_date(match.group(1))

    doc_type = _classify(haystack, DOC_TYPE_KEYWORDS, "unknown")
    # A routine is its own question category ("where is my class?"), not
    # curriculum, even though it mentions courses.
    if doc_type == "routine":
        category = "Class Routine"
    else:
        category = _classify(haystack, CATEGORY_KEYWORDS, "General Information")

    return {
        "title": title,
        "doc_type": doc_type,
        "category": category,
        "semester": semester,
        "version_label": version,
        "effective_from": effective_from,
        "issued_date": issued_date,
    }
