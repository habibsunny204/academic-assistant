"""Core data structures shared by the extraction and indexing stages.

The guiding rule: a citation can only show what ingestion captured. The UI
mockup promises "Page 45 - Last Updated: February 10, 2025" for every answer,
so page number and document date are mandatory fields, not optional extras.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

BlockKind = Literal["heading", "para", "table", "table_fact", "list"]
DocStatus = Literal["indexed", "quarantined", "superseded", "unchanged"]


@dataclass
class Block:
    """A contiguous piece of a document, tied to the page it came from."""

    kind: BlockKind
    text: str
    page_no: int
    level: int = 0  # heading depth; 0 for body content
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocMeta:
    """Document-level facts used for citation and temporal filtering."""

    doc_id: str
    source_path: str
    filename: str
    sha256: str
    title: str
    doc_type: str = "unknown"      # notice | routine | form | policy | curriculum
    category: str = "general"      # maps to the UI sidebar categories
    effective_from: date | None = None
    issued_date: date | None = None
    semester: str | None = None    # e.g. "Fall 2026"
    version_label: str | None = None  # e.g. "V1.1"
    n_pages: int = 0
    extractor: str = ""
    ocr_pages: list[int] = field(default_factory=list)


@dataclass
class ExtractedDoc:
    meta: DocMeta
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


@dataclass
class Chunk:
    """The retrieval unit. Everything needed to cite it travels with it."""

    chunk_id: str
    doc_id: str
    ord: int
    text: str
    page_start: int
    page_end: int
    section_path: str = ""   # "Examination > Improvement Exam > Eligibility"
    kind: BlockKind = "para"
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, ord_: int, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        return f"{doc_id}:{ord_:04d}:{digest}"

    def citation(self) -> dict[str, Any]:
        """The payload the frontend renders as a source card."""
        return {
            "doc_id": self.doc_id,
            "title": self.meta.get("title", ""),
            "page": self.page_start,
            "page_end": self.page_end,
            "section": self.section_path,
            "source_path": self.meta.get("source_path", ""),
            "effective_from": self.meta.get("effective_from"),
            "issued_date": self.meta.get("issued_date"),
            "semester": self.meta.get("semester"),
            "category": self.meta.get("category"),
        }
