"""DOCX extraction, preserving the order of paragraphs and tables.

python-docx exposes paragraphs and tables as separate collections, which loses
their interleaving. Forms like the improvement-examination form are almost
entirely tables with a few headings between them, so document order carries
the meaning and has to be reconstructed from the underlying XML.
"""
from __future__ import annotations

from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..models import Block
from .tables import table_to_blocks


def _iter_body(parent: DocxDocument):
    body = parent.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def extract_docx(path: str | Path, cfg) -> tuple[list[Block], list[str], str]:
    """Returns (blocks, warnings, leading_text_for_metadata)."""
    document = docx.Document(str(path))
    blocks: list[Block] = []
    warnings: list[str] = []
    leading: list[str] = []
    last_heading = ""

    for item in _iter_body(document):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            if len(leading) < 12:
                leading.append(text)
            style = (item.style.name or "").lower()
            if style.startswith("heading") or style == "title":
                level = 1 if style in ("title", "heading 1") else 2
                last_heading = text
                blocks.append(Block(kind="heading", text=text, page_no=1, level=level))
            else:
                blocks.append(Block(kind="para", text=text, page_no=1))
        else:
            rows = [[cell.text for cell in row.cells] for row in item.rows]
            if not any(any(c.strip() for c in r) for r in rows):
                continue
            # A form table is meaningless without the heading above it.
            produced = table_to_blocks(rows, page_no=1, caption=last_heading)
            if produced:
                blocks.extend(produced)
            else:
                warnings.append("a table produced no usable text")

    # DOCX has no fixed pagination, so every block is reported as page 1 and
    # citations fall back to the section path.
    return blocks, warnings, "\n".join(leading)
