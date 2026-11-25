"""PDF extraction: text layer, tables, and OCR fallback for scanned pages.

Three kinds of PDF show up in DIU sources and each needs different handling:
  * born-digital prose (policies, handbooks)      -> text layer + heading detection
  * born-digital tables (class routines)          -> table extraction, see tables.py
  * scanned images with no text layer (notices)   -> OCR

The decision is made per page, not per document, because a single handbook can
contain a scanned annexure.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import pymupdf
from PIL import Image

from ..models import Block
from .ocr import OcrBackend, OcrResult, OcrUnavailable
from .tables import RoutineState, table_to_blocks

# A text block is treated as belonging to a table if most of its area sits
# inside the table's bounding box.
_TABLE_OVERLAP_THRESHOLD = 0.5


def _overlaps_table(rect: pymupdf.Rect, table_rects: list[pymupdf.Rect]) -> bool:
    area = abs(rect.get_area())
    if area <= 0:
        return False
    for table_rect in table_rects:
        intersection = rect & table_rect
        if abs(intersection.get_area()) / area > _TABLE_OVERLAP_THRESHOLD:
            return True
    return False


def _span_sizes(page: pymupdf.Page) -> list[float]:
    sizes = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    sizes.append(round(span["size"], 1))
    return sizes


def _text_blocks(
    page: pymupdf.Page, table_rects: list[pymupdf.Rect], body_size: float
) -> list[Block]:
    """Pull prose out of a page, marking larger short lines as headings."""
    blocks: list[Block] = []
    page_no = page.number + 1

    for raw in page.get_text("dict").get("blocks", []):
        if raw.get("type") != 0:  # 0 = text, 1 = image
            continue
        rect = pymupdf.Rect(raw["bbox"])
        if _overlaps_table(rect, table_rects):
            continue

        lines, max_size, bold = [], 0.0, False
        for line in raw.get("lines", []):
            parts = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    parts.append(text)
                    max_size = max(max_size, span.get("size", 0.0))
                    if span.get("flags", 0) & 16:  # bit 4 = bold
                        bold = True
            if parts:
                lines.append("".join(parts).strip())

        text = "\n".join(lines).strip()
        if not text:
            continue

        is_heading = (
            len(text) < 120
            and "\n" not in text
            and (max_size > body_size * 1.15 or (bold and max_size >= body_size))
        )
        if is_heading:
            # Bigger type means a shallower heading level.
            level = 1 if max_size > body_size * 1.4 else 2
            blocks.append(Block(kind="heading", text=text, page_no=page_no, level=level))
        else:
            blocks.append(Block(kind="para", text=text, page_no=page_no))
    return blocks


def _ocr_page(
    page: pymupdf.Page, ocr: OcrBackend, dpi: int
) -> tuple[list[Block], OcrResult]:
    pix = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    result = ocr.run(image)

    blocks: list[Block] = []
    page_no = page.number + 1
    # OCR output has no reliable structure, so split on blank lines only.
    for paragraph in (p.strip() for p in result.text.split("\n\n")):
        if paragraph:
            blocks.append(
                Block(
                    kind="para",
                    text=paragraph,
                    page_no=page_no,
                    meta={"ocr": True, "ocr_confidence": round(result.confidence, 1)},
                )
            )
    return blocks, result


def extract_pdf(
    path: str | Path, cfg, ocr: OcrBackend
) -> tuple[list[Block], list[str], list[int], int, str]:
    """Returns (blocks, warnings, ocr_page_numbers, n_pages, page1_raw_text)."""
    doc = pymupdf.open(path)
    min_chars = cfg.ocr.get("min_text_chars_per_page", 120)
    dpi = cfg.ocr.get("dpi", 300)

    all_sizes = [s for page in doc for s in _span_sizes(page)]
    body_size = statistics.median(all_sizes) if all_sizes else 10.0

    blocks: list[Block] = []
    warnings: list[str] = []
    ocr_pages: list[int] = []
    page1_text = ""
    # One routine context for the whole document: a day's grid continues
    # onto following pages without repeating its headers.
    routine_state = RoutineState()

    for page in doc:
        page_no = page.number + 1
        raw_text = page.get_text().strip()
        if page_no == 1:
            page1_text = raw_text

        table_rects: list[pymupdf.Rect] = []
        page_blocks: list[Block] = []
        try:
            finder = page.find_tables()
            for table in finder.tables:
                rows = table.extract()
                if not rows:
                    continue
                table_rects.append(pymupdf.Rect(table.bbox))
                page_blocks.extend(
                    table_to_blocks(rows, page_no, state=routine_state)
                )
        except Exception as exc:  # table detection is best-effort
            warnings.append(f"p{page_no}: table detection failed ({exc})")

        page_blocks.extend(_text_blocks(page, table_rects, body_size))

        if len(raw_text) < min_chars:
            # No usable text layer: this page is a scan.
            try:
                ocr_blocks, result = _ocr_page(page, ocr, dpi)
                if ocr_blocks:
                    page_blocks = ocr_blocks + [
                        b for b in page_blocks if b.kind in ("table", "table_fact")
                    ]
                    ocr_pages.append(page_no)
                    if 0 <= result.confidence < 60:
                        warnings.append(
                            f"p{page_no}: low OCR confidence "
                            f"({result.confidence:.0f}/100)"
                        )
                    if page_no == 1 and not page1_text:
                        page1_text = result.text
            except OcrUnavailable as exc:
                warnings.append(f"p{page_no}: scanned page not read - {exc}")
            except Exception as exc:
                warnings.append(f"p{page_no}: OCR failed ({exc})")

        blocks.extend(page_blocks)

    if routine_state.rejected_rows:
        warnings.append(
            f"{routine_state.rejected_rows} routine cells skipped: the value "
            f"in the course column was not a course code"
        )

    n_pages = len(doc)
    doc.close()
    return blocks, warnings, ocr_pages, n_pages, page1_text
