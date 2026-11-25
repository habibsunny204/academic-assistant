"""Standalone image files (JPG/PNG notices photographed or screenshotted)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..models import Block
from .ocr import OcrBackend, OcrUnavailable


def extract_image(
    path: str | Path, cfg, ocr: OcrBackend
) -> tuple[list[Block], list[str], str]:
    warnings: list[str] = []
    try:
        image = Image.open(path)
    except Exception as exc:
        return [], [f"could not open image ({exc})"], ""

    try:
        result = ocr.run(image)
    except OcrUnavailable as exc:
        return [], [f"image needs OCR - {exc}"], ""
    except Exception as exc:
        return [], [f"OCR failed ({exc})"], ""

    if 0 <= result.confidence < 60:
        warnings.append(f"low OCR confidence ({result.confidence:.0f}/100)")

    blocks = [
        Block(
            kind="para",
            text=paragraph.strip(),
            page_no=1,
            meta={"ocr": True, "ocr_confidence": round(result.confidence, 1)},
        )
        for paragraph in result.text.split("\n\n")
        if paragraph.strip()
    ]
    return blocks, warnings, result.text[:1500]
