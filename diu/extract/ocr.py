"""Pluggable OCR.

OCR is the least reliable stage in the pipeline, so it reports a confidence
score alongside the text. Callers use that score to decide whether a page is
trustworthy enough to index -- silently indexing garbled text is worse than
not indexing the document at all.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from PIL import Image


class OcrUnavailable(RuntimeError):
    """Raised when no OCR backend is configured but a scanned page needs one."""


@dataclass
class OcrResult:
    text: str
    confidence: float  # mean per-word confidence, 0..100; -1.0 if unknown
    backend: str


class OcrBackend(Protocol):
    name: str

    def run(self, image: Image.Image) -> OcrResult: ...


def _preprocess(image: Image.Image) -> Image.Image:
    """Grayscale, and upscale small renders. Tesseract wants ~300 DPI input."""
    if image.mode != "L":
        image = image.convert("L")
    if min(image.size) < 1000:
        scale = 1000 / min(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    return image


class TesseractBackend:
    name = "tesseract"

    def __init__(self, cmd: str, tessdata_dir: str | None, langs: str = "eng"):
        import pytesseract

        self._pt = pytesseract
        if cmd and os.path.exists(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd
        if tessdata_dir and os.path.isdir(tessdata_dir):
            # Project-local traineddata, so adding languages needs no admin rights.
            os.environ["TESSDATA_PREFIX"] = tessdata_dir
        self.langs = langs

    def run(self, image: Image.Image) -> OcrResult:
        image = _preprocess(image)
        data = self._pt.image_to_data(
            image, lang=self.langs, output_type=self._pt.Output.DICT
        )
        words, confidences = [], []
        for word, conf in zip(data["text"], data["conf"]):
            word = (word or "").strip()
            if not word:
                continue
            words.append(word)
            try:
                value = float(conf)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                confidences.append(value)

        # image_to_data loses line breaks, so take layout from image_to_string
        # and keep image_to_data purely for the confidence signal.
        text = self._pt.image_to_string(image, lang=self.langs)
        mean_conf = sum(confidences) / len(confidences) if confidences else -1.0
        return OcrResult(text=text.strip(), confidence=mean_conf, backend=self.name)


class NullBackend:
    name = "none"

    def run(self, image: Image.Image) -> OcrResult:
        raise OcrUnavailable(
            "A scanned page needs OCR but ocr.backend is 'none'. "
            "Set ocr.backend to 'tesseract' in config.yaml."
        )


def get_backend(cfg) -> OcrBackend:
    kind = (cfg.ocr.get("backend") or "none").lower()
    if kind == "tesseract":
        return TesseractBackend(
            cmd=cfg.ocr.get("tesseract_cmd", ""),
            tessdata_dir=cfg.ocr.get("tessdata_dir"),
            langs=cfg.ocr.get("langs", "eng"),
        )
    return NullBackend()
