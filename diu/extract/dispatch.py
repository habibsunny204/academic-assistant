"""Route a file to the right extractor based on its extension."""
from __future__ import annotations

from pathlib import Path

from ..models import Block, DocMeta, ExtractedDoc
from .docx_ import extract_docx
from .image import extract_image
from .meta import derive_metadata
from .ocr import get_backend
from .pdf import extract_pdf

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _extract_text_file(path: Path) -> tuple[list[Block], list[str], str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [
        Block(kind="para", text=para.strip(), page_no=1)
        for para in text.split("\n\n")
        if para.strip()
    ]
    return blocks, [], text[:1500]


def extract_document(path: str | Path, cfg, doc_id: str, sha256: str) -> ExtractedDoc:
    path = Path(path)
    suffix = path.suffix.lower()
    ocr = get_backend(cfg)

    ocr_pages: list[int] = []
    n_pages = 1

    if suffix == ".pdf":
        blocks, warnings, ocr_pages, n_pages, head = extract_pdf(path, cfg, ocr)
        extractor = "pymupdf+tesseract" if ocr_pages else "pymupdf"
    elif suffix == ".docx":
        blocks, warnings, head = extract_docx(path, cfg)
        extractor = "python-docx"
    elif suffix in _IMAGE_EXTENSIONS:
        blocks, warnings, head = extract_image(path, cfg, ocr)
        extractor = "tesseract"
        if blocks:
            ocr_pages = [1]
    elif suffix in (".txt", ".md"):
        blocks, warnings, head = _extract_text_file(path)
        extractor = "plaintext"
    else:
        return ExtractedDoc(
            meta=DocMeta(
                doc_id=doc_id, source_path=str(path), filename=path.name,
                sha256=sha256, title=path.stem, extractor="none",
            ),
            warnings=[f"unsupported file type '{suffix}'"],
        )

    derived = derive_metadata(path.name, head)
    meta = DocMeta(
        doc_id=doc_id,
        source_path=str(path),
        filename=path.name,
        sha256=sha256,
        n_pages=n_pages,
        extractor=extractor,
        ocr_pages=ocr_pages,
        **derived,
    )
    if not (meta.issued_date or meta.effective_from):
        warnings.append("no date found; temporal ranking will fall back to file mtime")

    return ExtractedDoc(meta=meta, blocks=blocks, warnings=warnings)
