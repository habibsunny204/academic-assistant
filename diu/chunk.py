"""Structure-aware chunking.

Two ideas do most of the work here.

1. Chunks follow document structure. Prose accumulates under its heading path;
   a Markdown table is never split mid-row; routine facts are grouped by the
   day they belong to.

2. Every chunk is prefixed with a context header naming the document, semester
   and section. Without it, "Course CSE315 is scheduled on Saturday..." contains
   no evidence that it belongs to the CSE class routine for Fall 2026, so a
   question mentioning either would fail to retrieve it. The header is part of
   the embedded and displayed text, so the answer sees the context too.
"""
from __future__ import annotations

from .models import Block, Chunk, DocMeta


def _context_header(meta: DocMeta, section_path: str) -> str:
    parts = [meta.title]
    if meta.semester:
        parts.append(meta.semester)
    if meta.version_label:
        parts.append(meta.version_label)
    header = " | ".join(p for p in parts if p)
    if section_path:
        header = f"{header} > {section_path}"
    return header


def _chunk_meta(meta: DocMeta) -> dict:
    return {
        "title": meta.title,
        "source_path": meta.source_path,
        "filename": meta.filename,
        "doc_type": meta.doc_type,
        "category": meta.category,
        "semester": meta.semester or "",
        "version_label": meta.version_label or "",
        "effective_from": meta.effective_from.isoformat() if meta.effective_from else "",
        "issued_date": meta.issued_date.isoformat() if meta.issued_date else "",
    }


def _split_long_text(text: str, target: int, overlap: int) -> list[str]:
    """Split oversized prose on sentence-ish boundaries with a little overlap."""
    if len(text) <= target:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + target, len(text))
        if end < len(text):
            window = text.rfind(". ", start + target // 2, end)
            if window == -1:
                window = text.rfind("\n", start + target // 2, end)
            if window != -1:
                end = window + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


class _Builder:
    def __init__(self, meta: DocMeta, cfg):
        self.meta = meta
        self.target = cfg.chunking.get("target_chars", 1000)
        self.overlap = cfg.chunking.get("overlap_chars", 150)
        self.min_chars = cfg.chunking.get("min_chunk_chars", 80)
        self.rows_per_chunk = cfg.chunking.get("table_rows_per_chunk", 12)
        self.chunks: list[Chunk] = []
        self.base_meta = _chunk_meta(meta)

    def emit(self, body: str, section_path: str, page_start: int,
             page_end: int, kind: str, extra: dict | None = None) -> None:
        body = body.strip()
        if len(body) < self.min_chars and kind != "table_fact":
            return
        header = _context_header(self.meta, section_path)
        text = f"{header}\n\n{body}" if header else body
        ordinal = len(self.chunks)
        meta = dict(self.base_meta)
        meta["context_header"] = header
        if extra:
            meta.update(extra)
        self.chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(self.meta.doc_id, ordinal, text),
                doc_id=self.meta.doc_id,
                ord=ordinal,
                text=text,
                page_start=page_start,
                page_end=page_end,
                section_path=section_path,
                kind=kind,
                meta=meta,
            )
        )


def chunk_document(blocks: list[Block], meta: DocMeta, cfg) -> list[Chunk]:
    builder = _Builder(meta, cfg)
    heading_stack: list[tuple[int, str]] = []

    buffer: list[str] = []
    buf_start = buf_end = 0
    buf_section = ""

    def section_path() -> str:
        return " > ".join(text for _, text in heading_stack)

    def flush_prose() -> None:
        nonlocal buffer, buf_start, buf_end
        if not buffer:
            return
        joined = "\n\n".join(buffer).strip()
        for piece in _split_long_text(joined, builder.target, builder.overlap):
            builder.emit(piece, buf_section, buf_start, buf_end, "para")
        buffer = []

    index = 0
    while index < len(blocks):
        block = blocks[index]

        if block.kind == "heading":
            flush_prose()
            while heading_stack and heading_stack[-1][0] >= block.level:
                heading_stack.pop()
            heading_stack.append((block.level, block.text))
            index += 1
            continue

        if block.kind == "table_fact":
            flush_prose()
            # Group consecutive facts that share a grouping key (the weekday for
            # routines) so a chunk answers "what happens on Saturday?" coherently.
            group_key = block.meta.get("day") or block.meta.get("table_type", "")
            group: list[Block] = []
            while index < len(blocks) and blocks[index].kind == "table_fact":
                candidate = blocks[index]
                candidate_key = (
                    candidate.meta.get("day") or candidate.meta.get("table_type", "")
                )
                if candidate_key != group_key:
                    break
                group.append(candidate)
                index += 1

            for start in range(0, len(group), builder.rows_per_chunk):
                window = group[start:start + builder.rows_per_chunk]
                body = "\n".join(b.text for b in window)
                path = section_path()
                if group_key and group_key not in path:
                    path = f"{path} > {group_key}" if path else str(group_key)
                extra = {"group_key": str(group_key), "n_facts": len(window)}
                if window[0].meta.get("table_type") == "class_routine":
                    extra["day"] = window[0].meta.get("day", "")
                    extra["courses"] = ",".join(
                        sorted({b.meta.get("course_code", "") for b in window})
                    )
                builder.emit(
                    body, path, window[0].page_no, window[-1].page_no,
                    "table_fact", extra,
                )
            continue

        if block.kind == "table":
            flush_prose()
            lines = block.text.split("\n")
            if len(block.text) <= builder.target * 2 or len(lines) < 4:
                builder.emit(block.text, section_path(), block.page_no,
                             block.page_no, "table")
            else:
                # Repeat the header row in every slice so each stands alone.
                head, rows = lines[:2], lines[2:]
                step = builder.rows_per_chunk
                for start in range(0, len(rows), step):
                    body = "\n".join(head + rows[start:start + step])
                    builder.emit(body, section_path(), block.page_no,
                                 block.page_no, "table")
            index += 1
            continue

        # Ordinary prose.
        if not buffer:
            buf_start = block.page_no
            buf_section = section_path()
        buf_end = block.page_no
        buffer.append(block.text)
        if sum(len(b) for b in buffer) >= builder.target:
            flush_prose()
        index += 1

    flush_prose()
    return builder.chunks
