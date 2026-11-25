"""SQLite registry: the authoritative record of what has been ingested.

The vector store holds embeddings; this holds truth. Keeping chunk text here
means the BM25 index can be rebuilt at any time, a document can be re-indexed
after an embedding-model change without re-parsing the PDF, and "what exactly
did we index from this file?" is always answerable.

Ingestion is idempotent. doc_id is derived from the source path, so re-running
against an updated file supersedes the old chunks rather than duplicating them.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import Chunk, DocMeta

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    filename        TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    title           TEXT,
    doc_type        TEXT,
    category        TEXT,
    semester        TEXT,
    version_label   TEXT,
    effective_from  TEXT,
    issued_date     TEXT,
    n_pages         INTEGER DEFAULT 0,
    n_chunks        INTEGER DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'indexed',
    extractor       TEXT,
    ocr_pages       TEXT,
    warnings        TEXT,
    ingested_at     TEXT,
    file_mtime      TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ord          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    page_start   INTEGER,
    page_end     INTEGER,
    section_path TEXT,
    kind         TEXT,
    meta_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_sha  ON documents(sha256);

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def doc_id_for(source_path: str | Path, sources_root: str | Path) -> str:
    """Stable per-file identity, independent of content.

    Using the path (not the hash) means an updated version of the same file
    keeps its doc_id, so the old chunks can be replaced instead of accumulating.
    """
    try:
        relative = Path(source_path).resolve().relative_to(Path(sources_root).resolve())
    except ValueError:
        relative = Path(source_path).resolve()
    key = str(relative).replace("\\", "/").lower()
    return "d_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    chunk_size = 1048576
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class Registry:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- index-level settings -------------------------------------------------

    def get_index_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM index_meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_index_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO index_meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    # -- documents ------------------------------------------------------------

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_document(
        self,
        meta: DocMeta,
        status: str,
        n_chunks: int,
        warnings: list[str],
        file_mtime: str = "",
    ) -> None:
        payload = (
            meta.doc_id, meta.source_path, meta.filename, meta.sha256, meta.title,
            meta.doc_type, meta.category, meta.semester, meta.version_label,
            meta.effective_from.isoformat() if meta.effective_from else None,
            meta.issued_date.isoformat() if meta.issued_date else None,
            meta.n_pages, n_chunks, status, meta.extractor,
            json.dumps(meta.ocr_pages), json.dumps(warnings),
            datetime.now(timezone.utc).isoformat(timespec="seconds"), file_mtime,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, source_path, filename, sha256, title, doc_type,
                    category, semester, version_label, effective_from, issued_date,
                    n_pages, n_chunks, status, extractor, ocr_pages, warnings,
                    ingested_at, file_mtime
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path=excluded.source_path, filename=excluded.filename,
                    sha256=excluded.sha256, title=excluded.title,
                    doc_type=excluded.doc_type, category=excluded.category,
                    semester=excluded.semester, version_label=excluded.version_label,
                    effective_from=excluded.effective_from,
                    issued_date=excluded.issued_date, n_pages=excluded.n_pages,
                    n_chunks=excluded.n_chunks, status=excluded.status,
                    extractor=excluded.extractor, ocr_pages=excluded.ocr_pages,
                    warnings=excluded.warnings, ingested_at=excluded.ingested_at,
                    file_mtime=excluded.file_mtime
                """,
                payload,
            )

    def replace_chunks(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Delete this document's previous chunks, then insert the new ones."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                "INSERT INTO chunks (chunk_id, doc_id, ord, text, page_start, "
                "page_end, section_path, kind, meta_json) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (c.chunk_id, c.doc_id, c.ord, c.text, c.page_start, c.page_end,
                     c.section_path, c.kind, json.dumps(c.meta, default=str))
                    for c in chunks
                ],
            )

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def list_documents(self, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM documents"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY category, title"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def all_chunks(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.* FROM chunks c JOIN documents d USING(doc_id) "
                "WHERE d.status = 'indexed' ORDER BY c.doc_id, c.ord"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Chunk rows by id, with their document's filename and title."""
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT c.*, d.filename, d.title FROM chunks c JOIN documents d "
                f"USING(doc_id) WHERE c.chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    def known_doc_ids(self) -> set[str]:
        with self._connect() as conn:
            return {r["doc_id"] for r in conn.execute("SELECT doc_id FROM documents")}

    def last_ingested_at(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(ingested_at) AS latest FROM documents "
                "WHERE status = 'indexed'"
            ).fetchone()
        return row["latest"] if row else None

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            by_status = conn.execute(
                "SELECT status, COUNT(*) n FROM documents GROUP BY status"
            ).fetchall()
            n_chunks = conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
            n_pages = conn.execute(
                "SELECT COALESCE(SUM(n_pages), 0) n FROM documents "
                "WHERE status = 'indexed'"
            ).fetchone()["n"]
            categories = conn.execute(
                "SELECT category, COUNT(*) n FROM documents "
                "WHERE status='indexed' GROUP BY category ORDER BY n DESC"
            ).fetchall()
        return {
            "by_status": {r["status"]: r["n"] for r in by_status},
            "chunks": n_chunks,
            "pages": n_pages,
            "by_category": {r["category"]: r["n"] for r in categories},
        }
