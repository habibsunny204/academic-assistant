"""Application service: the one object the HTTP layer talks to.

It owns the long-lived components (models, vector store, registry, history)
so they load once per process, and it serialises index writes against reads.
Routes in app.py stay thin and only translate between HTTP and this class.
"""
from __future__ import annotations

import json
import mimetypes
import re
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..answer import Answer, Answerer
from ..config import Config, ensure_dirs
from ..extract import SUPPORTED_EXTENSIONS
from ..ingest import ingest, ingest_file
from ..llm import LLMUnavailable, get_client
from ..registry import Registry
from .history import HistoryStore


class ServiceError(Exception):
    """An error with an HTTP status and a stable machine-readable code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class ReadWriteLock:
    """Many concurrent questions, or one index update, never both.

    Re-indexing swaps Chroma rows and the in-memory BM25 index in several steps.
    A question answered halfway through would retrieve from a mix of old and
    new chunks, and could cite a chunk that no longer exists.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writing = False

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writing:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            while self._writing or self._readers:
                self._cond.wait()
            self._writing = True
        try:
            yield
        finally:
            with self._cond:
                self._writing = False
                self._cond.notify_all()


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _safe_filename(name: str) -> str:
    """Strip any directory component and characters unsafe on Windows."""
    base = Path(name.replace("\\", "/")).name
    base = _SAFE_NAME_RE.sub("-", base).strip(" .-")
    return base


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AssistantService:
    UPLOAD_SUBDIR = "uploads"

    def __init__(self, cfg: Config, answerer: Answerer | None = None):
        ensure_dirs(cfg)
        self.cfg = cfg
        self.registry = Registry(cfg.paths["registry_db"])
        self.answerer = answerer or Answerer(cfg)
        self.retriever = self.answerer.retriever
        self.history = HistoryStore(cfg.paths["app_db"])
        self.lock = ReadWriteLock()
        self.sources_root = Path(cfg.paths["sources"]).resolve()
        self.max_upload_bytes = int(cfg.api.get("max_upload_mb", 50)) * 1024 * 1024

    def warm_up(self) -> None:
        """Load the embedding and reranking models before the first request."""
        _ = self.retriever.embedder.model
        if self.retriever.reranker.enabled:
            _ = self.retriever.reranker.model

    # -- status ---------------------------------------------------------------

    def llm_status(self) -> dict[str, Any]:
        provider = self.cfg.answer.get("provider", "groq")
        model = self.cfg.answer.get("model", "")
        try:
            if self.answerer._client is None:
                self.answerer._client = get_client(self.cfg)
            return {"configured": True, "provider": provider,
                    "model": getattr(self.answerer._client, "model", model),
                    "message": ""}
        except LLMUnavailable as exc:
            return {"configured": False, "provider": provider, "model": model,
                    "message": str(exc)}

    def index_status(self) -> dict[str, Any]:
        stats = self.registry.stats()
        recorded = self.registry.get_index_meta("embedding_signature")
        expected = self.retriever.embedder.signature()
        if recorded is None:
            state, message = "empty", "The index is empty. Run 'python cli.py ingest'."
        elif recorded != expected:
            state = "incompatible"
            message = (f"Index was built with {recorded} but the server uses "
                       f"{expected}. Run 'python cli.py ingest --reset'.")
        else:
            state, message = "ready", ""
        return {
            "state": state,
            "message": message,
            "documents": stats["by_status"].get("indexed", 0),
            "quarantined": stats["by_status"].get("quarantined", 0),
            "chunks": stats["chunks"],
            "pages": stats["pages"],
            "last_updated": self.registry.last_ingested_at(),
            "embedding_model": expected,
        }

    def health(self) -> dict[str, Any]:
        index = self.index_status()
        llm = self.llm_status()
        if index["state"] != "ready":
            status = "degraded"
        elif not llm["configured"]:
            status = "degraded"
        else:
            status = "ok"
        return {"status": status, "index": index, "llm": llm,
                "feedback": self.history.feedback_summary()}

    def categories(self) -> list[dict[str, Any]]:
        by_category = self.registry.stats()["by_category"]
        return [{"name": name, "documents": count}
                for name, count in sorted(by_category.items())]

    # -- answering ------------------------------------------------------------

    def _serialise(self, answer: Answer, category: str | None) -> dict[str, Any]:
        citations = []
        for c in answer.citations:
            suffix = Path(c.filename).suffix.lower().lstrip(".")
            citations.append({
                "n": c.n,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "title": c.title,
                "filename": c.filename,
                "file_type": suffix.upper() or "FILE",
                "page": c.page or None,
                "section": c.section,
                "doc_date": c.doc_date or None,
                "category": c.category,
                "relevance": round(c.score, 3),
                "url": f"/api/documents/{c.doc_id}/file" if c.doc_id else None,
            })
        return {
            "question": answer.question,
            "category": category,
            "answer": answer.text,
            "abstained": answer.abstained,
            "reason": answer.reason,
            "citations": citations,
            "support": [asdict(s) for s in answer.support],
            "warnings": answer.warnings,
            "details": {
                "provider": self.cfg.answer.get("provider", "groq"),
                "model": (getattr(self.answerer._client, "model", None)
                          if answer.llm_called else None),
                "llm_called": answer.llm_called,
                "response_time_s": round(answer.elapsed_s, 2),
                "retrieved_chunks": answer.n_retrieved,
                "confidence": answer.confidence,
                "confidence_band": answer.confidence_band,
                "top_relevance": answer.top_score,
                "answered_at": _now(),
            },
        }

    def ask(
        self,
        question: str,
        category: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        question = " ".join((question or "").split())
        if not question:
            raise ServiceError(400, "empty_question", "Please type a question.")
        if len(question) > 1000:
            raise ServiceError(400, "question_too_long",
                               "Questions are limited to 1000 characters.")

        index = self.index_status()
        if index["state"] != "ready":
            raise ServiceError(409, f"index_{index['state']}", index["message"])

        filters = {"category": category} if category else None
        with self.lock.read():
            try:
                answer = self.answerer.ask(question, filters=filters)
            except LLMUnavailable as exc:
                raise ServiceError(503, "llm_unavailable", str(exc)) from exc

        response = self._serialise(answer, category)
        conversation_id, message_id = self.history.record(
            question, response, category, conversation_id
        )
        return {**response, "message_id": message_id,
                "conversation_id": conversation_id, "feedback": None}

    # -- documents ------------------------------------------------------------

    def documents(self) -> list[dict[str, Any]]:
        rows = []
        for doc in self.registry.list_documents():
            rows.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "filename": doc["filename"],
                "file_type": Path(doc["filename"]).suffix.lstrip(".").upper(),
                "doc_type": doc["doc_type"],
                "category": doc["category"],
                "semester": doc["semester"],
                "version_label": doc["version_label"],
                "doc_date": doc["issued_date"] or doc["effective_from"],
                "status": doc["status"],
                "pages": doc["n_pages"],
                "chunks": doc["n_chunks"],
                "ocr_pages": json.loads(doc["ocr_pages"] or "[]"),
                "warnings": json.loads(doc["warnings"] or "[]"),
                "ingested_at": doc["ingested_at"],
                "url": f"/api/documents/{doc['doc_id']}/file",
            })
        return rows

    def document_file(self, doc_id: str) -> tuple[Path, str]:
        doc = self.registry.get_document(doc_id)
        if not doc:
            raise ServiceError(404, "document_not_found", "No such document.")
        path = Path(doc["source_path"]).resolve()
        # Only ever serve files that live under Sources/, whatever the registry
        # says: a tampered or stale path must not expose the rest of the disk.
        if not path.is_relative_to(self.sources_root) or not path.is_file():
            raise ServiceError(404, "file_missing",
                               "The source file is no longer available.")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path, media_type

    def upload(self, filename: str, data: bytes, replace: bool = False) -> dict[str, Any]:
        safe = _safe_filename(filename or "")
        suffix = Path(safe).suffix.lower()
        if not safe or not Path(safe).stem:
            raise ServiceError(400, "invalid_filename", "The file has no usable name.")
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ServiceError(
                415, "unsupported_type",
                f"'{suffix or 'no extension'}' is not supported. Accepted: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            )
        if not data:
            raise ServiceError(400, "empty_file", "The uploaded file is empty.")
        if len(data) > self.max_upload_bytes:
            raise ServiceError(413, "file_too_large",
                               f"Files are limited to {self.cfg.api.get('max_upload_mb', 50)} MB.")

        target_dir = self.sources_root / self.UPLOAD_SUBDIR
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / safe

        with self.lock.write():
            # Checked under the lock so two simultaneous uploads of the same
            # name cannot both pass the check.
            if target.exists() and not replace:
                raise ServiceError(409, "file_exists",
                                   f"'{safe}' already exists. Upload again with "
                                   f"replace enabled to update it.")
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(data)
            partial.replace(target)

            result = ingest_file(
                target, self.cfg, self.registry,
                self.retriever.store, self.retriever.embedder, force=True,
            )
            self.registry.set_index_meta(
                "embedding_signature", self.retriever.embedder.signature()
            )
            self.retriever.store.build_bm25(self.registry.all_chunks())

        return {
            "filename": safe,
            "status": result.status,
            "title": result.title,
            "category": result.category,
            "chunks": result.n_chunks,
            "reason": result.reason,
            "warnings": result.warnings,
        }

    def reindex(self, force: bool = False) -> dict[str, Any]:
        with self.lock.write():
            try:
                report = ingest(
                    self.cfg, force=force,
                    registry=self.registry,
                    store=self.retriever.store,
                    embedder=self.retriever.embedder,
                )
            except RuntimeError as exc:
                raise ServiceError(409, "reindex_refused", str(exc)) from exc
        return {
            "counts": report.counts(),
            "pruned": report.pruned,
            "vectors": report.vector_count,
            "results": [
                {"filename": Path(r.path).name, "status": r.status,
                 "title": r.title, "chunks": r.n_chunks, "reason": r.reason,
                 "warnings": r.warnings}
                for r in report.results
            ],
        }
