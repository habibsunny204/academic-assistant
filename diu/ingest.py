"""Ingestion orchestrator.

The pipeline per file is: hash -> skip-if-unchanged -> extract -> clean ->
quality gate -> chunk -> embed -> upsert.

Two rules make this safe to run on a schedule:

* Idempotency. A file's doc_id comes from its path, so re-ingesting a revised
  document replaces its chunks in both the registry and the vector store
  instead of leaving stale copies behind to be retrieved and cited.

* The quality gate. A document that yields too little text after every
  extraction attempt is quarantined rather than indexed. An empty or garbled
  chunk that reaches retrieval becomes a confident, well-formatted, wrong
  citation, which is the exact failure this project exists to avoid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .chunk import chunk_document
from .clean import clean_blocks
from .config import Config, ensure_dirs
from .embed import Embedder
from .extract import SUPPORTED_EXTENSIONS, extract_document
from .models import Chunk
from .registry import Registry, doc_id_for, sha256_of
from .store import VectorStore


@dataclass
class FileResult:
    path: str
    status: str            # indexed | unchanged | quarantined | failed | skipped
    n_chunks: int = 0
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    title: str = ""
    category: str = ""


@dataclass
class IngestReport:
    results: list[FileResult] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    bm25_size: int = 0
    vector_count: int = 0

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for result in self.results:
            tally[result.status] = tally.get(result.status, 0) + 1
        return tally


def discover_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    files = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith("~$")  # Word lock files
    ]
    return files


def _quarantine(cfg: Config, path: Path, reason: str, detail: dict) -> None:
    """Record why a document could not be indexed, for manual follow-up."""
    out_dir = Path(cfg.paths["quarantine"])
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "source_path": str(path),
        "filename": path.name,
        "reason": reason,
        "quarantined_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **detail,
    }
    target = out_dir / f"{path.stem}.json"
    target.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def ingest_file(
    path: Path,
    cfg: Config,
    registry: Registry,
    store: VectorStore,
    embedder: Embedder,
    force: bool = False,
    show_progress: bool = False,
) -> FileResult:
    sources_root = cfg.paths["sources"]
    doc_id = doc_id_for(path, sources_root)

    try:
        digest = sha256_of(path)
    except OSError as exc:
        return FileResult(str(path), "failed", reason=f"unreadable: {exc}")

    existing = registry.get_document(doc_id)
    if existing and existing["sha256"] == digest and not force:
        if existing["status"] == "indexed":
            return FileResult(str(path), "unchanged",
                              n_chunks=existing["n_chunks"],
                              title=existing["title"] or "",
                              category=existing["category"] or "")
        if existing["status"] == "quarantined":
            return FileResult(str(path), "quarantined",
                              reason="unchanged since last quarantine",
                              title=existing["title"] or "")

    try:
        extracted = extract_document(path, cfg, doc_id, digest)
    except Exception as exc:
        return FileResult(str(path), "failed", reason=f"extraction error: {exc}")

    extracted.blocks = clean_blocks(extracted.blocks)
    meta = extracted.meta
    warnings = list(extracted.warnings)

    min_chars = cfg.quality.get("min_doc_chars", 200)
    if extracted.total_chars < min_chars:
        reason = (
            f"only {extracted.total_chars} characters extracted "
            f"(minimum {min_chars})"
        )
        _quarantine(cfg, path, reason,
                    {"warnings": warnings, "extractor": meta.extractor})
        store.delete_document(doc_id)  # drop any previously indexed version
        registry.upsert_document(meta, "quarantined", 0, warnings + [reason])
        registry.replace_chunks(doc_id, [])
        return FileResult(str(path), "quarantined", reason=reason,
                          warnings=warnings, title=meta.title,
                          category=meta.category)

    chunks: list[Chunk] = chunk_document(extracted.blocks, meta, cfg)
    if not chunks:
        reason = "extraction produced text but no chunks survived filtering"
        _quarantine(cfg, path, reason, {"warnings": warnings})
        store.delete_document(doc_id)
        registry.upsert_document(meta, "quarantined", 0, warnings + [reason])
        registry.replace_chunks(doc_id, [])
        return FileResult(str(path), "quarantined", reason=reason,
                          warnings=warnings, title=meta.title)

    for chunk in chunks:
        chunk.meta.setdefault("source_path", meta.source_path)

    vectors = embedder.encode_documents([c.text for c in chunks], show_progress)

    # Replace rather than add, so a revised document never coexists with its
    # own previous version in the index.
    store.delete_document(doc_id)
    store.upsert_chunks(chunks, vectors)

    mtime = datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    ).isoformat(timespec="seconds")
    # The document row must exist before its chunks: chunks.doc_id is a
    # foreign key into documents.
    registry.upsert_document(meta, "indexed", len(chunks), warnings, mtime)
    registry.replace_chunks(doc_id, chunks)

    return FileResult(str(path), "indexed", n_chunks=len(chunks),
                      warnings=warnings, title=meta.title, category=meta.category)


def ingest(
    cfg: Config,
    target: str | Path | None = None,
    force: bool = False,
    reset: bool = False,
    prune: bool = True,
    progress_callback=None,
    registry: Registry | None = None,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> IngestReport:
    """Ingest Sources/ (or `target`).

    A long-running process such as the API server passes its own registry,
    store and embedder so ingestion updates the index it is already serving
    from, instead of opening a second Chroma client and loading the model twice.
    """
    ensure_dirs(cfg)
    registry = registry or Registry(cfg.paths["registry_db"])
    embedder = embedder or Embedder(cfg)
    signature = embedder.signature()

    recorded = registry.get_index_meta("embedding_signature")
    if recorded and recorded != signature and not reset:
        raise RuntimeError(
            f"Index was built with '{recorded}' but config specifies "
            f"'{signature}'. Vectors from different models are not comparable. "
            f"Re-run with --reset to rebuild the index."
        )

    store = store or VectorStore(cfg, signature)
    if reset:
        store.reset()
        for doc in registry.list_documents():
            registry.delete_document(doc["doc_id"])
    registry.set_index_meta("embedding_signature", signature)

    root = Path(target) if target else Path(cfg.paths["sources"])
    files = discover_files(root)

    report = IngestReport()
    seen_doc_ids: set[str] = set()

    for index, path in enumerate(files, start=1):
        if progress_callback:
            progress_callback(index, len(files), path)
        seen_doc_ids.add(doc_id_for(path, cfg.paths["sources"]))
        report.results.append(
            ingest_file(path, cfg, registry, store, embedder, force=force)
        )

    # Documents whose source file has disappeared must leave the index too,
    # otherwise the assistant keeps citing a document nobody can open.
    if prune and target is None:
        for doc_id in registry.known_doc_ids() - seen_doc_ids:
            doc = registry.get_document(doc_id)
            store.delete_document(doc_id)
            registry.delete_document(doc_id)
            report.pruned.append(doc["source_path"] if doc else doc_id)

    report.bm25_size = store.build_bm25(registry.all_chunks())
    report.vector_count = store.count()
    return report
