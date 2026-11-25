"""Vector store and lexical index.

Chroma holds the dense vectors; rank_bm25 provides the lexical half of hybrid
retrieval. Chroma is used in preference to FAISS because ingestion is
continuous: documents are revised every semester, and FAISS offers no way to
delete a superseded document's vectors by metadata.

The two result lists are merged with Reciprocal Rank Fusion, which combines
rankings rather than scores and so needs no score calibration between a cosine
similarity and a BM25 score.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

COLLECTION_NAME = "diu_chunks"
# Underscores are kept: section labels like "67_B" and teacher codes like
# "FNT_4" are exactly the rare, high-IDF tokens lexical search needs. Splitting
# them into "67" and "b" throws that signal away.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
RRF_K = 60  # standard damping constant; larger values flatten rank influence


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _sanitize(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata accepts only str, int, float and bool."""
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (list, tuple, set)):
            clean[key] = ",".join(str(v) for v in value)
        else:
            clean[key] = str(value)
    return clean


class VectorStore:
    def __init__(self, cfg, signature: str):
        import chromadb
        from chromadb.config import Settings

        self.dir = Path(cfg.paths["vector_dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self.client = chromadb.PersistentClient(
            path=str(self.dir), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embedding_signature": signature},
        )
        self._bm25_path = self.dir / "bm25.pkl"
        self._bm25 = None
        self._bm25_ids: list[str] = []

    # -- writes ---------------------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def upsert_chunks(self, chunks, vectors: np.ndarray) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.meta)
            meta.update(
                {
                    "doc_id": chunk.doc_id,
                    "ord": chunk.ord,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_path": chunk.section_path,
                    "kind": chunk.kind,
                }
            )
            metadatas.append(_sanitize(meta))

        # Chroma rejects very large single batches; 512 is comfortably safe.
        step = 512
        for start in range(0, len(ids), step):
            stop = start + step
            self.collection.upsert(
                ids=ids[start:stop],
                embeddings=vectors[start:stop].tolist(),
                documents=documents[start:stop],
                metadatas=metadatas[start:stop],
            )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embedding_signature": self.signature},
        )
        if self._bm25_path.exists():
            self._bm25_path.unlink()
        self._bm25, self._bm25_ids = None, []

    # -- lexical index --------------------------------------------------------

    def build_bm25(self, chunk_rows: list[dict[str, Any]]) -> int:
        """Rebuild the BM25 index from the registry's chunk texts."""
        from rank_bm25 import BM25Okapi

        ids = [row["chunk_id"] for row in chunk_rows]
        corpus = [tokenize(row["text"]) for row in chunk_rows]
        if not corpus:
            if self._bm25_path.exists():
                self._bm25_path.unlink()
            self._bm25, self._bm25_ids = None, []
            return 0
        self._bm25 = BM25Okapi(corpus)
        self._bm25_ids = ids
        with open(self._bm25_path, "wb") as fh:
            pickle.dump({"ids": ids, "corpus": corpus}, fh)
        return len(ids)

    def _load_bm25(self) -> bool:
        if self._bm25 is not None:
            return True
        if not self._bm25_path.exists():
            return False
        from rank_bm25 import BM25Okapi

        with open(self._bm25_path, "rb") as fh:
            payload = pickle.load(fh)
        self._bm25_ids = payload["ids"]
        self._bm25 = BM25Okapi(payload["corpus"])
        return True

    # -- reads ----------------------------------------------------------------

    def dense_search(
        self, vector: np.ndarray, k: int, where: dict | None = None
    ) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for rank, chunk_id in enumerate(result["ids"][0]):
            distance = result["distances"][0][rank]
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": result["documents"][0][rank],
                    "meta": result["metadatas"][0][rank],
                    "score": 1.0 - float(distance),  # cosine distance -> similarity
                    "rank": rank + 1,
                    "source": "dense",
                }
            )
        return hits

    def sparse_search(
        self, query: str, k: int, where: dict | None = None
    ) -> list[dict[str, Any]]:
        if not self._load_bm25():
            return []
        scores = self._bm25.get_scores(tokenize(query))
        if not len(scores):
            return []
        # With a filter, over-fetch and then keep only chunks that satisfy it.
        # BM25 knows nothing about metadata, so without this step a filtered
        # hybrid search would still let lexical hits from other categories in.
        pool = k * 10 if where else k
        order = [int(i) for i in np.argsort(scores)[::-1][:pool] if scores[i] > 0]
        if where and order:
            ids = [self._bm25_ids[i] for i in order]
            allowed = set(self.collection.get(ids=ids, where=where, include=[])["ids"])
            order = [i for i in order if self._bm25_ids[i] in allowed]
        return [
            {
                "chunk_id": self._bm25_ids[i],
                "score": float(scores[i]),
                "rank": rank + 1,
                "source": "sparse",
            }
            for rank, i in enumerate(order[:k])
        ]

    def hybrid_search(
        self,
        query: str,
        vector: np.ndarray,
        k: int = 10,
        candidates: int = 30,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        dense = self.dense_search(vector, candidates, where)
        sparse = self.sparse_search(query, candidates, where)

        fused: dict[str, dict[str, Any]] = {}
        for hits in (dense, sparse):
            for hit in hits:
                entry = fused.setdefault(
                    hit["chunk_id"],
                    {"chunk_id": hit["chunk_id"], "rrf": 0.0,
                     "dense_score": None, "sparse_score": None,
                     "text": hit.get("text"), "meta": hit.get("meta")},
                )
                entry["rrf"] += 1.0 / (RRF_K + hit["rank"])
                entry[f"{hit['source']}_score"] = hit["score"]
                if hit.get("text") and not entry.get("text"):
                    entry["text"] = hit["text"]
                if hit.get("meta") and not entry.get("meta"):
                    entry["meta"] = hit["meta"]

        # BM25-only hits carry no text or metadata; fetch them from Chroma.
        missing = [cid for cid, e in fused.items() if not e.get("text")]
        if missing:
            got = self.collection.get(ids=missing, include=["documents", "metadatas"])
            for i, cid in enumerate(got["ids"]):
                fused[cid]["text"] = got["documents"][i]
                fused[cid]["meta"] = got["metadatas"][i]

        ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
        return ranked[:k]
