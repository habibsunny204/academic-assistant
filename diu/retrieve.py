"""The retrieval pipeline: query -> hybrid search -> rerank -> top-k.

This is the runtime path from the architecture diagram, minus the answering
step. It is kept separate from answer.py so retrieval can be evaluated on its
own -- recall@k and MRR are measurable here without spending an LLM call, which
matters when tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .config import Config
from .embed import Embedder
from .rerank import Reranker
from .store import VectorStore


@dataclass
class RetrievalResult:
    query: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    n_candidates: int = 0
    top_score: float = 0.0
    filters: dict[str, Any] = field(default_factory=dict)


class Retriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = Embedder(cfg)
        self.store = VectorStore(cfg, self.embedder.signature())
        self.reranker = Reranker(cfg)
        self.candidates = int(cfg.retrieval.get("candidates", 30))
        self.final_k = int(cfg.retrieval.get("final_k", 5))

    @staticmethod
    def _where(filters: dict[str, Any] | None) -> dict | None:
        filters = {k: v for k, v in (filters or {}).items() if v}
        if not filters:
            return None
        if len(filters) == 1:
            return filters
        return {"$and": [{k: v} for k, v in filters.items()]}

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        today: date | None = None,
    ) -> RetrievalResult:
        k = k or self.final_k
        vector = self.embedder.encode_query(query)
        candidates = self.store.hybrid_search(
            query,
            vector,
            k=self.candidates,
            candidates=self.candidates,
            where=self._where(filters),
        )
        ranked = self.reranker.rerank(query, candidates, top_k=k, today=today)
        top = ranked[0].get("rerank_score", 0.0) if ranked else 0.0
        return RetrievalResult(
            query=query,
            hits=ranked,
            n_candidates=len(candidates),
            top_score=float(top),
            filters=filters or {},
        )
