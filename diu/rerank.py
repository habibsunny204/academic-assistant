"""Cross-encoder reranking.

Hybrid retrieval is tuned for recall: it casts a wide net and accepts that the
best chunk may land at rank 3 rather than rank 1. A cross-encoder reads the
query and the chunk *together* rather than comparing two independently computed
vectors, so it can tell that "Where is CSE315 section 67_B on Saturday?" is
answered by the one chunk naming that section, not by the twelve other chunks
that merely talk about Saturday classes.

It is far too slow to run over the whole corpus, which is exactly why it runs
only over the candidates hybrid search already narrowed down to.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _recency_factor(meta: dict[str, Any], today: date) -> float:
    """1.0 for a document dated today, decaying to 0.0 over three years.

    Undated documents get a neutral 0.5 so they are neither promoted nor
    buried on the strength of missing metadata.
    """
    raw = meta.get("issued_date") or meta.get("effective_from") or ""
    if not raw:
        return 0.5
    try:
        doc_date = date.fromisoformat(str(raw)[:10])
    except ValueError:
        return 0.5
    age_days = (today - doc_date).days
    if age_days <= 0:
        return 1.0
    return max(0.0, 1.0 - age_days / (365 * 3))


class Reranker:
    def __init__(self, cfg):
        self.model_name = cfg.retrieval.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2")
        self.enabled = bool(cfg.retrieval.get("rerank", True))
        self.recency_weight = float(cfg.retrieval.get("recency_weight", 0.05))
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int,
        today: date | None = None,
    ) -> list[dict[str, Any]]:
        if not hits:
            return []
        if not self.enabled:
            return hits[:top_k]

        today = today or date.today()
        pairs = [(query, hit.get("text") or "") for hit in hits]
        raw_scores = np.asarray(self.model.predict(pairs), dtype=np.float32)
        relevance = _sigmoid(raw_scores)

        for hit, score in zip(hits, relevance):
            meta = hit.get("meta") or {}
            recency = _recency_factor(meta, today)
            hit["rerank_score"] = float(score)
            hit["recency"] = round(recency, 3)
            # Recency only breaks near-ties. A superseded notice should lose to
            # its replacement, but relevance must stay the dominant signal.
            hit["final_score"] = float(score) + self.recency_weight * recency

        ranked = sorted(hits, key=lambda h: h["final_score"], reverse=True)
        return ranked[:top_k]
