"""Embedding model wrapper.

BGE-family models expect an instruction prefix on *queries* but not on
documents. Getting that backwards quietly degrades retrieval, so the asymmetry
is handled here once rather than at every call site.
"""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, cfg):
        self.model_name = cfg.embedding.model
        self.dim = int(cfg.embedding.dim)
        self.batch_size = int(cfg.embedding.get("batch_size", 32))
        self.query_prefix = cfg.embedding.get("query_prefix", "")
        self.normalize = bool(cfg.embedding.get("normalize", True))
        self._model = None

    @property
    def model(self):
        # Loaded lazily so that CLI commands which do not embed (list, doctor)
        # start instantly instead of pulling in torch.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            # Renamed in sentence-transformers 5.x; keep both for compatibility.
            getter = getattr(
                self._model, "get_embedding_dimension",
                self._model.get_sentence_embedding_dimension,
            )
            actual = getter()
            if actual != self.dim:
                raise ValueError(
                    f"Model {self.model_name} produces {actual}-dim vectors but "
                    f"config.yaml declares embedding.dim={self.dim}. "
                    f"Fix the config, then re-index."
                )
        return self._model

    def encode_documents(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            [self.query_prefix + query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vector, dtype=np.float32)[0]

    def signature(self) -> str:
        """Identifies the vector space. A change here invalidates the index."""
        return f"{self.model_name}@{self.dim}"
