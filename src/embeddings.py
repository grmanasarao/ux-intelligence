from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEngine:
    """Loads the embedding model once and provides normalized embeddings."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.array([], dtype=np.float32)
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        return np.matmul(matrix, query_vector.T).ravel()
