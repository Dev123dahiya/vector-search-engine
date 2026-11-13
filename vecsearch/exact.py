from __future__ import annotations

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class ExactIndex:
    """Brute-force search. O(n) per query, always correct: the ground truth for recall."""

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.dim, self.metric = dim, metric
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        if self.metric == "cosine":
            vectors = normalize(vectors)
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """The k nearest ids with their distances, closest first."""
        query = np.asarray(query, dtype=np.float32)
        if self.metric == "cosine":
            dists = 1.0 - self._vectors @ normalize(query)
        else:
            dists = np.linalg.norm(self._vectors - query, axis=1)
        k = min(k, len(dists))
        top = np.argpartition(dists, k - 1)[:k] if k < len(dists) else np.arange(len(dists))
        top = top[np.argsort(dists[top], kind="stable")]
        return [(int(i), float(dists[i])) for i in top]
