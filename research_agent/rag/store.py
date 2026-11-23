"""Vector stores. Both use cosine similarity on pre-normalised vectors."""

from __future__ import annotations

import uuid
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    name: str

    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def close(self) -> None: ...


class InMemoryStore:
    """Brute-force numpy search. Exact, and plenty fast for the few hundred chunks per run."""

    name = "memory"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vecs: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        self._ids.extend(ids)
        self._vecs = vectors if self._vecs is None else np.vstack([self._vecs, vectors])

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._vecs is None or not self._ids:
            return []
        scores = self._vecs @ vector
        top = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in top]

    def close(self) -> None:
        self._ids, self._vecs = [], None


class ChromaStore:
    """ChromaDB in-memory collection (one per worker, dropped when the worker finishes)."""

    name = "chroma"

    def __init__(self) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
        self._collection_name = f"run-{uuid.uuid4().hex[:12]}"
        self._col = self._client.create_collection(self._collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if ids:
            self._col.add(ids=ids, embeddings=[v.tolist() for v in vectors])

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        count = self._col.count()
        if count == 0:
            return []
        res = self._col.query(query_embeddings=[vector.tolist()], n_results=min(k, count))
        ids, dists = res["ids"][0], res["distances"][0]
        return [(i, 1.0 - float(d)) for i, d in zip(ids, dists)]  # cosine distance -> similarity

    def close(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001
            pass


def make_store(kind: str = "auto") -> VectorStore:
    kind = kind.lower()
    if kind in ("chroma", "auto"):
        try:
            return ChromaStore()
        except Exception:
            if kind == "chroma":
                raise
    return InMemoryStore()
