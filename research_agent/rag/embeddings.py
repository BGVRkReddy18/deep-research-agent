"""Embedding backends.

* GeminiEmbedder - free Gemini embedding API (default; no model in RAM, ideal for free hosting)
* LocalEmbedder  - sentence-transformers running on CPU (optional extra: `pip install .[local]`)
* HashEmbedder   - deterministic feature hashing; no network, used for --mock and tests
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Protocol

import numpy as np
from openai import AsyncOpenAI

from ..resilience import RateLimiter, estimate_tokens, retry_async


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


def _normalise(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class GeminiEmbedder:
    """Free-tier Gemini embeddings.

    The free tier counts *each text* as a request (~100/min) and caps tokens (~30k/min), so we
    pace by both, truncate long passages (the first ~300 tokens carry most of a chunk's meaning),
    and cache vectors so the same text is never embedded twice.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    BATCH = 32
    MAX_CHARS = 1200

    def __init__(
        self, api_key: str, model: str = "gemini-embedding-001", texts_per_minute: int = 90, tokens_per_minute: int = 25_000
    ) -> None:
        self.name = f"gemini:{model}"
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=self.BASE_URL, timeout=60.0, max_retries=0)
        self._text_limiter = RateLimiter(texts_per_minute)
        self._token_limiter = RateLimiter(tokens_per_minute)
        self._cache: dict[str, np.ndarray] = {}

    async def embed(self, texts: list[str]) -> np.ndarray:
        clipped = [t[: self.MAX_CHARS] for t in texts]
        missing = list(dict.fromkeys(t for t in clipped if t not in self._cache))
        for start in range(0, len(missing), self.BATCH):
            batch = missing[start : start + self.BATCH]

            async def call(batch: list[str] = batch) -> list[list[float]]:
                await self._text_limiter.acquire(len(batch))
                await self._token_limiter.acquire(sum(estimate_tokens(t) for t in batch))
                resp = await self._client.embeddings.create(model=self._model, input=batch)
                data = list(resp.data)
                if all(d.index is not None for d in data):
                    data.sort(key=lambda d: d.index)
                if len(data) != len(batch):
                    raise ValueError(f"expected {len(batch)} embeddings, got {len(data)}")
                return [d.embedding for d in data]

            vectors = _normalise(np.array(await retry_async(call, attempts=3, label="gemini-embed")))
            self._cache.update(zip(batch, vectors, strict=True))
        if not clipped:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self._cache[t] for t in clipped])


class LocalEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.name = f"local:{model}"
        self._model = SentenceTransformer(model)

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs = await asyncio.to_thread(self._model.encode, texts, normalize_embeddings=True)
        return _normalise(vecs)


_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Bag of words + bigrams hashed into a fixed-size vector. Crude, but offline and deterministic."""

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hash:{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return v * (1.0 / math.sqrt(len(feats) or 1))

    async def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim)))


def build_embedder(backend: str, *, gemini_key: str, gemini_model: str, local_model: str) -> Embedder:
    backend = backend.lower()
    if backend == "local":
        return LocalEmbedder(local_model)
    if backend == "gemini" and gemini_key:
        return GeminiEmbedder(gemini_key, gemini_model)
    return HashEmbedder()
