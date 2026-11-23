"""Embedding backends.

* OnnxEmbedder   - all-MiniLM-L6-v2 on CPU via ONNX Runtime (default; no API, no quota)
* GeminiEmbedder - Gemini embedding API (free tier: ~100 texts/min, ~1,000/day)
* LocalEmbedder  - any sentence-transformers model (optional extra: `pip install .[local]`)
* HashEmbedder   - deterministic feature hashing; no network, used for --mock and tests
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import threading
from typing import Protocol

import numpy as np
from openai import AsyncOpenAI

from ..resilience import RateLimiter, estimate_tokens, retry_async

log = logging.getLogger(__name__)


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
    MAX_CHARS = 800  # ~200 tokens: the opening of a passage carries most of its meaning
    MAX_RETRY_WAIT = 8.0  # embeddings are optional (BM25 is the fallback), so never stall long

    # Quota trackers and vector cache are process-wide, keyed by (model, key), because the API's
    # quota is shared by every research run in this process (e.g. all visitors of the web app).
    _shared: dict[tuple[str, str], tuple[RateLimiter, RateLimiter, dict[str, np.ndarray]]] = {}

    def __init__(
        self, api_key: str, model: str = "gemini-embedding-001", texts_per_minute: int = 90, tokens_per_minute: int = 25_000
    ) -> None:
        self.name = f"gemini:{model}"
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=self.BASE_URL, timeout=30.0, max_retries=0)
        key = (model, hashlib.sha256(api_key.encode()).hexdigest()[:16])
        if key not in self._shared:
            self._shared[key] = (RateLimiter(texts_per_minute), RateLimiter(tokens_per_minute), {})
        self._text_limiter, self._token_limiter, self._cache = self._shared[key]

    def _missing(self, texts: list[str]) -> list[str]:
        return list(dict.fromkeys(t[: self.MAX_CHARS] for t in texts if t[: self.MAX_CHARS] not in self._cache))

    def wait_time(self, texts: list[str]) -> float:
        """Seconds the free-tier quota would make us wait to embed `texts` (0 = right now)."""
        missing = self._missing(texts)
        if not missing:
            return 0.0
        return max(
            self._text_limiter.wait_time(len(missing)),
            self._token_limiter.wait_time(sum(estimate_tokens(t) for t in missing)),
        )

    async def embed(self, texts: list[str]) -> np.ndarray:
        clipped = [t[: self.MAX_CHARS] for t in texts]
        missing = self._missing(texts)
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

            vectors = _normalise(
                np.array(await retry_async(call, attempts=2, max_wait=self.MAX_RETRY_WAIT, label="gemini-embed"))
            )
            self._cache.update(zip(batch, vectors, strict=True))
        if not clipped:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self._cache[t] for t in clipped])


class OnnxEmbedder:
    """all-MiniLM-L6-v2 running locally via ONNX Runtime (bundled with ChromaDB).

    The default backend: no API key, no quota, ~2s for 64 passages on a laptop CPU. Why not an
    embedding API? Gemini's free tier counts every text as a request and allows only ~1,000 a
    day, about a dozen reports. The model (~80 MB) downloads once; while it loads, `wait_time`
    reports it as busy so the retriever uses BM25 instead of stalling a visitor's request.
    """

    name = "onnx:all-MiniLM-L6-v2"
    MAX_CHARS = 1000  # the model reads at most 256 word-pieces anyway

    _fn = None
    _lock = threading.Lock()
    _loader: threading.Thread | None = None
    load_error: str | None = None

    @classmethod
    def _load(cls) -> None:
        with cls._lock:
            if cls._fn is None:
                from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

                fn = ONNXMiniLM_L6_V2()
                fn(["warm up"])  # first call downloads + initialises the model
                cls._fn = fn

    @classmethod
    def warm_up_in_background(cls) -> None:
        if cls._fn is not None or (cls._loader is not None and cls._loader.is_alive()):
            return

        def run() -> None:
            try:
                cls._load()
            except Exception as exc:  # noqa: BLE001 - retrieval falls back to BM25
                cls.load_error = f"{type(exc).__name__}: {exc}"
                log.warning("local embedding model failed to load: %s", exc)

        cls._loader = threading.Thread(target=run, name="embedding-warmup", daemon=True)
        cls._loader.start()

    def __init__(self) -> None:
        self.warm_up_in_background()

    @property
    def ready(self) -> bool:
        return type(self)._fn is not None

    def unavailable_reason(self) -> str | None:
        if self.ready:
            return None
        if type(self).load_error:
            self.warm_up_in_background()  # try again later
            return "Local embedding model failed to load"
        return "Local embedding model is still loading"

    def wait_time(self, texts: list[str]) -> float:
        return 0.0 if self.ready else float("inf")

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not self.ready:
            await asyncio.to_thread(self._load)
        clipped = [t[: self.MAX_CHARS] for t in texts]

        def run() -> list:
            with type(self)._lock:  # one inference at a time: predictable CPU and memory use
                return type(self)._fn(clipped)

        return _normalise(np.array(await asyncio.to_thread(run)))


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
    if backend == "onnx":
        return OnnxEmbedder()
    if backend == "local":
        return LocalEmbedder(local_model)
    if backend == "gemini" and gemini_key:
        return GeminiEmbedder(gemini_key, gemini_model)
    return HashEmbedder()
