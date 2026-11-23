"""Hybrid retrieval: BM25 keyword search + dense vector search, fused with Reciprocal Rank Fusion.

Why hybrid? Vectors catch paraphrases ("cost" ~ "price"); BM25 catches exact terms that
embeddings blur (model numbers, names, years). RRF combines the rankings without having
to calibrate their very different score scales.

Pipeline per worker (multi-query):
  1. BM25 over *all* chunks for every query            -> cheap, local, exact-term recall
  2. union of each query's BM25 top hits = candidates   -> bounds embedding cost
  3. embed queries + candidates in one batched call, index candidates in the vector store
  4. dense search for every query over the candidates
  5. RRF over all 2 x len(queries) rankings, then a per-source diversity cap
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable

from rank_bm25 import BM25Okapi

from ..models import Chunk
from .embeddings import Embedder
from .store import VectorStore

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have how in is it of on or that the this to was were what when "
    "where which who why will with does do did vs versus".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return dict(scores)


class HybridRetriever:
    def __init__(self, embedder: Embedder | None, store_factory: Callable[[], VectorStore]) -> None:
        self._embedder = embedder
        self._store_factory = store_factory
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._bm25: BM25Okapi | None = None
        self.dense_enabled = embedder is not None
        self.warning: str | None = None
        self.embedded = 0  # how many chunks were embedded (for stats / tests)

    async def index(self, chunks: list[Chunk]) -> None:
        self._chunks = {c.id: c for c in chunks}
        self._order = [c.id for c in chunks]
        if chunks:
            self._bm25 = BM25Okapi([tokenize(f"{c.title} {c.text}") or ["_"] for c in chunks])

    def _bm25_ranking(self, query: str, top: int) -> list[str]:
        assert self._bm25 is not None
        scores = self._bm25.get_scores(tokenize(query) or ["_"])
        best = sorted(range(len(self._order)), key=lambda i: -scores[i])[:top]
        return [self._order[i] for i in best if scores[i] > 0] or [self._order[i] for i in best]

    async def _dense_rankings(self, queries: list[str], candidates: list[str], top: int) -> list[list[str]]:
        assert self._embedder is not None
        texts = [f"{self._chunks[c].title}\n{self._chunks[c].text}" for c in candidates]
        vectors = await self._embedder.embed(queries + texts)
        self.embedded += len(texts)
        store = self._store_factory()
        try:
            store.add(candidates, vectors[len(queries) :])
            return [[cid for cid, _ in store.query(vectors[i], top)] for i in range(len(queries))]
        finally:
            store.close()

    async def search(
        self,
        queries: str | list[str],
        k: int = 6,
        per_source_cap: int = 2,
        candidates: int = 24,
    ) -> list[Chunk]:
        if isinstance(queries, str):
            queries = [queries]
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
        if not self._chunks or self._bm25 is None or not queries:
            return []

        per_query = max(8, candidates // len(queries))
        rankings = [self._bm25_ranking(q, per_query) for q in queries]

        if self.dense_enabled:
            pool = list(dict.fromkeys(cid for r in rankings for cid in r))[:candidates]
            try:
                rankings += await self._dense_rankings(queries, pool, top=per_query)
            except Exception as exc:  # noqa: BLE001 - degrade to keyword-only retrieval
                log.warning("dense retrieval failed, falling back to BM25 only: %s", exc)
                self.warning = f"Embedding failed ({type(exc).__name__}); used keyword search only."
                self.dense_enabled = False

        fused = reciprocal_rank_fusion(rankings)
        picked: list[Chunk] = []
        per_url: dict[str, int] = defaultdict(int)
        for cid in sorted(fused, key=lambda c: -fused[c]):
            chunk = self._chunks[cid]
            if per_url[chunk.url] >= per_source_cap:
                continue
            per_url[chunk.url] += 1
            picked.append(chunk.model_copy(update={"score": round(fused[cid], 5)}))
            if len(picked) >= k:
                break
        return picked

    def close(self) -> None:
        self._chunks, self._order, self._bm25 = {}, [], None
