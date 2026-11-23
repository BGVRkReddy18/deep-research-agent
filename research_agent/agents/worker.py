"""Worker agent: one sub-question -> search -> read -> RAG -> cited findings."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from ..config import Settings
from ..fetch import PageFetcher
from ..llm import LLMRouter, Message
from ..models import Chunk, Claim, Finding, ProgressEvent, SearchResult, Stage, SubQuestion
from ..rag.chunking import chunk_documents
from ..rag.embeddings import Embedder
from ..rag.retriever import HybridRetriever
from ..rag.store import VectorStore
from ..search import FallbackSearch
from .prompts import WORKER_SYSTEM, WORKER_USER

log = logging.getLogger(__name__)

Emit = Callable[[ProgressEvent], Awaitable[None]]


class _WorkerOut(BaseModel):
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    gaps: str = ""


def format_evidence(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] ({c.title} - {c.url})\n{c.text}" for c in chunks)


def validate_claims(claims: list[Claim], evidence: list[Chunk]) -> list[Claim]:
    """Drop citations to ids we never showed the model, and claims left with no support."""
    valid_ids = {c.id for c in evidence}
    out = []
    for claim in claims:
        ids = list(dict.fromkeys(i.strip().strip("[]【】 ") for i in claim.evidence_ids))
        ids = [i for i in ids if i in valid_ids]
        if ids and claim.statement.strip():
            out.append(Claim(statement=claim.statement.strip(), evidence_ids=ids))
    return out


class Worker:
    def __init__(
        self,
        *,
        llm: LLMRouter,
        search: FallbackSearch,
        fetcher: PageFetcher,
        embedder: Embedder | None,
        store_factory: Callable[[], VectorStore],
        settings: Settings,
        emit: Emit,
    ) -> None:
        self.llm = llm
        self.search = search
        self.fetcher = fetcher
        self.embedder = embedder
        self.store_factory = store_factory
        self.s = settings
        self.emit = emit

    async def _say(self, stage: Stage, wid: int, message: str, **data) -> None:
        await self.emit(ProgressEvent(stage=stage, message=message, worker_id=wid, data=data))

    async def run(self, sq: SubQuestion) -> Finding:
        wid = sq.id
        await self._say(Stage.WORKER_START, wid, sq.question)

        # 1. Search all queries in parallel, dedupe URLs, keep the best-ranked ones.
        await self._say(Stage.SEARCHING, wid, f"Searching: {', '.join(sq.search_queries)}")
        batches = await asyncio.gather(
            *(self.search.search(q, self.s.results_per_query) for q in sq.search_queries)
        )
        results = _interleave_unique(batches)[: self.s.max_pages_per_worker]
        if not results:
            return Finding(sub_question_id=wid, sub_question=sq.question, error="No search results found.")

        # 2. Fetch pages concurrently (falls back to snippets when a page can't be read).
        await self._say(
            Stage.FETCHING, wid, f"Reading {len(results)} pages",
            urls=[r.url for r in results],
        )
        docs = await asyncio.gather(
            *(self.fetcher.fetch(r, self.search.cached_page(r.url)) for r in results)
        )
        docs = [d for d in docs if d.text.strip()]

        # 3. RAG: chunk -> embed -> index -> hybrid retrieve the most relevant passages.
        chunks = chunk_documents(docs, f"w{wid}", self.s.chunk_size, self.s.chunk_overlap)
        retriever = HybridRetriever(self.embedder, self.store_factory)
        try:
            await retriever.index(chunks)
            if retriever.warning:
                await self._say(Stage.WARNING, wid, retriever.warning)
            evidence = await self._retrieve(retriever, sq)
        finally:
            retriever.close()
        await self._say(
            Stage.RETRIEVING, wid,
            f"Selected {len(evidence)} of {len(chunks)} passages from {len(docs)} sources",
        )
        if not evidence:
            return Finding(sub_question_id=wid, sub_question=sq.question, error="No readable content in sources.")

        # 4. Ask the LLM to extract cited claims from the evidence only.
        await self._say(Stage.ANALYZING, wid, "Extracting cited findings")
        out = await self.llm.complete_json(
            [
                Message("system", WORKER_SYSTEM),
                Message("user", WORKER_USER.format(question=sq.question, evidence=format_evidence(evidence))),
            ],
            _WorkerOut,
            max_tokens=1200,
        )
        claims = validate_claims(out.claims, evidence)
        used = {i for c in claims for i in c.evidence_ids}
        finding = Finding(
            sub_question_id=wid,
            sub_question=sq.question,
            summary=out.summary.strip(),
            claims=claims,
            gaps=out.gaps.strip(),
            evidence=[c for c in evidence if c.id in used],
            error=None if claims else "Sources did not contain supported claims.",
        )
        return finding

    async def _retrieve(self, retriever: HybridRetriever, sq: SubQuestion) -> list[Chunk]:
        """Multi-query hybrid retrieval: the sub-question plus each search query, fused with RRF."""
        return await retriever.search([sq.question, *sq.search_queries], k=self.s.chunks_per_worker)


def _interleave_unique(batches: list[list[SearchResult]]) -> list[SearchResult]:
    """Round-robin across query results so every query contributes its top hits."""
    seen: set[str] = set()
    out: list[SearchResult] = []
    for i in range(max((len(b) for b in batches), default=0)):
        for batch in batches:
            if i < len(batch):
                url = batch[i].url.split("#")[0].rstrip("/")
                if url not in seen:
                    seen.add(url)
                    out.append(batch[i])
    return out
