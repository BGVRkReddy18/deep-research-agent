"""Regression tests for problems found in the first live run against free-tier APIs."""

import time

from research_agent.agents.planner import clean_queries
from research_agent.models import Chunk
from research_agent.rag.embeddings import HashEmbedder
from research_agent.rag.retriever import HybridRetriever
from research_agent.rag.store import InMemoryStore
from research_agent.resilience import RateLimiter, retry_after_hint


def test_clean_queries_repairs_glued_json_from_model():
    raw = ['definition of RAG hallucination","RAG hallucination causes"]', "rationale", ":"]
    assert clean_queries(raw, "fallback") == ["definition of RAG hallucination", "RAG hallucination causes"]


def test_clean_queries_falls_back_when_nothing_usable():
    assert clean_queries([":", "x", "rationale"], "what is rag") == ["what is rag"]


def test_retry_hint_parsed_from_error_message():
    groq = Exception("Rate limit reached ... Please try again in 15.24s. Need more tokens?")
    gemini = Exception("Quota exceeded ... Please retry in 3.249360179s.")
    assert 15 < retry_after_hint(groq) < 17
    assert 3 < retry_after_hint(gemini) < 5
    assert retry_after_hint(Exception("boom")) is None


async def test_weighted_rate_limiter_blocks_when_token_budget_is_used():
    limiter = RateLimiter(100, period=0.3)
    await limiter.acquire(70)
    t = time.monotonic()
    await limiter.acquire(50)  # would exceed 100 -> must wait for the window to slide
    assert time.monotonic() - t >= 0.25
    await limiter.acquire(10_000)  # oversized costs are clamped instead of deadlocking


async def test_retriever_embeds_only_bm25_candidates():
    chunks = [Chunk(id=f"c{i}", url=f"https://s/{i % 7}", title="t", text=f"filler text number {i}") for i in range(80)]
    chunks.append(Chunk(id="hit", url="https://s/hit", title="t", text="RAG hallucination mitigation with citations"))
    r = HybridRetriever(HashEmbedder(), InMemoryStore)
    await r.index(chunks)
    hits = await r.search(["RAG hallucination mitigation", "citations hallucination"], k=4, candidates=20)
    assert hits[0].id == "hit"
    assert r.embedded <= 20  # not all 81 chunks


def test_normalize_citation_dialects():
    from research_agent.agents.synthesizer import normalize_citations

    text = "A【1】 B【2†source】 C[^3] D[Source 4] E[2-4] F[1, 2] G (2023)"
    assert normalize_citations(text) == "A[1] B[2] C[3] D[4] E[2][3][4] F[1, 2] G (2023)"


async def test_run_stops_early_with_clear_error_when_no_model_is_reachable():
    import pytest

    from research_agent.llm import AllProvidersFailed
    from research_agent.mock import mock_components
    from research_agent.orchestrator import research

    class Down:
        name = "groq:retired-model"

        async def complete(self, messages, **kw):
            raise RuntimeError("Error code: 404 - model does not exist")

    comps = mock_components([Down()])
    with pytest.raises(AllProvidersFailed, match="Could not reach any AI model") as info:
        await research("How many centuries has Virat Kohli scored?", components=comps)
    assert "404" in str(info.value)
    assert comps.search.calls == 0  # no search quota wasted


async def test_retriever_skips_embeddings_when_quota_wait_exceeds_budget():
    """Deployed-app finding: workers stalled ~60s waiting for embedding quota."""

    class BusyEmbedder(HashEmbedder):
        calls = 0

        def wait_time(self, texts):
            return 45.0

        async def embed(self, texts):
            self.calls += 1
            return await super().embed(texts)

    emb = BusyEmbedder()
    chunks = [Chunk(id=f"c{i}", url=f"https://s/{i}", title="t", text=f"solar cost fact {i}") for i in range(10)]
    r = HybridRetriever(emb, InMemoryStore, max_dense_wait=8.0)
    await r.index(chunks)
    t = time.monotonic()
    hits = await r.search(["solar cost"], k=3)
    assert time.monotonic() - t < 1.0  # no stall
    assert hits and emb.calls == 0  # BM25 only for this step
    assert r.note and "keyword search" in r.note
    assert r.dense_enabled  # not disabled for good: later steps may embed once quota frees up


async def test_retry_gives_up_instead_of_exceeding_latency_budget():
    import pytest

    from research_agent.resilience import retry_async

    calls = 0

    async def rate_limited():
        nonlocal calls
        calls += 1
        raise RuntimeError("429 quota exceeded. Please retry in 59.4s.")

    t = time.monotonic()
    with pytest.raises(RuntimeError):
        await retry_async(rate_limited, attempts=3, max_wait=8.0, is_retryable=lambda e: True)
    assert calls == 1 and time.monotonic() - t < 1.0


def test_gemini_embedder_shares_quota_and_cache_across_runs():
    from research_agent.rag.embeddings import GeminiEmbedder

    a, b = GeminiEmbedder("same-key"), GeminiEmbedder("same-key")
    assert a._text_limiter is b._text_limiter and a._cache is b._cache
    assert GeminiEmbedder("other-key")._cache is not a._cache
    a._cache["x" * 10] = None  # cached texts cost no quota
    assert b.wait_time(["x" * 10]) == 0.0
