import numpy as np
import pytest

from research_agent.models import Chunk, Document
from research_agent.rag.chunking import chunk_documents, chunk_text
from research_agent.rag.embeddings import HashEmbedder
from research_agent.rag.retriever import HybridRetriever, reciprocal_rank_fusion
from research_agent.rag.store import ChromaStore, InMemoryStore


def test_chunk_text_respects_size_and_keeps_all_words():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(10))
    chunks = chunk_text(text, size=500, overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    for i in range(10):
        assert any(f"Paragraph {i}." in c for c in chunks)


def test_chunk_text_splits_giant_paragraph():
    chunks = chunk_text("x" * 5000, size=1000, overlap=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(len(c) for c in chunks) >= 5000


def test_chunk_ids_are_unique():
    docs = [Document(url=f"https://a/{i}", title="t", text="hello world. " * 400) for i in range(3)]
    chunks = chunk_documents(docs, "w1", 500, 50)
    assert len({c.id for c in chunks}) == len(chunks)


def test_rrf_rewards_agreement():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert scores["a"] > scores["c"] and scores["b"] > scores["d"]


@pytest.mark.parametrize("store_cls", [InMemoryStore, ChromaStore])
def test_vector_store_returns_nearest(store_cls):
    store = store_cls()
    vecs = np.eye(4, dtype=np.float32)
    store.add(["a", "b", "c", "d"], vecs)
    hits = store.query(vecs[2], 2)
    assert hits[0][0] == "c"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)
    store.close()


def _chunks():
    texts = {
        "u1": "Solar panel prices fell sharply; solar electricity is cheap in sunny regions.",
        "u2": "Nuclear reactors provide baseload power with high capacity factors.",
        "u3": "The Eiffel Tower is in Paris and was completed in 1889.",
    }
    return [Chunk(id=f"c-{k}", url=f"https://x/{k}", title=k, text=v) for k, v in texts.items()]


async def test_hybrid_retriever_finds_relevant_chunk():
    r = HybridRetriever(HashEmbedder(), InMemoryStore)
    await r.index(_chunks())
    hits = await r.search("how cheap is solar electricity", k=2)
    assert hits[0].id == "c-u1"
    r.close()


async def test_retriever_degrades_to_bm25_when_embeddings_fail():
    class Broken:
        name = "broken"

        async def embed(self, texts):
            raise RuntimeError("quota exceeded")

    r = HybridRetriever(Broken(), InMemoryStore)
    await r.index(_chunks())
    hits = await r.search("Eiffel Tower Paris", k=1)
    assert hits[0].id == "c-u3"
    assert not r.dense_enabled and r.warning


async def test_per_source_cap():
    chunks = [Chunk(id=f"c{i}", url="https://same", title="t", text=f"solar fact {i}") for i in range(5)]
    chunks.append(Chunk(id="other", url="https://other", title="t", text="solar other"))
    r = HybridRetriever(None, InMemoryStore)
    await r.index(chunks)
    hits = await r.search("solar", k=5, per_source_cap=2)
    assert sum(h.url == "https://same" for h in hits) == 2
