import json

import pytest

from research_agent.agents.synthesizer import assign_sources, renumber_citations
from research_agent.agents.worker import validate_claims
from research_agent.llm import Completion, LLMError, LLMRouter, Message, extract_json
from research_agent.mock import MockLLM, mock_components
from research_agent.models import Chunk, Claim, Finding, Source, Stage
from research_agent.orchestrator import research

# ---------------- JSON handling

def test_extract_json_from_fenced_and_chatty_output():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} hope it helps') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


class ScriptedLLM:
    """Returns canned responses in order; raises if an item is an exception."""

    def __init__(self, name, replies):
        self.name = name
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages, *, temperature, max_tokens, json_mode):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Completion(text=reply, provider=self.name)


async def test_router_fails_over_to_next_provider():
    bad = ScriptedLLM("bad", [RuntimeError("429 rate limited")])
    good = ScriptedLLM("good", ["hello"])
    router = LLMRouter([bad, good])
    out = await router.complete([Message("user", "hi")])
    assert out.text == "hello" and router.usage.providers_used == ["good"]


async def test_complete_json_repairs_invalid_output():
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    llm = ScriptedLLM("m", ["not json at all", json.dumps({"x": 7})])
    out = await LLMRouter([llm]).complete_json([Message("user", "give x")], Out)
    assert out.x == 7 and llm.calls == 2


async def test_complete_json_gives_up():
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    llm = ScriptedLLM("m", ["{}", "{}", "{}"])
    with pytest.raises(LLMError):
        await LLMRouter([llm]).complete_json([Message("user", "x")], Out, repair_attempts=2)


# ---------------- citation integrity

def test_validate_claims_drops_hallucinated_ids():
    evidence = [Chunk(id="w1-d1-c1", url="u", title="t", text="x")]
    claims = [
        Claim(statement="real", evidence_ids=["w1-d1-c1", "w9-d9-c9"]),
        Claim(statement="made up", evidence_ids=["w9-d9-c9"]),
    ]
    out = validate_claims(claims, evidence)
    assert [c.statement for c in out] == ["real"]
    assert out[0].evidence_ids == ["w1-d1-c1"]


def test_renumber_citations_orders_by_first_use_and_strips_invalid():
    sources = {n: Source(number=n, url=f"u{n}", title=f"t{n}") for n in (1, 2, 3)}
    md, cited, invalid = renumber_citations("A [3]. B [1, 3] and C [9]. See [link](http://x).", sources)
    assert md == "A [1]. B [2][1] and C. See [link](http://x)."
    assert [s.url for s in cited] == ["u3", "u1"]
    assert invalid == 1


def test_assign_sources_groups_chunks_by_url():
    f = Finding(
        sub_question_id=1, sub_question="q",
        claims=[Claim(statement="s", evidence_ids=["a", "b"])],
        evidence=[Chunk(id="a", url="u1", title="t", text="p1"), Chunk(id="b", url="u1", title="t", text="p2")],
    )
    chunk_to_num, sources = assign_sources([f])
    assert chunk_to_num == {"a": 1, "b": 1}
    assert sources[1].passages == ["p1", "p2"]


# ---------------- end to end (offline)

async def test_full_pipeline_offline():
    events = []

    async def emit(ev):
        events.append(ev.stage)

    report = await research("Is solar cheaper than nuclear?", emit=emit, components=mock_components())
    assert len(report.plan.sub_questions) == 3
    assert all(f.ok for f in report.findings)
    assert report.sources and "## Sources" in report.markdown
    assert "[1]" in report.markdown
    assert Stage.PLAN_READY in events and events[-1] is Stage.DONE


async def test_pipeline_survives_total_llm_outage_after_planning():
    """Planner works, then every later call fails: run still returns a report with warnings."""

    class DiesAfterPlan(MockLLM):
        n = 0

        async def complete(self, messages, **kw):
            self.n += 1
            if self.n > 1:
                raise RuntimeError("provider down")
            return await super().complete(messages, **kw)

    report = await research("Is solar cheaper than nuclear?", components=mock_components([DiesAfterPlan()]))
    assert all(not f.ok for f in report.findings)
    assert report.warnings
    assert "No sub-question produced supported findings" in report.markdown


async def test_planner_failure_falls_back_to_direct_research():
    class BadPlanner(MockLLM):
        async def complete(self, messages, **kw):
            if "planning agent" in messages[0].content:
                return Completion(text="I cannot do JSON", provider=self.name)
            return await super().complete(messages, **kw)

    report = await research("Is solar cheaper than nuclear?", components=mock_components([BadPlanner()]))
    assert len(report.plan.sub_questions) == 1
    assert any("Planner failed" in w for w in report.warnings)
    assert report.sources


async def test_rejects_empty_and_huge_questions():
    with pytest.raises(ValueError):
        await research("   ", components=mock_components())
    with pytest.raises(ValueError):
        await research("x" * 10_000, components=mock_components())
