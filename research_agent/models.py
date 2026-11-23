"""Typed data that flows between the agents."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

# ---------- Planner ----------

class SubQuestion(BaseModel):
    id: int
    question: str = Field(description="A focused, answerable sub-question.")
    search_queries: list[str] = Field(description="1-3 web search queries for this sub-question.")
    rationale: str = Field(default="", description="Why this part matters for the main question.")


class ResearchPlan(BaseModel):
    main_question: str
    sub_questions: list[SubQuestion]


# ---------- Retrieval ----------

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    provider: str = ""


class Document(BaseModel):
    """A fetched web page (or a snippet-only fallback when the page cannot be fetched)."""

    url: str
    title: str
    text: str
    from_snippet: bool = False


class Chunk(BaseModel):
    id: str  # globally unique, e.g. "w2-d1-c3"
    url: str
    title: str
    text: str
    score: float = 0.0


# ---------- Worker ----------

class Claim(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(description="Chunk ids that support this statement.")


class Finding(BaseModel):
    sub_question_id: int
    sub_question: str
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    gaps: str = Field(default="", description="What the sources did not answer.")
    evidence: list[Chunk] = Field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.claims)


# ---------- Report ----------

class Source(BaseModel):
    number: int
    url: str
    title: str
    passages: list[str] = Field(default_factory=list)


class UsageStats(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    searches: int = 0
    pages_fetched: int = 0
    pages_failed: int = 0
    providers_used: list[str] = Field(default_factory=list)
    calls_by_model: dict[str, int] = Field(default_factory=dict)


class Report(BaseModel):
    question: str
    markdown: str
    sources: list[Source]
    plan: ResearchPlan
    findings: list[Finding]
    usage: UsageStats
    duration_seconds: float
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: list[str] = Field(default_factory=list)


# ---------- Progress events (for live UIs) ----------

class Stage(str, Enum):
    PLANNING = "planning"
    PLAN_READY = "plan_ready"
    WORKER_START = "worker_start"
    SEARCHING = "searching"
    FETCHING = "fetching"
    RETRIEVING = "retrieving"
    ANALYZING = "analyzing"
    WORKER_DONE = "worker_done"
    WORKER_FAILED = "worker_failed"
    SYNTHESIZING = "synthesizing"
    DONE = "done"
    WARNING = "warning"


class ProgressEvent(BaseModel):
    stage: Stage
    message: str
    worker_id: int | None = None
    data: dict = Field(default_factory=dict)
