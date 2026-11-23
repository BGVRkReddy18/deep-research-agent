"""Orchestrator: planner -> parallel workers -> synthesizer.

Error-recovery strategy (each layer degrades instead of crashing the run):
  * LLM call fails            -> retry w/ backoff -> fail over to next provider
  * bad JSON from the model   -> feed validation error back, ask it to repair
  * planner fails             -> research the question directly (1-step plan)
  * search provider fails     -> next provider (Tavily -> DuckDuckGo)
  * page can't be fetched     -> use the search snippet instead
  * embeddings fail           -> BM25 keyword retrieval only
  * a worker fails / times out-> report continues with the other workers
  * synthesizer fails         -> deterministic report from raw findings
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from .agents.planner import Planner
from .agents.synthesizer import Synthesizer
from .agents.worker import Worker
from .config import Settings, build_providers, get_settings
from .fetch import PageFetcher
from .llm import LLMRouter, build_router
from .models import Finding, ProgressEvent, Report, Stage
from .rag.embeddings import Embedder, build_embedder
from .rag.store import VectorStore, make_store
from .search import FallbackSearch, build_search

log = logging.getLogger(__name__)

Emit = Callable[[ProgressEvent], Awaitable[None]]


async def _noop(_: ProgressEvent) -> None:
    return None


@dataclass
class UserKeys:
    """Optional bring-your-own-key overrides supplied by an app user."""

    gemini: str = ""
    groq: str = ""
    tavily: str = ""


@dataclass
class Components:
    llm: LLMRouter
    search: FallbackSearch
    embedder: Embedder | None
    store_factory: Callable[[], VectorStore]
    http: httpx.AsyncClient | None = None
    owns_http: bool = field(default=False)


def build_components(settings: Settings, keys: UserKeys | None = None) -> Components:
    keys = keys or UserKeys()
    overrides = {"gemini": keys.gemini, "groq": keys.groq}
    providers = [
        cfg for name in settings.provider_order for cfg in build_providers(name, overrides.get(name) or None)
    ]
    llm = build_router(providers, timeout=settings.llm_timeout, max_retries=settings.max_retries)
    http = httpx.AsyncClient(limits=httpx.Limits(max_connections=20))
    gemini_key = keys.gemini or next((p.api_key for p in providers if p.name == "gemini"), "")
    embedder = build_embedder(
        settings.embedding_backend,
        gemini_key=gemini_key,
        gemini_model=settings.gemini_embedding_model,
        local_model=settings.local_embedding_model,
    )
    return Components(
        llm=llm,
        search=build_search(keys.tavily or settings.tavily_api_key, http),
        embedder=embedder,
        store_factory=lambda: make_store(settings.vector_store),
        http=http,
        owns_http=True,
    )


class ResearchOrchestrator:
    def __init__(self, components: Components, settings: Settings | None = None) -> None:
        self.c = components
        self.s = settings or get_settings()
        self.http = components.http or httpx.AsyncClient()
        self.fetcher = PageFetcher(self.http, timeout=self.s.fetch_timeout)

    async def aclose(self) -> None:
        if self.c.owns_http or self.c.http is None:
            await self.http.aclose()

    async def run(self, question: str, emit: Emit = _noop, worker_timeout: float = 240.0) -> Report:
        question = " ".join(question.split())
        if not question:
            raise ValueError("question is empty")
        if len(question) > self.s.max_question_chars:
            raise ValueError(f"question is longer than {self.s.max_question_chars} characters")

        started = time.monotonic()
        warnings: list[str] = []

        async def warn(msg: str) -> None:
            warnings.append(msg)
            await emit(ProgressEvent(stage=Stage.WARNING, message=msg))

        # 1. Plan
        await emit(ProgressEvent(stage=Stage.PLANNING, message="Breaking the question into sub-questions"))
        plan, plan_warning = await Planner(self.c.llm, self.s.max_sub_questions).plan(question)
        if plan_warning:
            await warn(plan_warning)
        await emit(
            ProgressEvent(
                stage=Stage.PLAN_READY,
                message=f"Plan ready: {len(plan.sub_questions)} sub-questions",
                data={"plan": plan.model_dump()},
            )
        )

        # 2. Research every sub-question in parallel (bounded, isolated, time-limited).
        worker = Worker(
            llm=self.c.llm,
            search=self.c.search,
            fetcher=self.fetcher,
            embedder=self.c.embedder,
            store_factory=self.c.store_factory,
            settings=self.s,
            emit=emit,
        )
        sem = asyncio.Semaphore(self.s.worker_concurrency)

        async def run_one(sq) -> Finding:
            async with sem:
                try:
                    finding = await asyncio.wait_for(worker.run(sq), timeout=worker_timeout)
                except asyncio.TimeoutError:
                    finding = Finding(sub_question_id=sq.id, sub_question=sq.question, error="Timed out.")
                except Exception as exc:
                    log.exception("worker %s crashed", sq.id)
                    finding = Finding(
                        sub_question_id=sq.id, sub_question=sq.question, error=f"{type(exc).__name__}: {exc}"
                    )
            if finding.ok:
                await emit(
                    ProgressEvent(
                        stage=Stage.WORKER_DONE,
                        worker_id=sq.id,
                        message=f"{len(finding.claims)} cited claims",
                        data={"summary": finding.summary},
                    )
                )
            else:
                await emit(ProgressEvent(stage=Stage.WORKER_FAILED, worker_id=sq.id, message=finding.error or "failed"))
            return finding

        findings = list(await asyncio.gather(*(run_one(sq) for sq in plan.sub_questions)))
        failed = [f for f in findings if not f.ok]
        if failed and len(failed) < len(findings):
            await warn(f"{len(failed)} of {len(findings)} sub-questions could not be fully researched.")

        # 3. Synthesize
        await emit(ProgressEvent(stage=Stage.SYNTHESIZING, message="Writing the cited report"))
        markdown, sources, synth_warnings = await Synthesizer(self.c.llm).write(question, findings)
        for w in synth_warnings:
            await warn(w)

        usage = self.c.llm.usage.model_copy()
        usage.searches = self.c.search.calls
        usage.pages_fetched = self.fetcher.fetched
        usage.pages_failed = self.fetcher.failed

        report = Report(
            question=question,
            markdown=markdown,
            sources=sources,
            plan=plan,
            findings=findings,
            usage=usage,
            duration_seconds=round(time.monotonic() - started, 1),
            warnings=warnings,
        )
        await emit(ProgressEvent(stage=Stage.DONE, message=f"Done in {report.duration_seconds}s"))
        return report


async def research(
    question: str,
    *,
    emit: Emit = _noop,
    settings: Settings | None = None,
    keys: UserKeys | None = None,
    components: Components | None = None,
) -> Report:
    """One-call entry point used by the CLI and the web app."""
    settings = settings or get_settings()
    orch = ResearchOrchestrator(components or build_components(settings, keys), settings)
    try:
        return await orch.run(question, emit=emit)
    finally:
        await orch.aclose()
