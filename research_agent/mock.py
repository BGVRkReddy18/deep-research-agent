"""Offline doubles for the LLM and search, so the whole pipeline runs with no API keys.

Used by `--mock` (a demo anyone can run instantly) and by the test-suite.
"""

from __future__ import annotations

import json
import re

import httpx

from .llm import ChatBackend, Completion, LLMRouter, Message
from .models import SearchResult
from .orchestrator import Components
from .rag.embeddings import HashEmbedder
from .rag.store import InMemoryStore
from .search import FallbackSearch

_CORPUS = {
    "https://example.org/solar-costs": (
        "Solar Power Costs 2025",
        "Utility-scale solar levelized cost of electricity fell roughly 90 percent between 2010 and 2024. "
        "In 2024 new solar farms typically cost between 30 and 50 dollars per megawatt-hour in sunny regions.\n\n"
        "Battery storage adds cost but prices for lithium-ion packs dropped below 120 dollars per kilowatt-hour.",
    ),
    "https://example.org/nuclear-costs": (
        "Nuclear Power Economics",
        "New nuclear plants in Europe and the United States have cost between 110 and 180 dollars per "
        "megawatt-hour, driven by long construction times and financing costs.\n\n"
        "Nuclear provides firm, low-carbon baseload power with capacity factors above 90 percent.",
    ),
    "https://example.org/grid-integration": (
        "Integrating Renewables Into The Grid",
        "Variable renewables such as solar require storage, transmission or backup capacity. "
        "System integration costs rise as solar share grows beyond 30 percent of generation.\n\n"
        "Small modular reactors promise lower costs but no commercial fleet existed in 2024.",
    ),
}


class MockSearchProvider:
    name = "mock-search"

    def __init__(self) -> None:
        self.raw_content = {url: text for url, (_, text) in _CORPUS.items()}

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return [
            SearchResult(title=title, url=url, snippet=text[:160], provider=self.name)
            for url, (title, text) in list(_CORPUS.items())[:max_results]
        ]


class MockLLM:
    """Recognises which agent is calling from the system prompt and answers in kind."""

    name = "mock-llm"

    async def complete(self, messages: list[Message], *, temperature: float, max_tokens: int, json_mode: bool) -> Completion:
        system, user = messages[0].content, messages[-1].content
        if "planning agent" in system:
            text = json.dumps(
                {
                    "sub_questions": [
                        {"id": 1, "question": "What does solar power cost today?", "search_queries": ["solar LCOE 2025"]},
                        {"id": 2, "question": "What does new nuclear power cost?", "search_queries": ["nuclear LCOE 2025"]},
                        {"id": 3, "question": "What hidden system costs apply?", "search_queries": ["grid integration costs"]},
                    ]
                }
            )
        elif "You are a research worker" in system:
            ids = re.findall(r"^\[(w\d+-d\d+-c\d+)\]", user, flags=re.MULTILINE)
            text = json.dumps(
                {
                    "summary": "The sources give concrete cost ranges.",
                    "claims": [
                        {"statement": f"Evidence passage {cid} reports relevant cost data.", "evidence_ids": [cid]}
                        for cid in ids[:3]
                    ],
                    "gaps": "Regional variation is not covered.",
                }
            )
        else:
            nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", user)})
            cite = lambda i: f"[{nums[i % len(nums)]}]" if nums else ""
            text = (
                "# Is solar cheaper than nuclear?\n\n## Summary\n"
                f"New solar is generally cheaper per megawatt-hour than new nuclear {cite(0)}{cite(1)}. "
                f"Integration costs narrow the gap at high solar shares {cite(2)}.\n\n"
                "## Limitations\n- Mock data only."
            )
        return Completion(text=text, provider=self.name, prompt_tokens=len(user) // 4, completion_tokens=len(text) // 4)


def mock_components(llm_backends: list[ChatBackend] | None = None) -> Components:
    return Components(
        llm=LLMRouter(llm_backends or [MockLLM()]),
        search=FallbackSearch([MockSearchProvider()]),
        embedder=HashEmbedder(),
        store_factory=InMemoryStore,
        http=httpx.AsyncClient(),
        owns_http=True,
    )
