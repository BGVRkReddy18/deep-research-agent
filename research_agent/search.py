"""Web search with automatic fallback: Tavily (best quality) -> DuckDuckGo (no key needed)."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import httpx

from .models import SearchResult
from .resilience import PermanentError, retry_async

log = logging.getLogger(__name__)


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class BadQueryError(Exception):
    """The provider rejected this particular query (4xx). Don't retry it; try the next provider."""


class TavilySearch:
    """Tavily search API. Also returns the cleaned page text, which saves us a fetch."""

    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, client: httpx.AsyncClient) -> None:
        self._key = api_key
        self._client = client
        self.raw_content: dict[str, str] = {}  # url -> page text Tavily already extracted

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        async def call() -> list[SearchResult]:
            resp = await self._client.post(
                self.URL,
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "query": query[:380],  # Tavily rejects queries over 400 characters
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_raw_content": True,
                },
                timeout=20.0,
            )
            if resp.status_code in (401, 403):
                raise PermanentError("Tavily rejected the API key")
            if resp.status_code in (432, 433):  # Tavily: plan / pay-as-you-go limit exceeded
                raise PermanentError("Tavily plan limit reached")
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise BadQueryError(f"Tavily {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            out = []
            for item in resp.json().get("results", []):
                url = item.get("url", "")
                if not url:
                    continue
                if item.get("raw_content"):
                    self.raw_content[url] = item["raw_content"]
                out.append(
                    SearchResult(
                        title=item.get("title") or url,
                        url=url,
                        snippet=item.get("content", ""),
                        provider=self.name,
                    )
                )
            return out

        return await retry_async(
            call, attempts=2, label="tavily",
            is_retryable=lambda e: not isinstance(e, (PermanentError, BadQueryError)),
        )


class DuckDuckGoSearch:
    """Keyless fallback via the `ddgs` package (sync, so it runs in a worker thread)."""

    name = "duckduckgo"

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        def run() -> list[dict]:
            from ddgs import DDGS

            return DDGS().text(query, max_results=max_results) or []

        async def call() -> list[SearchResult]:
            rows = await asyncio.to_thread(run)
            return [
                SearchResult(
                    title=r.get("title") or r.get("href", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    provider=self.name,
                )
                for r in rows
                if r.get("href")
            ]

        return await retry_async(call, attempts=2, base_delay=2.0, label="duckduckgo")


class FallbackSearch:
    """Tries each provider in order. A provider that fails permanently is skipped from then on."""

    def __init__(self, providers: list[SearchProvider]) -> None:
        if not providers:
            raise ValueError("at least one search provider is required")
        self.providers = providers
        self._disabled: set[str] = set()
        self.calls = 0

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self.providers)

    def cached_page(self, url: str) -> str | None:
        for p in self.providers:
            cache = getattr(p, "raw_content", None)
            if cache and url in cache:
                return cache[url]
        return None

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        errors = []
        for provider in self.providers:
            if provider.name in self._disabled:
                continue
            try:
                results = await provider.search(query, max_results)
            except PermanentError as exc:
                log.warning("disabling search provider %s: %s", provider.name, exc)
                self._disabled.add(provider.name)
                errors.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001
                log.warning("search provider %s failed: %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
                continue
            self.calls += 1
            if results:
                return results
        if errors:
            log.warning("all search providers failed for %r: %s", query, errors)
        return []


def build_search(tavily_key: str, client: httpx.AsyncClient) -> FallbackSearch:
    providers: list[SearchProvider] = []
    if tavily_key:
        providers.append(TavilySearch(tavily_key, client))
    providers.append(DuckDuckGoSearch())
    return FallbackSearch(providers)
