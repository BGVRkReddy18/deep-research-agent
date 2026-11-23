"""Download web pages and extract their main readable text."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx

from .models import Document, SearchResult

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 DeepResearchAgent/1.0"
)
MAX_BYTES = 3_000_000
MIN_TEXT_CHARS = 300
MAX_TEXT_CHARS = 60_000

# Pages that rarely yield useful extractable text, or block bots outright.
SKIP_DOMAINS = {"youtube.com", "www.youtube.com", "instagram.com", "www.instagram.com", "tiktok.com"}


def is_fetchable(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in SKIP_DOMAINS or host in ("localhost",) or host.startswith(("127.", "10.", "192.168.")):
        return False
    return not parsed.path.lower().endswith((".pdf", ".zip", ".mp4", ".mp3", ".png", ".jpg"))


def extract_text(html: str, url: str) -> str:
    import trafilatura

    text = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=True, favor_precision=True
    )
    return (text or "").strip()


class PageFetcher:
    def __init__(self, client: httpx.AsyncClient, *, timeout: float = 12.0, concurrency: int = 6) -> None:
        self._client = client
        self._timeout = timeout
        self._sem = asyncio.Semaphore(concurrency)
        self.fetched = 0
        self.failed = 0

    async def fetch(self, result: SearchResult, cached_text: str | None = None) -> Document:
        """Return the page as a Document. Never raises: falls back to the search snippet."""
        # Text the search provider already extracted (e.g. Tavily raw_content) is trusted as-is.
        if cached_text and len(cached_text.strip()) >= 50:
            self.fetched += 1
            return Document(url=result.url, title=result.title, text=cached_text[:MAX_TEXT_CHARS])

        text = ""
        if is_fetchable(result.url):
            try:
                async with self._sem:
                    text = await self._download(result.url)
            except Exception as exc:  # noqa: BLE001 - any failure -> snippet fallback
                log.info("fetch failed for %s: %s", result.url, exc)

        if len(text) >= MIN_TEXT_CHARS:
            self.fetched += 1
            return Document(url=result.url, title=result.title, text=text[:MAX_TEXT_CHARS])

        self.failed += 1
        return Document(url=result.url, title=result.title, text=result.snippet, from_snippet=True)

    async def _download(self, url: str) -> str:
        async with self._client.stream(
            "GET",
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=self._timeout,
            follow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                return ""
            body = bytearray()
            async for part in resp.aiter_bytes():
                body.extend(part)
                if len(body) > MAX_BYTES:
                    break
        html = body.decode(resp.encoding or "utf-8", errors="replace")
        return await asyncio.to_thread(extract_text, html, url)
