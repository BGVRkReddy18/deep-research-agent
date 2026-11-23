"""Error-recovery primitives: retries with backoff, and a client-side rate limiter.

Free-tier APIs are strict about requests-per-minute. Instead of hammering them and
getting 429s, we pace our own calls and back off when told to slow down.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Sliding-window limiter: at most `rate` units of cost per `period` seconds.

    With the default cost of 1 it limits requests per minute; pass an estimated token
    count as the cost to limit tokens per minute.
    """

    def __init__(self, rate: int, period: float = 60.0) -> None:
        self.rate = max(1, rate)
        self.period = period
        self._events: deque[tuple[float, int]] = deque()
        self._used = 0
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        # Limiters outlive event loops (the web app runs each request in a fresh loop), and an
        # asyncio.Lock is bound to the loop it was first used in, so keep one lock per loop.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    def _clamp(self, cost: int) -> int:
        return max(1, min(cost, self.rate))  # a single oversized request must still be able to run

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= self.period:
            self._used -= self._events.popleft()[1]

    def wait_time(self, cost: int = 1) -> float:
        """Seconds until `cost` units would be available (0 if available now). Does not reserve."""
        cost = self._clamp(cost)
        now = time.monotonic()
        self._prune(now)
        if self._used + cost <= self.rate:
            return 0.0
        freed = 0
        for stamp, used in self._events:
            freed += used
            if self._used - freed + cost <= self.rate:
                return stamp + self.period - now
        return self.period

    def try_acquire(self, cost: int = 1) -> bool:
        """Reserve `cost` units if available right now. Synchronous, so atomic within asyncio."""
        cost = self._clamp(cost)
        now = time.monotonic()
        self._prune(now)
        if self._used + cost > self.rate:
            return False
        self._events.append((now, cost))
        self._used += cost
        return True

    async def acquire(self, cost: int = 1) -> None:
        async with self._get_lock():
            while not self.try_acquire(cost):
                await asyncio.sleep(max(self.wait_time(cost), 0.0) + 0.05)


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


class PermanentError(Exception):
    """An error that retrying will not fix (bad key, invalid request, ...)."""


_RETRY_IN = re.compile(r"(?:retry|try again) in ([\d.]+)\s*s", re.IGNORECASE)


def retry_after_hint(exc: BaseException) -> float | None:
    """Server-suggested wait: the Retry-After header, else 'try again in 12.3s' in the message."""
    response = getattr(exc, "response", None)
    if response is not None:
        value = response.headers.get("retry-after")
        try:
            if value:
                return min(float(value), 60.0)
        except ValueError:
            pass
    match = _RETRY_IN.search(str(exc))
    return min(float(match.group(1)) + 0.5, 60.0) if match else None


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    is_retryable: Callable[[BaseException], bool] = lambda e: not isinstance(e, PermanentError),
    retry_after: Callable[[BaseException], float | None] = retry_after_hint,
    label: str = "call",
) -> T:
    """Run `fn` with exponential backoff + jitter. Honors server Retry-After hints."""
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            last = exc
            if attempt == attempts or not is_retryable(exc):
                raise
            hinted = retry_after(exc)
            delay = hinted if hinted is not None else min(max_delay, base_delay * 2 ** (attempt - 1))
            delay += random.uniform(0, 0.5)
            log.warning("%s failed (%s), retry %d/%d in %.1fs", label, exc, attempt, attempts - 1, delay)
            await asyncio.sleep(delay)
    assert last is not None
    raise last
