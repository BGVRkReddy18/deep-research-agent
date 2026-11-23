"""Provider-agnostic LLM layer.

* `ChatProvider`  - one OpenAI-compatible endpoint (Gemini, Groq, OpenRouter, ...),
                    with its own rate limiter and retry policy.
* `LLMRouter`     - tries providers in order and fails over when one is rate-limited
                    or down. Agents only ever talk to the router.
* `complete_json` - asks for JSON, validates it against a Pydantic model, and repairs
                    malformed output by feeding the validation error back to the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .config import ProviderConfig
from .models import UsageStats
from .resilience import PermanentError, RateLimiter, estimate_tokens, retry_async

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


class LLMError(Exception):
    pass


class AllProvidersFailed(LLMError):
    pass


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Completion:
    text: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatBackend(Protocol):
    name: str

    async def complete(
        self, messages: list[Message], *, temperature: float, max_tokens: int, json_mode: bool
    ) -> Completion: ...


class RequestTooLarge(LLMError):
    """This request can never fit this model's per-minute limits. Skip to another model; don't retry."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (PermanentError, RequestTooLarge)):
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


# Rate limiters live for the whole process, keyed by (endpoint, model, key), so back-to-back runs
# (e.g. several visitors of the web app) share one view of each model's quota, as the API does.
_LIMITERS: dict[tuple[str, str, str], dict[str, RateLimiter]] = {}


def _limiters_for(cfg: ProviderConfig) -> dict[str, RateLimiter]:
    key = (cfg.base_url, cfg.model, hashlib.sha256(cfg.api_key.encode()).hexdigest()[:16])
    if key not in _LIMITERS:
        _LIMITERS[key] = {
            "requests": RateLimiter(cfg.requests_per_minute),
            "tokens": RateLimiter(cfg.tokens_per_minute),
            "output": RateLimiter(cfg.output_tokens_per_minute or 10**9),
        }
    return _LIMITERS[key]


class ChatProvider:
    """A single OpenAI-compatible chat model with quota pacing and retries.

    Paces three per-minute budgets, because free tiers enforce all of them:
    requests, total tokens (prompt + reserved output), and, for some models, output tokens.
    """

    def __init__(self, cfg: ProviderConfig, *, timeout: float = 90.0, max_retries: int = 3) -> None:
        self.cfg = cfg
        self.name = f"{cfg.name}:{cfg.model}"
        self._client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=timeout, max_retries=0)
        lim = _limiters_for(cfg)
        self._requests, self._tokens, self._output = lim["requests"], lim["tokens"], lim["output"]
        self._max_retries = max_retries

    @property
    def output_cap(self) -> int | None:
        """Largest max_tokens this model accepts per request (None = no special cap)."""
        return self.cfg.output_tokens_per_minute or None

    def _budget(self, prompt_tokens: int, max_tokens: int) -> tuple[int, int]:
        out = min(max_tokens, self.output_cap) if self.output_cap else max_tokens
        return prompt_tokens + out, out

    def wait_time(self, prompt_tokens: int, max_tokens: int) -> float:
        """Seconds this model would make the request wait for quota (0 = can start now)."""
        total, out = self._budget(prompt_tokens, max_tokens)
        return max(self._requests.wait_time(1), self._tokens.wait_time(total), self._output.wait_time(out))

    def try_reserve(self, prompt_tokens: int, max_tokens: int) -> bool:
        """Claim quota for one request now, or claim nothing. Synchronous, so atomic in asyncio."""
        if self.wait_time(prompt_tokens, max_tokens) > 0:
            return False
        total, out = self._budget(prompt_tokens, max_tokens)
        return self._requests.try_acquire(1) and self._tokens.try_acquire(total) and self._output.try_acquire(out)

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        reserved: bool = False,
    ) -> Completion:
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        total, max_out = self._budget(prompt_tokens, max_tokens)
        first_attempt = True

        async def call() -> Completion:
            nonlocal first_attempt
            # If the router already reserved quota, the first attempt goes straight to the API.
            if not (reserved and first_attempt):
                await self._requests.acquire()
                await self._tokens.acquire(total)
                await self._output.acquire(max_out)
            first_attempt = False
            kwargs: dict = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if self.cfg.extra_body:
                kwargs["extra_body"] = dict(self.cfg.extra_body)
            try:
                resp = await self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    temperature=temperature,
                    max_tokens=max_out,
                    **kwargs,
                )
            except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as exc:
                raise PermanentError(f"{self.cfg.name}: {exc}") from exc
            except openai.RateLimitError as exc:
                if "request too large" in str(exc).lower():
                    raise RequestTooLarge(f"{self.name}: {str(exc).splitlines()[0][:200]}") from exc
                raise
            if not resp.choices:
                raise LLMError(f"{self.cfg.name}: empty response")
            text = _THINK.sub("", resp.choices[0].message.content or "")
            if not text.strip():
                raise LLMError(f"{self.cfg.name}: blank completion")
            usage = resp.usage
            return Completion(
                text=text,
                provider=self.name,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )

        return await retry_async(
            call,
            attempts=self._max_retries,
            is_retryable=lambda e: _is_retryable(e) or (isinstance(e, LLMError) and not isinstance(e, RequestTooLarge)),
            label=self.name,
        )


@dataclass
class LLMRouter:
    """Quota-aware load balancer + failover across models/providers.

    Free tiers limit tokens-per-minute *per model*, so N models give ~N x the throughput.
    For every call the router picks the backend that can start soonest, reserving its quota
    atomically so parallel workers spread out instead of queueing on one model:

        score(backend) = seconds_until_quota_free + prefer_primary * rank

    `prefer_primary` > 0 biases quality-critical calls (planning, the final report) toward
    the strongest model (rank 0), accepting up to that many seconds of extra wait per rank.
    If a call fails, the next-best backend is tried; models that fail permanently (retired,
    bad key) are disabled for the rest of the run.
    """

    backends: list[ChatBackend]
    usage: UsageStats = field(default_factory=UsageStats)
    _disabled: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.backends:
            raise LLMError(
                "No LLM provider configured. Set GEMINI_API_KEY or GROQ_API_KEY "
                "(see .env.example), or run with --mock for an offline demo."
            )

    def _pick(
        self, candidates: list[ChatBackend], prompt_tokens: int, max_tokens: int, prefer_primary: float
    ) -> tuple[ChatBackend, bool]:
        def score(b: ChatBackend) -> tuple[float, int]:
            rank = self.backends.index(b)
            wait = b.wait_time(prompt_tokens, max_tokens) if hasattr(b, "wait_time") else 0.0
            cap = getattr(b, "output_cap", None)
            truncation_penalty = 120.0 if cap and cap < max_tokens * 0.8 else 0.0  # long answers need room
            return wait + prefer_primary * rank + truncation_penalty, rank

        best = min(candidates, key=score)
        reserved = hasattr(best, "try_reserve") and best.try_reserve(prompt_tokens, max_tokens)
        return best, reserved

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
        prefer_primary: float = 0.0,
    ) -> Completion:
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        errors: list[str] = []
        tried: set[str] = set()
        while candidates := [b for b in self.backends if b.name not in tried and b.name not in self._disabled]:
            backend, reserved = self._pick(candidates, prompt_tokens, max_tokens, prefer_primary)
            tried.add(backend.name)
            try:
                result = await backend.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    **({"reserved": True} if reserved else {}),
                )
            except Exception as exc:  # noqa: BLE001 - fall through to the next backend
                log.warning("backend %s failed: %s", backend.name, exc)
                if isinstance(exc, PermanentError):
                    self._disabled.add(backend.name)
                errors.append(f"{backend.name}: {type(exc).__name__}: {str(exc).splitlines()[0][:180]}")
                continue
            self.usage.llm_calls += 1
            self.usage.prompt_tokens += result.prompt_tokens
            self.usage.completion_tokens += result.completion_tokens
            self.usage.calls_by_model[result.provider] = self.usage.calls_by_model.get(result.provider, 0) + 1
            if result.provider not in self.usage.providers_used:
                self.usage.providers_used.append(result.provider)
            return result
        raise AllProvidersFailed(
            "Could not reach any AI model. Check your API keys and model names in .env, or try again in a minute (free tiers have per-minute limits).\n" + "\n".join(f"- {e}" for e in errors)
        )

    async def complete_json(
        self,
        messages: list[Message],
        schema: type[M],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        repair_attempts: int = 2,
        prefer_primary: float = 0.0,
    ) -> M:
        """Get a response that validates against `schema`, repairing bad JSON if needed."""
        convo = list(messages)
        last_error = ""
        for _ in range(repair_attempts + 1):
            result = await self.complete(
                convo, temperature=temperature, max_tokens=max_tokens, json_mode=True, prefer_primary=prefer_primary
            )
            try:
                return schema.model_validate(extract_json(result.text))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:1500]
                convo = list(messages) + [
                    Message("assistant", result.text[:4000]),
                    Message(
                        "user",
                        "That response was not valid JSON for the required schema. "
                        f"Error:\n{last_error}\n\nReply again with ONLY the corrected JSON object.",
                    ),
                ]
        raise LLMError(f"Model never produced valid {schema.__name__} JSON: {last_error}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)  # reasoning leaked into content


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    candidates = [text]
    candidates += _FENCE.findall(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in model output")


def build_router(provider_configs: list[ProviderConfig], *, timeout: float, max_retries: int) -> LLMRouter:
    return LLMRouter([ChatProvider(c, timeout=timeout, max_retries=max_retries) for c in provider_configs])
