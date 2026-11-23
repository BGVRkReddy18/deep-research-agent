"""Quota-aware load balancing across models that each have their own tokens-per-minute limit."""

import asyncio

import pytest

from research_agent.config import build_providers
from research_agent.llm import Completion, LLMRouter, Message
from research_agent.resilience import PermanentError, RateLimiter


class QuotaModel:
    """Fake model with a real per-model token limiter, like a Groq free-tier model."""

    def __init__(self, name: str, tpm: int, fail: Exception | None = None) -> None:
        self.name = name
        self.tokens = RateLimiter(tpm, period=60.0)
        self.fail = fail
        self.calls = 0

    def wait_time(self, prompt_tokens: int, max_tokens: int) -> float:
        return self.tokens.wait_time(prompt_tokens + max_tokens)

    def try_reserve(self, prompt_tokens: int, max_tokens: int) -> bool:
        return self.tokens.try_acquire(prompt_tokens + max_tokens)

    async def complete(self, messages, *, temperature, max_tokens, json_mode, reserved=False):
        if self.fail:
            raise self.fail
        if not reserved:
            await self.tokens.acquire(sum(len(m.content) // 4 + 1 for m in messages) + max_tokens)
        self.calls += 1
        await asyncio.sleep(0.01)
        return Completion(text="ok", provider=self.name)


MSG = [Message("user", "x" * 400)]  # ~100 tokens + max_tokens below = ~1100 per call


async def test_parallel_calls_spread_across_models_instead_of_queueing():
    models = [QuotaModel("big", 2500), QuotaModel("mid", 2500), QuotaModel("small", 2500)]
    router = LLMRouter(models)
    # 6 parallel calls x ~1100 tokens = 6600 tokens: more than one model's 2500/min,
    # but it fits across three models, so nothing should have to wait a minute.
    results = await asyncio.wait_for(
        asyncio.gather(*(router.complete(MSG, max_tokens=1000) for _ in range(6))), timeout=5
    )
    assert len(results) == 6
    assert [m.calls for m in models] == [2, 2, 2]
    assert sum(router.usage.calls_by_model.values()) == 6


async def test_primary_is_used_while_it_has_quota():
    models = [QuotaModel("big", 10_000), QuotaModel("small", 10_000)]
    router = LLMRouter(models)
    for _ in range(3):
        await router.complete(MSG, max_tokens=1000)
    assert models[0].calls == 3 and models[1].calls == 0


def test_prefer_primary_accepts_a_short_wait_for_the_best_model():
    big, small = QuotaModel("big", 2000), QuotaModel("small", 2000)
    router = LLMRouter([big, small])
    assert big.tokens.try_acquire(1500)  # big is busy: a 1100-token call must wait ~60s
    pick, _ = router._pick([big, small], 100, 1000, prefer_primary=0.0)
    assert pick is small  # workers: take whichever model is free now
    pick, _ = router._pick([big, small], 100, 1000, prefer_primary=90.0)
    assert pick is big  # report writing: worth waiting for the strongest model


async def test_retired_model_is_disabled_and_skipped():
    dead = QuotaModel("retired", 10_000, fail=PermanentError("404 model does not exist"))
    alive = QuotaModel("alive", 10_000)
    router = LLMRouter([dead, alive])
    await router.complete(MSG, max_tokens=100)
    await router.complete(MSG, max_tokens=100)
    assert "retired" in router._disabled
    assert alive.calls == 2


@pytest.mark.parametrize("cost", [1, 500])
def test_rate_limiter_wait_time_matches_try_acquire(cost):
    limiter = RateLimiter(1000, period=60.0)
    assert limiter.wait_time(cost) == 0 and limiter.try_acquire(cost)
    assert limiter.try_acquire(1000 - cost)
    assert limiter.wait_time(cost) > 50  # full: must wait for the window
    assert not limiter.try_acquire(cost)


def test_groq_model_list_becomes_one_backend_per_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b, qwen/qwen3.8-27b ,openai/gpt-oss-120b")
    configs = build_providers("groq")
    assert [c.model for c in configs] == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]  # deduped, trimmed
    assert configs[0].extra_body == {"reasoning_effort": "low"}
    assert configs[1].extra_body == {"reasoning_effort": "none"}
    assert configs[1].output_tokens_per_minute == 1000 and configs[0].output_tokens_per_minute == 0


def _provider(model: str, otpm: int = 0):
    from research_agent.config import ProviderConfig
    from research_agent.llm import ChatProvider

    return ChatProvider(ProviderConfig(
        name="groq", base_url="https://example.invalid/v1", api_key=f"k-{model}", model=model,
        requests_per_minute=30, tokens_per_minute=8000, output_tokens_per_minute=otpm,
    ))


def test_output_capped_model_clamps_max_tokens_and_is_avoided_for_long_outputs():
    capped = _provider("qwen-test", otpm=1000)
    assert capped.output_cap == 1000
    assert capped._budget(500, 2500) == (1500, 1000)  # request is clamped to the cap
    free = _provider("big-test")
    router = LLMRouter([capped, free])
    pick, _ = router._pick([capped, free], 500, 2500, prefer_primary=0.0)
    assert pick is free  # a 2500-token report would be truncated on the capped model


def test_limiters_are_shared_across_runs_for_the_same_model_and_key():
    a, b = _provider("shared-test"), _provider("shared-test")
    assert a._tokens is b._tokens  # a second research run sees the first run's usage
    assert a.try_reserve(5000, 1000)  # 6000 of the 8000 tokens/min
    assert b.wait_time(5000, 1000) > 0


def test_rate_limiter_works_across_event_loops():
    limiter = RateLimiter(10, period=60.0)
    asyncio.run(limiter.acquire(3))
    asyncio.run(limiter.acquire(3))  # a fresh loop (as in the web app) must not crash on the old lock
    assert limiter.wait_time(5) > 0


async def test_request_too_large_fails_over_without_retrying():
    from research_agent.llm import RequestTooLarge

    too_big = QuotaModel("capped", 10_000, fail=RequestTooLarge("Request too large ... OTPM"))
    ok = QuotaModel("ok", 10_000)
    router = LLMRouter([too_big, ok])
    result = await router.complete(MSG, max_tokens=1200)
    assert result.provider == "ok"
    assert "capped" not in router._disabled  # only this request was too big, the model still works
