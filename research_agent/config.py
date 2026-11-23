"""Runtime configuration, loaded from environment variables (and a local .env file).

Every knob lives here so the rest of the code never reads os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class ProviderConfig:
    """One OpenAI-compatible chat provider (Gemini, Groq, OpenRouter, ...)."""

    name: str
    base_url: str
    api_key: str
    model: str
    requests_per_minute: int
    tokens_per_minute: int = 1_000_000
    output_tokens_per_minute: int = 0  # 0 = no separate output limit
    extra_body: dict = field(default_factory=dict, hash=False, compare=False)  # model-specific params

    @property
    def reasoning_effort(self) -> str:
        return self.extra_body.get("reasoning_effort", "")


def model_params(model: str) -> dict:
    """Per-model request tweaks that keep reasoning models fast and inside free token limits."""
    m = model.lower()
    if "gpt-oss" in m:
        return {"reasoning_effort": _env("REASONING_EFFORT", "low")}
    if "qwen3" in m:
        return {"reasoning_effort": "none"}  # no thinking: its small output budget goes to the answer
    return {}


def output_limit(model: str) -> int:
    """Known per-minute *output* token caps on free tiers (found the hard way: HTTP 429 'OTPM')."""
    return 1000 if "qwen3" in model.lower() else 0


# Known free-tier friendly providers. All speak the OpenAI chat-completions protocol,
# so a single client implementation covers every one of them.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-flash-latest",
        "rpm_env": "GEMINI_RPM",
        "default_rpm": "10",
        "tpm_env": "GEMINI_TPM",
        "default_tpm": "200000",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        # Comma-separated: each Groq model has its OWN free tokens-per-minute quota, so listing
        # several multiplies throughput. The first is the strongest and is preferred for planning
        # and writing the report.
        "default_model": "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b",
        "rpm_env": "GROQ_RPM",
        "default_rpm": "30",
        "tpm_env": "GROQ_TPM",
        "default_tpm": "7500",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "rpm_env": "OPENROUTER_RPM",
        "default_rpm": "15",
        "tpm_env": "OPENROUTER_TPM",
        "default_tpm": "100000",
    },
}


def build_providers(name: str, api_key: str | None = None) -> list[ProviderConfig]:
    """One ProviderConfig per configured model of this provider (each model has its own quota)."""
    preset = PROVIDER_PRESETS.get(name)
    if preset is None:
        return []
    key = api_key or _env(preset["key_env"])
    if not key:
        return []
    models = [m.strip() for m in _env(preset["model_env"], preset["default_model"]).split(",") if m.strip()]
    return [
        ProviderConfig(
            name=name,
            base_url=preset["base_url"],
            api_key=key,
            model=model,
            requests_per_minute=_int(preset["rpm_env"], int(preset["default_rpm"])),
            tokens_per_minute=_int(preset["tpm_env"], int(preset["default_tpm"])),
            output_tokens_per_minute=output_limit(model),
            extra_body=model_params(model),
        )
        for model in dict.fromkeys(models)
    ]


def build_provider(name: str, api_key: str | None = None) -> ProviderConfig | None:
    """The provider's primary (first) model, or None if it is not configured."""
    configs = build_providers(name, api_key)
    return configs[0] if configs else None


@dataclass(frozen=True)
class Settings:
    # LLM providers, tried in this order. The first one that works wins; the rest are fallbacks.
    provider_order: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip() for p in _env("LLM_PROVIDERS", "groq,gemini,openrouter").split(",") if p.strip()
        )
    )

    # Embeddings: "gemini" (free API, low RAM), "local" (sentence-transformers), or "hash" (offline).
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "gemini"))
    gemini_embedding_model: str = field(
        default_factory=lambda: _env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    )
    local_embedding_model: str = field(
        default_factory=lambda: _env("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Vector store: "chroma" or "memory" (pure numpy). "auto" picks chroma if installed.
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "auto"))

    # Search
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))

    # Research shape
    max_sub_questions: int = field(default_factory=lambda: _int("MAX_SUB_QUESTIONS", 4))
    results_per_query: int = field(default_factory=lambda: _int("RESULTS_PER_QUERY", 5))
    max_pages_per_worker: int = field(default_factory=lambda: _int("MAX_PAGES_PER_WORKER", 4))
    chunks_per_worker: int = field(default_factory=lambda: _int("CHUNKS_PER_WORKER", 6))
    worker_concurrency: int = field(default_factory=lambda: _int("WORKER_CONCURRENCY", 3))

    # Chunking (measured in characters; ~4 chars per token)
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 1200))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 200))

    # Network
    fetch_timeout: float = field(default_factory=lambda: _float("FETCH_TIMEOUT", 12.0))
    llm_timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT", 90.0))
    max_retries: int = field(default_factory=lambda: _int("MAX_RETRIES", 3))

    # Public-app guard rails
    daily_reports_per_user: int = field(default_factory=lambda: _int("DAILY_REPORTS_PER_USER", 3))
    max_question_chars: int = field(default_factory=lambda: _int("MAX_QUESTION_CHARS", 500))


def get_settings() -> Settings:
    return Settings()
