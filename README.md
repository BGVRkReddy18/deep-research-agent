# 🔎 Deep Research Agent

A **multi-agent research assistant** that turns a hard question into a **cited research report**.
A planner agent splits the question, parallel worker agents search the web, read pages and run a
**hybrid RAG pipeline**, and a synthesizer writes a report where every claim links back to the exact
source passage.

Runs entirely on **free tiers** (Gemini / Groq + Tavily / DuckDuckGo) and deploys for **$0**.

```
                       ┌──────────────┐
   question ──────────►│   Planner    │  splits into 2-6 focused sub-questions + search queries
                       └──────┬───────┘
          ┌───────────────────┼───────────────────┐        (parallel, bounded, time-limited)
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  Worker 1   │     │  Worker 2   │     │  Worker N   │
   │ search      │     │             │     │             │
   │ fetch pages │     │    ...      │     │    ...      │
   │ chunk       │     │             │     │             │
   │ embed ─► Chroma   │             │     │             │
   │ BM25 + vector (RRF)│            │     │             │
   │ cited claims│     │             │     │             │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
          └───────────────────┼───────────────────┘
                              ▼
                       ┌──────────────┐
                       │ Synthesizer  │  themed report, [n] citations verified + renumbered
                       └──────┬───────┘
                              ▼
                 report.md  +  sources with exact evidence passages
```

## Features

| Area | What it does |
|---|---|
| **Orchestration** | Planner → N parallel workers (`asyncio`, semaphore-bounded, per-worker timeout) → synthesizer |
| **RAG** | Paragraph-aware chunking with overlap · local MiniLM embeddings (ONNX, no API quota) · ChromaDB · BM25 · Reciprocal Rank Fusion · per-source diversity cap |
| **Grounded citations** | Workers may only cite passage IDs they were shown; invalid IDs are stripped, unsupported claims dropped, report citations verified and renumbered |
| **Context management** | Workers see only top-k retrieved passages, not whole pages; the synthesizer sees only distilled claims, not raw text |
| **Error recovery** | Retries with backoff + `Retry-After` · LLM provider failover · JSON self-repair · search fallback · snippet fallback · BM25-only fallback · latency budgets for optional stages · partial reports |
| **Free-tier aware** | Quota-aware load balancing across models (each has its own limit) · paces requests, tokens and output tokens per model · per-visitor daily quota · bring-your-own-key |
| **Observability** | Live progress events, token / call / page stats, full JSON trace of plan, findings and evidence |
| **Testable offline** | `--mock` mode with a fake LLM and search: the whole pipeline runs with no keys and no network |

## Quick start

```bash
cd deep-research-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**Try it with no keys at all (offline demo):**

```bash
python -m research_agent --mock "Is solar cheaper than nuclear?"
```

**Real research:** copy `.env.example` to `.env` and add at least one free LLM key:

```bash
cp .env.example .env
```

```bash
python -m research_agent "What are the trade-offs between RAG and fine-tuning for enterprise chatbots?"
```

Reports are saved to `reports/` as Markdown plus a JSON trace.

**Web app:**

```bash
streamlit run app.py
```

## Getting the free API keys

| Key | Where | Needed? |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) → *Create API Key* | Recommended main LLM (fast, `openai/gpt-oss-120b`) |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → *Create API key* | Optional backup LLM (and API embeddings if you choose them) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) → copy the key on the dashboard | Optional; better search than the DuckDuckGo fallback |

None of these need a credit card. Free-tier limits change over time; if you hit 429 errors, lower
`GEMINI_RPM` / `GROQ_RPM` in `.env`.

## Deploy for free

### Option A: Streamlit Community Cloud (easiest)
1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → *Create app* → choose the repo and `app.py`.
3. Under *Advanced settings → Secrets*, add your keys:
   ```toml
   GROQ_API_KEY = "..."
   GEMINI_API_KEY = "..."
   TAVILY_API_KEY = "..."
   ```
4. Deploy. You get a public `*.streamlit.app` URL.

### Option B: Hugging Face Spaces
1. Create a new Space → SDK **Streamlit**.
2. Push this folder to the Space repo.
3. Add your keys under *Settings → Variables and secrets*.

Visitors get `DAILY_REPORTS_PER_USER` free reports per day on your keys, or unlimited if they paste
their own free key in the sidebar. Their keys are used only for that request and never stored.

## Project layout

```
research_agent/
  config.py          settings + provider presets (all env-driven)
  llm.py             OpenAI-compatible client, provider router/failover, JSON repair
  resilience.py      retry with backoff, sliding-window rate limiter
  search.py          Tavily → DuckDuckGo fallback chain
  fetch.py           concurrent page download + main-text extraction (trafilatura)
  rag/
    chunking.py      paragraph/sentence-aware chunking with overlap
    embeddings.py    Gemini | sentence-transformers | hashing embedders
    store.py         ChromaDB | numpy vector stores
    retriever.py     hybrid BM25 + dense retrieval with RRF
  agents/
    prompts.py       every prompt in one place
    planner.py       question → sub-questions
    worker.py        sub-question → search → read → RAG → cited claims
    synthesizer.py   claims → report, citation verification + renumbering
  orchestrator.py    wires it together, parallelism, failure isolation
  mock.py            offline LLM + search doubles
  cli.py             rich terminal UI
app.py               Streamlit web app
tests/               44 tests: RAG, agents, failure modes, end-to-end, UI
```

## Performance on free tiers

Groq's free tier allows ~8,000 tokens/minute **per model**, and one report needs ~13,000, so a
single model forces the run to wait for the next minute. The router load-balances across three
models, reserving quota atomically so parallel workers spread out instead of queueing, while
planning and the final report still prefer the strongest model.

| Question | One model | Three models (load-balanced) |
|---|---|---|
| Is solar cheaper than nuclear for new generation? | 130 s | **63 s** |
| How many international centuries has Virat Kohli scored? | 121 s | **62 s** |

Measured live with fresh quota; back-to-back runs share the same per-minute budget and take longer.

Lessons found in live runs, each now handled in code and covered by a test:
- Gemini's free embedding tier counts **every text as a request, ~1,000 a day**: about a dozen reports. The deployed
  app stalled on it (250 s per report), so embeddings now run locally (MiniLM via ONNX, ~2 s for 64 passages),
  preloaded at app start, with a latency budget that uses BM25 rather than waiting on any slow optional stage.
- `qwen3` has a separate **1,000 output-tokens/minute** cap, so requests are clamped and long outputs are routed elsewhere.
- Quota must be tracked **per process, not per run**, or the second visitor's run thinks the quota is fresh.
- `gpt-oss` cites as `【1】`, so citations are normalised before verification.
- Models sometimes emit broken JSON inside valid JSON, so planner queries are sanitised.

## Design decisions

- **No agent framework.** Orchestration is ~150 lines of plain `asyncio`, so every retry,
  timeout and fallback is explicit and testable.
- **OpenAI-compatible protocol for every provider.** Gemini, Groq and OpenRouter all expose it, so one
  client covers all of them and adding a provider means adding a config preset.
- **Hybrid retrieval over pure vectors.** Research questions are full of names, numbers and years,
  which BM25 matches exactly and embeddings blur. RRF fuses both without score calibration.
- **Citations are enforced in code, not just requested in prompts.** The model can only cite IDs it was
  shown, and anything else is removed before the user sees it.
- **Local embeddings by default.** A free embedding API sounds lighter, but its daily quota caps the whole app at
  ~12 reports. ONNX MiniLM ships with ChromaDB, needs no extra dependency, and fits a free host's RAM.
  `EMBEDDING_BACKEND=gemini` switches to the API; `local` runs any sentence-transformers model.

## Tests

```bash
pytest -q
```

They cover chunking, both vector stores, hybrid retrieval, BM25 fallback when embeddings fail,
provider failover, quota-aware load balancing, output-token caps, JSON repair, hallucinated-citation removal, citation renumbering, planner failure,
total LLM outage, input validation, and a full offline run through the web UI.

## Roadmap

- [ ] Cross-encoder reranker stage
- [ ] Evaluation set: retrieval recall, citation precision, faithfulness (LLM-as-judge)
- [ ] Semantic cache of past research in pgvector (Supabase free tier)
- [ ] FastAPI + Server-Sent Events backend and a Next.js frontend
- [ ] Shareable report links
