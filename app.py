"""Streamlit web app: live multi-agent research with cited reports.

    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import queue
import threading
import time
from collections import defaultdict
from datetime import date

import streamlit as st

from research_agent.config import PROVIDER_PRESETS, build_provider, get_settings
from research_agent.models import ProgressEvent, Report, Stage
from research_agent.orchestrator import UserKeys, research

st.set_page_config(page_title="Deep Research Agent", page_icon="🔎", layout="wide")

EXAMPLES = [
    "Is solar now cheaper than nuclear power for new electricity generation?",
    "What are the main techniques for reducing LLM hallucinations in RAG systems?",
    "How do the Indian and US startup ecosystems compare in 2026?",
]

ICONS = {
    Stage.WORKER_START: "🚀", Stage.SEARCHING: "🔎", Stage.FETCHING: "📄", Stage.RETRIEVING: "🧩",
    Stage.ANALYZING: "🧠", Stage.WORKER_DONE: "✅", Stage.WORKER_FAILED: "❌", Stage.WARNING: "⚠️",
}


# ---------------------------------------------------------------- quota (per visitor, per day)

@st.cache_resource
def _quota_store() -> dict[str, int]:
    return defaultdict(int)


def _visitor_id() -> str:
    ip = getattr(st.context, "ip_address", None)
    if not ip:
        headers = getattr(st.context, "headers", {}) or {}
        ip = (headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    if "sid" not in st.session_state:
        st.session_state.sid = str(time.time_ns())
    return f"{date.today()}:{ip or st.session_state.sid}"


def remaining_quota(limit: int) -> int:
    return max(0, limit - _quota_store()[_visitor_id()])


def use_quota() -> None:
    _quota_store()[_visitor_id()] += 1


# ---------------------------------------------------------------- background runner

def start_research(question: str, keys: UserKeys, settings, mock: bool) -> tuple[queue.Queue, dict]:
    """Run the async pipeline in a thread; progress events stream back through a queue."""
    events: queue.Queue = queue.Queue()
    result: dict = {}

    async def emit(ev: ProgressEvent) -> None:
        events.put(ev)

    def target() -> None:
        try:
            components = None
            if mock:
                from research_agent.mock import mock_components

                components = mock_components()
            result["report"] = asyncio.run(
                research(question, emit=emit, settings=settings, keys=keys, components=components)
            )
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            events.put(None)

    threading.Thread(target=target, daemon=True).start()
    return events, result


def live_progress(events: queue.Queue) -> None:
    status = st.status("Planning research…", expanded=True)
    with status:
        plan_box = st.empty()
        worker_boxes: dict[int, st.delta_generator.DeltaGenerator] = {}
        worker_logs: dict[int, list[str]] = defaultdict(list)
        cols = None
        while True:
            ev = events.get()
            if ev is None:
                break
            if ev.stage is Stage.PLAN_READY:
                subs = ev.data["plan"]["sub_questions"]
                plan_box.markdown(
                    "**Plan**\n\n" + "\n".join(f"{s['id']}. {s['question']}" for s in subs)
                )
                cols = st.columns(min(len(subs), 4))
                for i, s in enumerate(subs):
                    with cols[i % len(cols)]:
                        st.markdown(f"**Worker {s['id']}**")
                        worker_boxes[s["id"]] = st.empty()
                status.update(label=f"{len(subs)} workers researching in parallel…")
            elif ev.worker_id and ev.worker_id in worker_boxes:
                worker_logs[ev.worker_id].append(f"{ICONS.get(ev.stage, '•')} {ev.message}")
                worker_boxes[ev.worker_id].markdown("\n\n".join(worker_logs[ev.worker_id][-6:]))
            elif ev.stage is Stage.SYNTHESIZING:
                status.update(label="Synthesizer is writing the cited report…")
            elif ev.stage is Stage.WARNING:
                st.caption(f"⚠️ {ev.message}")
            elif ev.stage is Stage.DONE:
                status.update(label=ev.message, state="complete", expanded=False)


# ---------------------------------------------------------------- report view

def show_report(report: Report) -> None:
    u = report.usage
    m = st.columns(5)
    m[0].metric("Time", f"{report.duration_seconds}s")
    m[1].metric("Sub-questions", len(report.plan.sub_questions))
    m[2].metric("Sources cited", len(report.sources))
    m[3].metric("LLM calls", u.llm_calls)
    m[4].metric("Tokens", f"{(u.prompt_tokens + u.completion_tokens):,}")
    for w in report.warnings:
        st.warning(w, icon="⚠️")

    tab_report, tab_sources, tab_trace = st.tabs(["📝 Report", "📚 Sources & evidence", "🧭 Agent trace"])
    with tab_report:
        st.markdown(report.markdown)
        d1, d2, _ = st.columns([1, 1, 4])
        d1.download_button("⬇️ Markdown", report.markdown, file_name="report.md", mime="text/markdown")
        d2.download_button(
            "⬇️ JSON", report.model_dump_json(indent=2), file_name="report.json", mime="application/json"
        )
    with tab_sources:
        for s in report.sources:
            with st.expander(f"[{s.number}] {s.title}"):
                st.markdown(f"🔗 [{s.url}]({s.url})")
                for p in s.passages:
                    st.markdown(f"> {p}")
    with tab_trace:
        for f in report.findings:
            icon = "✅" if f.ok else "❌"
            with st.expander(f"{icon} Sub-question {f.sub_question_id}: {f.sub_question}", expanded=False):
                if f.error:
                    st.error(f.error)
                if f.summary:
                    st.markdown(f"**Summary:** {f.summary}")
                for c in f.claims:
                    st.markdown(f"- {c.statement}  \n  <small>evidence: {', '.join(c.evidence_ids)}</small>", unsafe_allow_html=True)
                if f.gaps:
                    st.caption(f"Gaps: {f.gaps}")
        models = ", ".join(f"{m} ×{n}" for m, n in u.calls_by_model.items()) or "-"
        st.caption(f"Models used: {models} · searches: {u.searches} · "
                   f"pages read: {u.pages_fetched} (snippet fallbacks: {u.pages_failed})")


# ---------------------------------------------------------------- page

def main() -> None:
    base = get_settings()
    server_has_llm = any(build_provider(name) for name in PROVIDER_PRESETS)

    with st.sidebar:
        st.header("⚙️ Settings")
        depth = st.slider("Research depth (sub-questions)", 2, 6, base.max_sub_questions)
        st.divider()
        st.subheader("🔑 Your own API keys (optional)")
        st.caption("Free keys skip the daily limit. They are used only for your request and never stored.")
        gemini = st.text_input("Gemini API key", type="password", help="aistudio.google.com/apikey")
        groq = st.text_input("Groq API key", type="password", help="console.groq.com/keys")
        tavily = st.text_input("Tavily API key", type="password", help="app.tavily.com")
        st.divider()
        mock = st.toggle("Offline demo mode", value=not server_has_llm,
                         help="Runs with sample data and a fake LLM. No keys needed.")
        st.divider()
        st.markdown(
            "**How it works**\n\n"
            "1. 🧭 **Planner** splits your question\n"
            "2. 🔎 **Workers** search, read & retrieve (hybrid RAG) in parallel\n"
            "3. ✍️ **Synthesizer** writes a report where every claim is cited"
        )

    st.title("🔎 Deep Research Agent")
    st.markdown("Ask a hard question. A team of AI agents researches the web in parallel and writes a **cited report**.")

    keys = UserKeys(gemini=gemini.strip(), groq=groq.strip(), tavily=tavily.strip())
    byok = bool(keys.gemini or keys.groq)
    if not mock and not server_has_llm and not byok:
        st.info("No server API key is configured. Add your free Gemini or Groq key in the sidebar, or turn on offline demo mode.")

    if "question" not in st.session_state:
        st.session_state.question = ""
    with st.expander("💡 What kind of questions work best?"):
        st.markdown(
            "This agent shines on questions that need **several sources and some analysis**:\n\n"
            "| ✅ Great | ⚠️ Works, but overkill |\n|---|---|\n"
            "| *How has Virat Kohli's ODI form changed since 2023, and how does he compare with Babar Azam and Joe Root?* "
            "| *How many centuries has Virat Kohli scored?* |\n"
            "| *Is solar cheaper than nuclear for new power plants, and what hidden costs apply?* | *What is the capital of France?* |\n"
            "| *What are the trade-offs between RAG and fine-tuning for a support chatbot?* | *Define machine learning.* |\n\n"
            "**Tips:** compare things, ask *why* or *how*, name a time period, and ask about trade-offs or evidence. "
            "Simple facts still work, but take 1-2 minutes, and fast-changing numbers (live scores, prices) "
            "are only as fresh as the pages the search finds."
        )

    ex_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(ex_cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state.question = ex

    with st.form("ask"):
        question = st.text_area(
            "Your research question", key="question", height=90, max_chars=base.max_question_chars,
            placeholder="e.g. What are the trade-offs between fine-tuning and RAG for enterprise chatbots?",
        )
        quota = remaining_quota(base.daily_reports_per_user)
        unlimited = byok or mock
        caption = "Unlimited (your keys / demo mode)" if unlimited else f"{quota} of {base.daily_reports_per_user} free reports left today"
        submitted = st.form_submit_button("Research ▶", type="primary", help=caption)
        st.caption(caption)

    if submitted:
        q = question.strip()
        if len(q) < 10:
            st.error("Please ask a fuller question (at least 10 characters).")
        elif not unlimited and quota <= 0:
            st.error("Daily free limit reached. Add your own free API key in the sidebar, or come back tomorrow.")
        elif not mock and not server_has_llm and not byok:
            st.error("No LLM key available. Add a key in the sidebar or enable offline demo mode.")
        else:
            settings = dataclasses.replace(base, max_sub_questions=depth)
            events, result = start_research(q, keys, settings, mock)
            live_progress(events)
            if "error" in result:
                st.error("Research failed.", icon="🚫")
                st.code(result["error"], language=None)
            else:
                if not unlimited:
                    use_quota()
                st.session_state.report = result["report"]

    if report := st.session_state.get("report"):
        st.divider()
        show_report(report)


main()
