"""Command-line interface.

    python -m research_agent "Is solar cheaper than nuclear in 2026?"
    python -m research_agent --mock "anything"      # offline demo, no API keys
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .models import ProgressEvent, Report, Stage
from .orchestrator import research

console = Console()

_STYLE = {
    Stage.PLANNING: ("🧭", "cyan"),
    Stage.PLAN_READY: ("📋", "cyan"),
    Stage.WORKER_START: ("🚀", "blue"),
    Stage.SEARCHING: ("🔎", "blue"),
    Stage.FETCHING: ("📄", "blue"),
    Stage.RETRIEVING: ("🧩", "magenta"),
    Stage.ANALYZING: ("🧠", "magenta"),
    Stage.WORKER_DONE: ("✅", "green"),
    Stage.WORKER_FAILED: ("❌", "red"),
    Stage.SYNTHESIZING: ("✍️ ", "yellow"),
    Stage.DONE: ("🏁", "green"),
    Stage.WARNING: ("⚠️ ", "yellow"),
}


async def _print_event(ev: ProgressEvent) -> None:
    icon, color = _STYLE.get(ev.stage, ("•", "white"))
    who = f"[bold]worker {ev.worker_id}[/] " if ev.worker_id else ""
    console.print(f"{icon} {who}[{color}]{ev.message}[/]")
    if ev.stage is Stage.PLAN_READY:
        for sq in ev.data["plan"]["sub_questions"]:
            console.print(f"    [dim]{sq['id']}.[/] {sq['question']}")


def save_report(report: Report, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", report.question.lower()).strip("-")[:60] or "report"
    md_path = out_dir / f"{slug}.md"
    md_path.write_text(report.markdown, encoding="utf-8")
    (out_dir / f"{slug}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return md_path


def _usage_table(report: Report) -> Table:
    u = report.usage
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("Time", f"{report.duration_seconds}s")
    t.add_row("LLM calls", f"{u.llm_calls}  ({u.prompt_tokens:,} in / {u.completion_tokens:,} out tokens)")
    t.add_row("Searches", str(u.searches))
    t.add_row("Pages read", f"{u.pages_fetched} ok, {u.pages_failed} fell back to snippets")
    t.add_row("Sources cited", str(len(report.sources)))
    t.add_row("Models", ", ".join(f"{m} x{n}" for m, n in u.calls_by_model.items()) or "-")
    return t


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-agent", description="Multi-agent deep research with citations.")
    parser.add_argument("question", nargs="+", help="The research question.")
    parser.add_argument("--mock", action="store_true", help="Run offline with fake LLM + search (no keys).")
    parser.add_argument("--out", default="reports", help="Directory to save the report (default: reports/).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    question = " ".join(args.question)
    settings = get_settings()

    components = None
    if not args.mock and settings.embedding_backend.lower() == "onnx":
        from .rag.embeddings import OnnxEmbedder

        with console.status("Loading local embedding model (one-time download ~80 MB on first use)…"):
            try:
                OnnxEmbedder._load()
            except Exception as exc:  # noqa: BLE001 - research still works with BM25 only
                console.print(f"[yellow]Embedding model unavailable ({exc}); using keyword search only.[/]")
    if args.mock:
        from .mock import mock_components

        components = mock_components()

    console.print(Panel.fit(f"[bold]{question}[/]", title="Deep Research Agent", border_style="cyan"))
    try:
        report = asyncio.run(research(question, emit=_print_event, settings=settings, components=components))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Research failed:[/] {exc}")
        return 1

    console.rule("Report")
    console.print(Markdown(report.markdown))
    console.rule()
    console.print(_usage_table(report))
    path = save_report(report, Path(args.out))
    console.print(f"\n[green]Saved:[/] {path}  (+ .json with plan, findings and evidence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
