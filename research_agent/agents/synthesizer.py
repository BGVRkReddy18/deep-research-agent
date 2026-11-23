"""Synthesizer agent: worker findings -> one coherent, cited markdown report.

Citation pipeline:
  chunk ids (w1-d2-c3)  --assign-->  source numbers per URL [1..n]
  LLM writes report with [n]  --verify-->  drop unknown numbers, renumber by first use
  --append-->  numbered "Sources" list with the exact supporting passages.
"""

from __future__ import annotations

import logging
import re

from ..llm import LLMRouter, Message
from ..models import Finding, Source
from .prompts import SYNTH_SYSTEM, SYNTH_USER

log = logging.getLogger(__name__)

CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\](?!\()")

# Models cite in many dialects; map them all to [n] before verification.
_FULLWIDTH = re.compile(r"【\s*(\d+(?:\s*[,;]\s*\d+)*)[^】]*】")  # 【1】 【1†source】 (gpt-oss style)
_FOOTNOTE = re.compile(r"\[\^(\d+)\]")  # [^1]
_WORDY = re.compile(r"\[(?:source|ref|citation)s?\s*:?\s*(\d+(?:\s*[,;]\s*\d+)*)\]", re.IGNORECASE)  # [Source 1]
_RANGE = re.compile(r"\[(\d+)\s*[-–]\s*(\d+)\]")  # [2-4]


def normalize_citations(text: str) -> str:
    text = _FULLWIDTH.sub(lambda m: f"[{m.group(1)}]", text)
    text = _FOOTNOTE.sub(r"[\1]", text)
    text = _WORDY.sub(lambda m: f"[{m.group(1)}]", text)

    def expand(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        return "".join(f"[{n}]" for n in range(a, b + 1)) if 0 < b - a < 10 else m.group(0)

    return _RANGE.sub(expand, text)


def assign_sources(findings: list[Finding]) -> tuple[dict[str, int], dict[int, Source]]:
    """Give each distinct URL a number. Returns (chunk_id -> number, number -> Source)."""
    url_to_num: dict[str, int] = {}
    sources: dict[int, Source] = {}
    chunk_to_num: dict[str, int] = {}
    for f in findings:
        for chunk in f.evidence:
            if chunk.url not in url_to_num:
                n = len(url_to_num) + 1
                url_to_num[chunk.url] = n
                sources[n] = Source(number=n, url=chunk.url, title=chunk.title)
            n = url_to_num[chunk.url]
            chunk_to_num[chunk.id] = n
            passage = chunk.text[:400] + ("…" if len(chunk.text) > 400 else "")
            if passage not in sources[n].passages:
                sources[n].passages.append(passage)
    return chunk_to_num, sources


def format_findings(findings: list[Finding], chunk_to_num: dict[str, int]) -> str:
    blocks = []
    for f in findings:
        lines = [f"### Sub-question {f.sub_question_id}: {f.sub_question}"]
        if f.summary:
            lines.append(f"Summary: {f.summary}")
        lines.append("Claims:")
        for c in f.claims:
            nums = sorted({chunk_to_num[i] for i in c.evidence_ids if i in chunk_to_num})
            lines.append(f"- {c.statement} " + "".join(f"[{n}]" for n in nums))
        if f.gaps:
            lines.append(f"Gaps: {f.gaps}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def renumber_citations(markdown: str, sources: dict[int, Source]) -> tuple[str, list[Source], int]:
    """Renumber citations by order of first appearance; strip numbers that don't exist.

    Returns (markdown, cited sources in order, count of invalid citations removed).
    """
    mapping: dict[int, int] = {}
    invalid = 0

    def repl(match: re.Match) -> str:
        nonlocal invalid
        out = []
        for raw in re.split(r"\s*[,;]\s*", match.group(1)):
            old = int(raw)
            if old not in sources:
                invalid += 1
                continue
            if old not in mapping:
                mapping[old] = len(mapping) + 1
            out.append(f"[{mapping[old]}]")
        return "".join(dict.fromkeys(out))

    text = CITE.sub(repl, markdown)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)  # tidy spaces left by removed citations
    ordered = [sources[old].model_copy(update={"number": new}) for old, new in sorted(mapping.items(), key=lambda x: x[1])]
    return text, ordered, invalid


def render_sources(sources: list[Source]) -> str:
    if not sources:
        return ""
    lines = ["", "## Sources", ""]
    for s in sources:
        lines.append(f"{s.number}. [{s.title}]({s.url})")
    return "\n".join(lines)


def fallback_report(question: str, findings: list[Finding], chunk_to_num: dict[str, int]) -> str:
    """Deterministic report when the synthesizer LLM is unavailable: the findings, verbatim."""
    parts = [f"# {question}", "", "> The synthesis step was unavailable, so worker findings are shown directly.", ""]
    for f in findings:
        parts.append(f"## {f.sub_question}")
        if f.summary:
            parts.append(f.summary)
        for c in f.claims:
            nums = sorted({chunk_to_num[i] for i in c.evidence_ids if i in chunk_to_num})
            parts.append(f"- {c.statement} " + "".join(f"[{n}]" for n in nums))
        parts.append("")
    return "\n".join(parts)


class Synthesizer:
    def __init__(self, llm: LLMRouter) -> None:
        self.llm = llm

    async def write(self, question: str, findings: list[Finding]) -> tuple[str, list[Source], list[str]]:
        """Returns (final markdown incl. sources list, cited sources, warnings)."""
        warnings: list[str] = []
        usable = [f for f in findings if f.claims]
        chunk_to_num, sources = assign_sources(usable)

        if not usable:
            md = (
                f"# {question}\n\nNo sub-question produced supported findings, so no report could be written. "
                "Try rephrasing the question or running again later."
            )
            return md, [], ["No usable findings."]

        try:
            result = await self.llm.complete(
                [
                    Message("system", SYNTH_SYSTEM),
                    Message("user", SYNTH_USER.format(question=question, findings=format_findings(usable, chunk_to_num))),
                ],
                temperature=0.3,
                max_tokens=2500,
                prefer_primary=30.0,  # the final report deserves the strongest model
            )
            body = result.text.strip()
            body = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", body)
            body = re.split(r"\n#{1,3}\s*(?:Sources|References)\s*\n", body)[0].rstrip()
            if not body.lstrip().startswith("# "):
                body = f"# {question}\n\n{body}"
        except Exception as exc:  # noqa: BLE001
            log.warning("synthesizer failed, using fallback report: %s", exc)
            warnings.append(f"Synthesis failed ({type(exc).__name__}); showing raw findings.")
            body = fallback_report(question, usable, chunk_to_num)

        body, cited, invalid = renumber_citations(normalize_citations(body), sources)
        if invalid:
            warnings.append(f"Removed {invalid} citation(s) that did not match any source.")
        if not cited:
            warnings.append("The report contains no citations.")
        return body + "\n" + render_sources(cited) + "\n", cited, warnings
