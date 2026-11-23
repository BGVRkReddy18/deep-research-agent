"""Planner agent: question -> research plan of focused sub-questions."""

from __future__ import annotations

import logging
import re
from datetime import date

from pydantic import BaseModel

from ..llm import AllProvidersFailed, LLMRouter, Message
from ..models import ResearchPlan, SubQuestion
from .prompts import PLANNER_SYSTEM

log = logging.getLogger(__name__)


class _PlanOut(BaseModel):
    sub_questions: list[SubQuestion]


def fallback_plan(question: str) -> ResearchPlan:
    """If planning fails we still research the question directly, as a single sub-question."""
    return ResearchPlan(
        main_question=question,
        sub_questions=[SubQuestion(id=1, question=question, search_queries=[question], rationale="direct")],
    )


_JUNK = re.compile(r'["\[\]{}]+')


def clean_queries(raw: list[str], fallback: str) -> list[str]:
    """Repair queries from sloppy model JSON, e.g. ['a","b"]', 'rationale', ':'] -> ['a', 'b'].

    Splits glued-together items, strips stray JSON punctuation, drops fragments that are not
    real queries (too short, or leaked field names), dedupes, and caps at 3.
    """
    out: list[str] = []
    for item in raw:
        for part in re.split(r'"\s*,\s*"', str(item)):
            q = " ".join(_JUNK.sub(" ", part).split()).strip(" ,:;")
            words = re.findall(r"[A-Za-z0-9]+", q)
            if len(words) < 2 or q.lower() in _FIELD_NAMES or len(q) > 200:
                continue
            if q.lower() not in (o.lower() for o in out):
                out.append(q)
    return out[:3] or [fallback[:200]]


_FIELD_NAMES = {"rationale", "question", "search_queries", "id"}


def _clean(question: str, subs: list[SubQuestion], max_sub: int) -> ResearchPlan:
    seen: set[str] = set()
    cleaned: list[SubQuestion] = []
    for sq in subs:
        key = sq.question.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        queries = clean_queries(sq.search_queries, sq.question)
        cleaned.append(sq.model_copy(update={"id": len(cleaned) + 1, "search_queries": queries}))
        if len(cleaned) >= max_sub:
            break
    return ResearchPlan(main_question=question, sub_questions=cleaned) if cleaned else fallback_plan(question)


class Planner:
    def __init__(self, llm: LLMRouter, max_sub_questions: int = 4) -> None:
        self.llm = llm
        self.max_sub = max_sub_questions

    async def plan(self, question: str) -> tuple[ResearchPlan, str | None]:
        """Returns (plan, warning). Never raises for LLM problems: degrades to a 1-step plan."""
        system = PLANNER_SYSTEM.format(max_sub=self.max_sub, today=date.today().isoformat())
        try:
            out = await self.llm.complete_json(
                [Message("system", system), Message("user", question)], _PlanOut, prefer_primary=30.0
            )
        except AllProvidersFailed:
            raise  # no model is reachable: stop now instead of burning search quota on a doomed run
        except Exception as exc:  # noqa: BLE001
            log.warning("planner failed, using fallback plan: %s", exc)
            return fallback_plan(question), f"Planner failed ({type(exc).__name__}); researched the question directly."
        return _clean(question, out.sub_questions, self.max_sub), None
