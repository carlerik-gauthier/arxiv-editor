"""Phase 1 ChrisAgent implementation with OpenAI Agents SDK."""

from __future__ import annotations

import os
import re
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agents import Agent, Runner, function_tool, trace
from openai import OpenAI

from arxiv_fetcher import ArxivFetcher
from data_object import Paper

CHRIS_SYSTEM_PROMPT = (
    "Probability theory expert, focuses on stochastic processes. "
    "You identify key concepts and see application in other fields, such as physics"
)
CHRIS_CATEGORIES = ("math.PR", "stat.TH")
DEFAULT_FETCH_THRESHOLD = 3
DEFAULT_MAX_RESULTS = 1000
DEFAULT_LOOKBACK_DAYS = 30


@function_tool(name_override="arxiv_fetcher_tool")
def arxiv_fetcher_tool(
    start_date: str,
    categories: List[str],
    end_date: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    min_threshold: int = DEFAULT_FETCH_THRESHOLD,
) -> Dict[str, Any]:
    """
    Fetch papers in ChrisAgent's allowed categories (math.PR, stat.TH).

    Use this tool when there is no Probability Theory paper in the requested date range.
    If fetched paper count is below `min_threshold`, return no papers and explain why.
    """
    fetcher = ArxivFetcher()
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date) if end_date else datetime.now()
    categories_set = set(categories).intersection(CHRIS_CATEGORIES)
    if not categories_set:
        categories_set = set(CHRIS_CATEGORIES)
    papers: List[Paper] = []
    for category in categories_set:
        papers.extend(
            fetcher.fetch_by_category(
            category=category,
            start_date=start,
            end_date=end,
            max_results=max_results,
        )
        )
    
    deduped: Dict[str, Paper] = {paper.arxiv_id: paper for paper in papers}
    final_papers = list(deduped.values())

    if len(final_papers) < min_threshold:
        return {
            "papers": [],
            "reason": (
                "Not enough papers fetched to proceed: "
                f"{len(final_papers)} < threshold {min_threshold}."
            ),
            "categories": list(categories_set),
        }

    return {
        "papers": [_paper_to_dict(paper) for paper in final_papers],
        "reason": (
            f"Successfully fetched {len(final_papers)} papers "
            "in the requested range."
        ),
        "categories": list(categories_set),
    }



def build_chris_agent() -> Agent:
    """Create ChrisAgent for Phase 1."""
    return Agent(
        name="ChrisAgent",
        instructions=(
            f"{CHRIS_SYSTEM_PROMPT}\n"
            f"Allowed categories only: {', '.join(CHRIS_CATEGORIES)}.\n"
            "When no Probability Theory nor Statistical Theory paper exists in the requested date range, call "
            "`arxiv_fetcher_tool`."
        ),
        tools=[arxiv_fetcher_tool],
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )


def run_chris_agent(message: str) -> str:
    """Run one ChrisAgent turn with SDK tracing enabled."""
    start_date, end_date = _extract_date_range(message)
    categories = _get_arxiv_categories(message)
    enriched_message = (
        f"{message}\n\n"
        f"Requested date range to use if needed: start_date={start_date}, end_date={end_date}.\n"
        f"Use categories={categories} when calling arxiv_fetcher_tool."
    )
    agent = build_chris_agent()
    with trace(
        "phase1-chris-agent-run",
        metadata={
            "agent": "ChrisAgent",
            "start_date": start_date,
            "end_date": end_date,
            "categories": ",".join(categories),
        },
        disabled=os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "0") == "1",
    ):
        result = Runner.run_sync(agent, enriched_message, max_turns=6)
    return str(getattr(result, "final_output", ""))


def _extract_date_range(message: str) -> tuple[str, str]:
    matches = re.findall(r"\d{4}-\d{2}-\d{2}", message)
    if len(matches) >= 2:
        try:
            start = datetime.fromisoformat(matches[0])
            end = datetime.fromisoformat(matches[1])
        except ValueError:
            matches = []
        else:
            return start.date().isoformat(), end.date().isoformat()
    if len(matches) == 1:
        try:
            start = datetime.fromisoformat(matches[0])
        except ValueError:
            matches = []
        else:
            end = datetime.now()
            return start.date().isoformat(), end.date().isoformat()
    end = datetime.now()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start.date().isoformat(), end.date().isoformat()


def _classify_categories_with_llm(message: str) -> List[str]:
    """Use an LLM to classify the requested ChrisAgent categories."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    prompt = (
        "Classify the user's message into arXiv categories from this allowed set only: "
        "['math.PR', 'stat.TH'].\n"
        "Return JSON only with this schema: {\"categories\": [\"math.PR\", \"stat.TH\"]}.\n"
        "Rules:\n"
        "- Include a category only if clearly requested or strongly implied.\n"
        "- If neither is requested, return both categories.\n"
        "- Never return any other category.\n"
        f"User message: {message}"
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    content = (response.output_text or "").strip()
    payload = json.loads(content)
    raw_categories = payload.get("categories", [])
    if not isinstance(raw_categories, list):
        return []
    return [str(category) for category in raw_categories]


def _get_arxiv_categories(message: str) -> List[str]:
    """
    Infer ChrisAgent categories from user message.

    Returns only categories allowed to ChrisAgent and defaults to all allowed
    categories when no explicit/implicit signal is found.
    """
    try:
        raw_categories = _classify_categories_with_llm(message)
    except Exception:
        raw_categories = []

    normalized = {category for category in raw_categories if category in CHRIS_CATEGORIES}
    if not normalized:
        return list(CHRIS_CATEGORIES)
    return list(normalized)


def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    data = asdict(paper)
    data["published"] = paper.published.isoformat()
    data["updated"] = paper.updated.isoformat()
    return data


def _parse_iso_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc
    