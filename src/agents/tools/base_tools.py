"""Base tool functions available to research agents."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from src.agents.base_agent import AgentTool
from src.agents.tools.embedding_tool import get_embedding_tool
from src.agents.tools.impact_assessment_tool import get_impact_assessment_tool
from src.agents.tools.formatting_tool import get_formatting_tool
from src.agents.tools.paper_selection_tool import (
    get_paper_relevance_tool,
    get_paper_selection_tool,
)
from src.agents.tools.problem_extraction_tool import get_problem_extraction_tool
from src.agents.tools.quality_check_tool import get_quality_check_tool
from src.agents.tools.results_extraction_tool import get_results_extraction_tool
from src.agents.tools.synthesis_tools import get_synthesis_tools
from src.agents.tools.topic_discovery_tool import (
    get_topic_discovery_tool,
    get_topic_title_tool,
)
from src.fetchers.arxiv_fetcher import ArxivFetcher, Paper

logger = logging.getLogger(__name__)


def _parse_datetime(value: Any, parameter_name: str) -> datetime:
    """Parse datetime-like tool parameters."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{parameter_name} must be an ISO datetime or date string"
            ) from exc
    raise TypeError(f"{parameter_name} must be a datetime or ISO string")


def _paper_to_dict(paper: Any) -> Dict[str, Any]:
    """Convert known paper objects or dictionaries to serializable dictionaries."""
    if isinstance(paper, Paper):
        return paper.to_dict()
    if isinstance(paper, dict):
        return dict(paper)
    if hasattr(paper, "to_dict") and callable(paper.to_dict):
        return paper.to_dict()
    raise TypeError(f"Unsupported paper type: {type(paper)!r}")


def fetch_papers_tool(
    categories: List[str],
    start_date: Any,
    end_date: Optional[Any] = None,
    max_results: int = 1000,
    min_count: Optional[int] = None,
    fetcher: Optional[ArxivFetcher] = None,
) -> Dict[str, Any]:
    """
    Fetch papers from ArXiv for categories and date range.

    When min_count is provided, this delegates to ArxivFetcher.fetch_with_threshold;
    otherwise it fetches the requested categories once.
    """
    if not categories:
        raise ValueError("categories cannot be empty")

    parsed_start = _parse_datetime(start_date, "start_date")
    parsed_end = _parse_datetime(end_date, "end_date") if end_date else datetime.now()
    fetcher = fetcher or ArxivFetcher()

    if min_count is not None:
        papers, actual_start, actual_end = fetcher.fetch_with_threshold(
            categories=categories,
            start_date=parsed_start,
            end_date=parsed_end,
            min_count=min_count,
            max_results_per_category=max_results,
        )
    else:
        papers = fetcher.fetch_multiple_categories(
            categories=categories,
            start_date=parsed_start,
            end_date=parsed_end,
            max_results_per_category=max_results,
        )
        actual_start = parsed_start
        actual_end = parsed_end

    paper_dicts = [_paper_to_dict(paper) for paper in papers]
    logger.info("Fetched %s papers for categories %s", len(paper_dicts), categories)

    return {
        "papers": paper_dicts,
        "paper_count": len(paper_dicts),
        "categories": categories,
        "start_date": actual_start.isoformat(),
        "end_date": actual_end.isoformat(),
        "threshold_met": min_count is None or len(paper_dicts) >= min_count,
        "min_count": min_count,
    }


def check_threshold_tool(paper_count: int, min_threshold: int = 100) -> Dict[str, Any]:
    """Check whether a paper count meets the minimum threshold."""
    if paper_count < 0:
        raise ValueError("paper_count cannot be negative")
    if min_threshold < 1:
        raise ValueError("min_threshold must be at least 1")

    missing_count = max(min_threshold - paper_count, 0)
    return {
        "paper_count": paper_count,
        "min_threshold": min_threshold,
        "threshold_met": missing_count == 0,
        "missing_count": missing_count,
    }


def analyze_paper_tool(paper_text: str, max_chars: int = 12000) -> Dict[str, Any]:
    """
    Extract likely problem and result statements from paper text.

    This deterministic implementation is a phase-3 placeholder. Later LLM-based
    tools can replace it while preserving the same output contract.
    """
    if not paper_text or not paper_text.strip():
        raise ValueError("paper_text cannot be empty")

    text = " ".join(paper_text[:max_chars].split())
    sentences = _split_sentences(text)

    problem_keywords = (
        "problem",
        "question",
        "challenge",
        "address",
        "study",
        "investigate",
        "aim",
    )
    result_keywords = (
        "we prove",
        "we show",
        "we establish",
        "we introduce",
        "we present",
        "main result",
        "theorem",
        "result",
    )

    problem_sentences = _select_sentences(sentences, problem_keywords, limit=3)
    result_sentences = _select_sentences(sentences, result_keywords, limit=5)

    if not problem_sentences:
        problem_sentences = sentences[:2]
    if not result_sentences:
        result_sentences = sentences[2:5] or sentences[:2]

    return {
        "problem": " ".join(problem_sentences).strip(),
        "main_results": result_sentences,
        "confidence": "heuristic",
        "text_chars_analyzed": min(len(paper_text), max_chars),
    }


def generate_summary_tool(
    papers: Iterable[Any],
    topic: str,
    max_papers: int = 5,
) -> Dict[str, Any]:
    """Create a compact deterministic topic summary from paper metadata."""
    paper_dicts = [_paper_to_dict(paper) for paper in papers]
    selected = paper_dicts[:max_papers]

    if not topic:
        raise ValueError("topic cannot be empty")
    if not selected:
        raise ValueError("papers cannot be empty")

    titles = [paper.get("title", "Untitled paper") for paper in selected]
    summaries = [paper.get("summary", "") for paper in selected if paper.get("summary")]
    combined_summary = " ".join(summaries)
    lead = _first_words(combined_summary, 70)

    return {
        "topic": topic,
        "paper_count": len(paper_dicts),
        "representative_papers": titles,
        "summary": (
            f"{topic}: {lead}"
            if lead
            else f"{topic}: representative work includes {', '.join(titles)}."
        ),
    }


def get_base_tools() -> List[AgentTool]:
    """Return the default tool set available to phase-3 agents."""
    return [
        AgentTool(
            name="fetch_papers_tool",
            description="Fetch papers from ArXiv by categories and date range.",
            function=fetch_papers_tool,
            required_parameters=["categories", "start_date"],
        ),
        AgentTool(
            name="check_threshold_tool",
            description="Check whether a fetched paper count meets a minimum threshold.",
            function=check_threshold_tool,
            required_parameters=["paper_count", "min_threshold"],
        ),
        AgentTool(
            name="analyze_paper_tool",
            description="Extract problem and main result statements from paper text.",
            function=analyze_paper_tool,
            required_parameters=["paper_text"],
        ),
        AgentTool(
            name="generate_summary_tool",
            description="Generate a concise topic summary from representative papers.",
            function=generate_summary_tool,
            required_parameters=["papers", "topic"],
        ),
        get_embedding_tool(),
        get_topic_discovery_tool(),
        get_topic_title_tool(),
        get_paper_selection_tool(),
        get_paper_relevance_tool(),
        get_problem_extraction_tool(),
        get_results_extraction_tool(),
        get_impact_assessment_tool(),
        *get_synthesis_tools(),
        get_formatting_tool(),
        get_quality_check_tool(),
    ]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentence-like units."""
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _select_sentences(
    sentences: List[str],
    keywords: Iterable[str],
    limit: int,
) -> List[str]:
    """Select sentences containing any keyword."""
    selected: List[str] = []
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in lowered_keywords):
            selected.append(sentence)
            if len(selected) >= limit:
                break
    return selected


def _first_words(text: str, limit: int) -> str:
    """Return the first words of text with a trailing period when possible."""
    words = text.split()
    if not words:
        return ""
    selected = words[:limit]
    suffix = "..." if len(words) > limit else ""
    return " ".join(selected) + suffix
