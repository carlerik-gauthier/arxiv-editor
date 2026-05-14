"""Base tool functions available to research agents."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from src.agents.base_agent import AgentTool
from src.analysis.paper_analyzer import PaperAnalyzer
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


def analyze_paper_tool(
    paper_text: str,
    max_chars: int = 12000,
    paper_metadata: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    llm_client: Optional[Any] = None,
    analyzer: Optional[PaperAnalyzer] = None,
    max_chunk_tokens: int = 1800,
) -> Dict[str, Any]:
    """
    Extract likely problem and result statements from paper text.

    Uses PaperAnalyzer, which calls an injected LLM client when available and
    falls back to documented section heuristics for offline workflows.
    """
    if not paper_text or not paper_text.strip():
        raise ValueError("paper_text cannot be empty")

    text = " ".join(paper_text[:max_chars].split())
    active_analyzer = analyzer or PaperAnalyzer(
        llm_client=llm_client,
        max_chunk_tokens=max_chunk_tokens,
    )
    problem_result = active_analyzer.extract_problem_statement(
        text,
        paper_metadata=paper_metadata or {},
    )
    key_results = active_analyzer.extract_key_results(
        text,
        paper_metadata=paper_metadata or {},
        domain=domain,
    )
    main_results = [
        str(result.get("statement", "")).strip()
        for result in key_results.get("results", [])
        if result.get("statement")
    ]

    if not main_results:
        main_results = _heuristic_result_sentences(text)

    return {
        "problem": problem_result.get("problem", ""),
        "main_results": main_results,
        "confidence": _combined_confidence(problem_result, key_results),
        "text_chars_analyzed": min(len(paper_text), max_chars),
        "problem_details": problem_result,
        "result_details": key_results,
    }


def generate_summary_tool(
    papers: Iterable[Any],
    topic: str,
    max_papers: int = 5,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create a compact topic summary from paper metadata, using an LLM when supplied."""
    paper_dicts = [_paper_to_dict(paper) for paper in papers]
    selected = paper_dicts[:max_papers]

    if not topic:
        raise ValueError("topic cannot be empty")
    if not selected:
        raise ValueError("papers cannot be empty")

    titles = [paper.get("title", "Untitled paper") for paper in selected]
    summaries = [paper.get("summary", "") for paper in selected if paper.get("summary")]
    combined_summary = " ".join(summaries)
    if llm_client is not None:
        llm_summary = _generate_summary_with_llm(llm_client, topic, selected)
        if llm_summary:
            return {
                "topic": topic,
                "paper_count": len(paper_dicts),
                "representative_papers": titles,
                "summary": llm_summary,
                "source": "llm",
            }

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
        "source": "heuristic",
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


def _heuristic_result_sentences(text: str) -> List[str]:
    """Fallback result extraction used only when PaperAnalyzer returns no statements."""
    sentences = _split_sentences(text)
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
    result_sentences = _select_sentences(sentences, result_keywords, limit=5)
    return result_sentences or sentences[2:5] or sentences[:2]


def _combined_confidence(
    problem_result: Dict[str, Any],
    key_results: Dict[str, Any],
) -> str:
    """Return a compact confidence/source label for the combined analysis."""
    sources = {
        str(problem_result.get("source") or problem_result.get("confidence") or "").lower(),
        str(key_results.get("source") or key_results.get("confidence") or "").lower(),
    }
    if "llm" in sources:
        return "llm"
    if any(source == "heuristic_fallback" for source in sources):
        return "heuristic_fallback"
    return "heuristic"


def _generate_summary_with_llm(
    llm_client: Any,
    topic: str,
    papers: List[Dict[str, Any]],
) -> str:
    """Call an injected LLM client and normalize a topic-summary response."""
    prompt = _summary_prompt(topic, papers)
    try:
        response = _call_summary_llm(llm_client, prompt)
    except Exception:
        logger.exception("LLM summary generation failed; falling back to heuristic summary.")
        return ""

    if isinstance(response, dict):
        response = response.get("summary") or response.get("content") or response.get("text")
    elif not isinstance(response, str):
        response = _extract_llm_text(response)

    summary = str(response or "").strip().strip('"')
    return " ".join(summary.split())


def _summary_prompt(topic: str, papers: List[Dict[str, Any]]) -> str:
    """Build a concise prompt for LLM-backed topic summaries."""
    paper_lines = []
    for index, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled paper")
        summary = _first_words(str(paper.get("summary") or paper.get("abstract") or ""), 55)
        paper_lines.append(f"{index}. {title}: {summary}")
    return (
        "Write a concise research-topic summary for Julius.\n"
        f"Topic: {topic}\n"
        "Representative papers:\n"
        + "\n".join(paper_lines)
        + "\nReturn 2-4 sentences only."
    )


def _call_summary_llm(llm_client: Any, prompt: str) -> Any:
    """Call common LLM client shapes used elsewhere in the repo."""
    if callable(llm_client):
        return llm_client(prompt)

    responses_api = getattr(llm_client, "responses", None)
    create_method = getattr(responses_api, "create", None)
    if callable(create_method):
        return create_method(model=getattr(llm_client, "model", "gpt-4o-mini"), input=prompt)

    chat_api = getattr(llm_client, "chat", None)
    completions_api = getattr(chat_api, "completions", None)
    create_method = getattr(completions_api, "create", None)
    if callable(create_method):
        return create_method(
            model=getattr(llm_client, "model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )

    for method_name in ("complete", "generate", "chat", "invoke"):
        method = getattr(llm_client, method_name, None)
        if callable(method):
            return method(prompt)

    raise TypeError("llm_client must be callable or expose a common generation method")


def _extract_llm_text(response: Any) -> str:
    """Extract text from common OpenAI-compatible response shapes."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            return content
    try:
        return json.dumps(response, default=str)
    except TypeError:
        return str(response)


def _first_words(text: str, limit: int) -> str:
    """Return the first words of text with a trailing period when possible."""
    words = text.split()
    if not words:
        return ""
    selected = words[:limit]
    suffix = "..." if len(words) > limit else ""
    return " ".join(selected) + suffix
