"""Deterministic synthesis tools for multi-agent draft generation.

These tools are small, documented contracts for step 6.3. They accept agent
analyses, selected papers, and SummaryRequest preferences, then return
structured draft pieces that can later be replaced by LLM-backed versions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.agents.base_agent import AgentTool
from src.generation.user_request import Audience, Depth, SummaryRequest


def create_topic_overview_tool(
    topic: str,
    papers: Iterable[Dict[str, Any]],
    analyses: Optional[Iterable[Dict[str, Any]]] = None,
    summary_request: Optional[Any] = None,
) -> Dict[str, Any]:
    """Synthesize a topic-level overview for the requested audience and depth."""
    request = _coerce_request(summary_request)
    paper_list = list(papers or [])
    analysis_list = list(analyses or [])
    paper_titles = [_paper_title(paper) for paper in paper_list[: request.max_papers]]
    key_points = _collect_key_points(analysis_list, paper_list)
    lead = key_points[0] if key_points else f"Recent work clusters around {topic}."
    if request.audience == Audience.NON_EXPERT:
        lead = generate_layperson_explanation_tool(lead)["explanation"]
    elif request.audience == Audience.EXPERT:
        lead = generate_expert_explanation_tool(lead, domain=topic)["explanation"]

    return {
        "topic": topic,
        "title": topic.title(),
        "overview": lead,
        "paper_titles": paper_titles,
        "paper_count": len(paper_list),
        "depth": request.depth.value,
        "confidence_notes": _confidence_notes(analysis_list),
    }


def create_paper_summary_tool(
    paper: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    summary_request: Optional[Any] = None,
) -> Dict[str, Any]:
    """Summarize one representative paper according to user preferences."""
    request = _coerce_request(summary_request)
    analysis = analysis or {}
    problem = analysis.get("problem") or analysis.get("problem_statement") or paper.get("summary", "")
    results = analysis.get("main_results") or analysis.get("key_results") or []
    if isinstance(results, str):
        results = [results]
    statement = results[0] if results else _first_words(paper.get("summary", ""), 32)
    if request.depth == Depth.BRIEF:
        statement = _first_words(statement, 24)

    return {
        "title": _paper_title(paper),
        "arxiv_id": paper.get("arxiv_id") or paper.get("id"),
        "problem": _first_words(problem, 40),
        "main_result": _first_words(statement, 45),
        "significance": _first_words(
            analysis.get("impact_summary") or analysis.get("significance") or statement,
            36,
        ),
        "audience": request.audience.value,
    }


def generate_expert_explanation_tool(content: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """Return a technical explanation shell for expert readers."""
    domain_text = f" in {domain}" if domain else ""
    return {
        "explanation": f"Technically{domain_text}, {content}",
        "style": "expert",
    }


def generate_layperson_explanation_tool(
    content: str,
    metaphors: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return an accessible explanation shell for non-expert readers."""
    metaphor_text = f" Think of it as {next(iter(metaphors), '')}." if metaphors else ""
    return {
        "explanation": f"In plain terms, {content}{metaphor_text}".strip(),
        "style": "layperson",
    }


def rank_summary_items_tool(
    items: Iterable[Dict[str, Any]],
    ranking_goal: str = "relevance",
) -> Dict[str, Any]:
    """Rank topics or papers by an available score while preserving stable ties."""
    ranked = sorted(
        list(items or []),
        key=lambda item: (
            item.get(f"{ranking_goal}_score")
            or item.get("score")
            or item.get("impact_score")
            or 0
        ),
        reverse=True,
    )
    return {
        "ranking_goal": ranking_goal,
        "items": [
            {**item, "rank": index + 1}
            for index, item in enumerate(ranked)
        ],
    }


def review_and_refine_tool(content: str, criteria: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Run a deterministic quality pass and return concise warnings."""
    criteria_list = list(criteria or [])
    warnings: List[str] = []
    if not content or not content.strip():
        warnings.append("Content is empty.")
    if "citation" in criteria_list and "arxiv" not in content.lower():
        warnings.append("No ArXiv citation marker found.")
    if "audience" in criteria_list and len(content.split()) < 40:
        warnings.append("Content may be too short for audience calibration.")
    return {
        "refined_content": content.strip(),
        "criteria": criteria_list,
        "warnings": warnings,
        "passed": not warnings,
    }


def get_synthesis_tools() -> List[AgentTool]:
    """Return synthesis tools for Julius and specialist agents."""
    return [
        AgentTool(
            name="create_topic_overview_tool",
            description="Synthesize a topic overview from papers, analyses, and SummaryRequest preferences.",
            function=create_topic_overview_tool,
            required_parameters=["topic", "papers"],
        ),
        AgentTool(
            name="create_paper_summary_tool",
            description="Summarize one representative paper for the requested audience and depth.",
            function=create_paper_summary_tool,
            required_parameters=["paper"],
        ),
        AgentTool(
            name="generate_expert_explanation_tool",
            description="Convert content into a concise expert-facing explanation.",
            function=generate_expert_explanation_tool,
            required_parameters=["content"],
        ),
        AgentTool(
            name="generate_layperson_explanation_tool",
            description="Convert content into a concise non-expert explanation.",
            function=generate_layperson_explanation_tool,
            required_parameters=["content"],
        ),
        AgentTool(
            name="rank_summary_items_tool",
            description="Rank draft topics or paper summaries by a stated goal.",
            function=rank_summary_items_tool,
            required_parameters=["items"],
        ),
        AgentTool(
            name="review_and_refine_tool",
            description="Check and lightly refine synthesized content against criteria.",
            function=review_and_refine_tool,
            required_parameters=["content"],
        ),
    ]


def _coerce_request(summary_request: Optional[Any]) -> SummaryRequest:
    """Normalize optional request input for tool calls."""
    if summary_request is None:
        return SummaryRequest()
    if isinstance(summary_request, SummaryRequest):
        return summary_request
    if isinstance(summary_request, dict) and "summary_request" in summary_request:
        return SummaryRequest.model_validate(summary_request["summary_request"])
    return SummaryRequest.model_validate(summary_request)


def _paper_title(paper: Dict[str, Any]) -> str:
    """Return a stable title for paper-like dictionaries."""
    return str(paper.get("title") or paper.get("name") or "Untitled paper")


def _collect_key_points(
    analyses: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
) -> List[str]:
    """Collect result-like sentences from analyses, falling back to abstracts."""
    points: List[str] = []
    for analysis in analyses:
        for key in ("impact_summary", "significance", "problem"):
            if analysis.get(key):
                points.append(str(analysis[key]))
        results = analysis.get("main_results") or analysis.get("key_results") or []
        if isinstance(results, str):
            points.append(results)
        else:
            points.extend(str(result) for result in results[:2])
    if not points:
        points.extend(str(paper.get("summary", "")) for paper in papers if paper.get("summary"))
    return [_first_words(point, 45) for point in points if point]


def _confidence_notes(analyses: List[Dict[str, Any]]) -> List[str]:
    """Extract uncertainty notes from specialist analyses."""
    notes = [
        str(analysis.get("confidence_note") or analysis.get("confidence"))
        for analysis in analyses
        if analysis.get("confidence_note") or analysis.get("confidence")
    ]
    return notes or ["deterministic synthesis; specialist validation pending"]


def _first_words(text: str, limit: int) -> str:
    """Return the first `limit` words of text."""
    words = str(text or "").split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + "..."
