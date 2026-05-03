"""Agent-facing tool for assessing paper impact and significance."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from src.agents.base_agent import AgentTool
from src.analysis.paper_analyzer import PaperAnalyzer


def assess_impact_tool(
    paper: Any,
    results: Optional[Iterable[Dict[str, Any]]] = None,
    field_context: Optional[str] = None,
    domain: Optional[str] = None,
    llm_client: Optional[Any] = None,
    analyzer: Optional[PaperAnalyzer] = None,
    max_chunk_tokens: int = 1800,
) -> Dict[str, Any]:
    """
    Assess a paper's likely research impact for specialist-agent review.

    Args:
        paper: Paper metadata dictionary, paper-like object, or raw paper text.
        results: Optional extracted key results from `extract_key_results_tool`.
        field_context: Optional specialist context about the field, known open
            problems, applications, or community needs.
        domain: Optional domain hint such as `math`, `ml`, `crypto`, or
            `general`.
        llm_client: Optional LLM client used when `analyzer` is not supplied.
        analyzer: Optional injected PaperAnalyzer for tests or alternate logic.
        max_chunk_tokens: Approximate token budget for analyzer setup.

    Returns:
        A tool-safe dictionary with novelty, open-problem, technique,
        application, community-impact, narrative, evidence, and provenance
        fields. Tool-level errors are returned as structured failure payloads.
    """
    active_analyzer = analyzer or PaperAnalyzer(
        llm_client=llm_client,
        max_chunk_tokens=max_chunk_tokens,
    )
    try:
        result = active_analyzer.assess_impact(
            paper=paper,
            results=results,
            field_context=field_context,
            domain=domain,
        )
        return {
            "status": "completed",
            **result,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "novelty_score": 0.0,
            "solves_open_problem": False,
            "introduces_new_techniques": False,
            "potential_applications": [],
            "community_impact": "unknown",
            "community_impact_score": 0.0,
            "impact_summary": "",
            "evidence": [],
            "confidence": "none",
            "source": "tool_error",
            "domain": domain or "unknown",
            "sections_used": [],
            "result_count": 0,
            "error": str(exc),
        }


def get_impact_assessment_tool() -> AgentTool:
    """
    Return the impact assessment tool registered with specialist agents.

    The tool helps specialists evaluate novelty, open-problem relevance,
    methods, applications, and likely community significance after problem and
    result extraction have produced the supporting facts.
    """
    return AgentTool(
        name="assess_impact_tool",
        description=(
            "Assess a paper's research impact and significance. Evaluates novelty, "
            "whether it addresses an open problem, whether it introduces new "
            "techniques, potential applications, and likely community impact. "
            "Accepts specialist field context to improve the assessment."
        ),
        function=assess_impact_tool,
        required_parameters=["paper"],
    )
