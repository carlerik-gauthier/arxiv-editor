"""Agent-facing tool for extracting research problem statements from papers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base_agent import AgentTool
from src.analysis.paper_analyzer import PaperAnalyzer


def extract_problem_statement_tool(
    paper_text: str,
    paper_metadata: Optional[Dict[str, Any]] = None,
    llm_client: Optional[Any] = None,
    analyzer: Optional[PaperAnalyzer] = None,
    max_chunk_tokens: int = 1800,
) -> Dict[str, Any]:
    """
    Extract a structured research problem statement from a paper.

    Args:
        paper_text: Full paper text or a substantial excerpt.
        paper_metadata: Optional title/authors/categories/summary metadata.
        llm_client: Optional LLM client used when `analyzer` is not supplied.
        analyzer: Optional injected PaperAnalyzer for tests or alternate logic.
        max_chunk_tokens: Approximate token budget for analyzer chunking.

    Returns:
        A tool-safe dictionary with problem, motivation, research gap, context,
        evidence, confidence/source metadata, and section/chunk diagnostics.
        Tool-level errors are returned as a structured failure payload so agent
        workflows can continue after a bad paper extraction.
    """
    active_analyzer = analyzer or PaperAnalyzer(
        llm_client=llm_client,
        max_chunk_tokens=max_chunk_tokens,
    )
    try:
        result = active_analyzer.extract_problem_statement(
            paper_text=paper_text,
            paper_metadata=paper_metadata or {},
        )
        return {
            "status": "completed",
            **result,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "problem": "",
            "motivation": "",
            "research_gap": "",
            "context": "",
            "evidence": [],
            "confidence": "none",
            "source": "tool_error",
            "sections_used": [],
            "chunks_analyzed": 0,
            "error": str(exc),
        }


def get_problem_extraction_tool() -> AgentTool:
    """
    Return the problem extraction tool registered with specialist agents.

    The tool helps agents identify a paper's motivation, research gap, and
    specific problem before later tools extract key results and assess impact.
    """
    return AgentTool(
        name="extract_problem_statement_tool",
        description=(
            "Extract a paper's research problem, motivation, research gap, and "
            "context from full text or a substantial excerpt. Uses an injected "
            "LLM client when available and falls back to documented deterministic "
            "section heuristics."
        ),
        function=extract_problem_statement_tool,
        required_parameters=["paper_text"],
    )
