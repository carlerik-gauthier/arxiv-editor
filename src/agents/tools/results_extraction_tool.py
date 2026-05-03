"""Agent-facing tool for extracting key results from research papers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base_agent import AgentTool
from src.analysis.paper_analyzer import PaperAnalyzer


def extract_key_results_tool(
    paper_text: str,
    paper_metadata: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    llm_client: Optional[Any] = None,
    analyzer: Optional[PaperAnalyzer] = None,
    max_chunk_tokens: int = 1800,
) -> Dict[str, Any]:
    """
    Extract the main findings, guarantees, or empirical results from a paper.

    Args:
        paper_text: Full paper text or a substantial excerpt.
        paper_metadata: Optional title/authors/categories/summary metadata.
        domain: Optional domain hint such as `math`, `ml`, `crypto`, or
            `general`.
        llm_client: Optional LLM client used when `analyzer` is not supplied.
        analyzer: Optional injected PaperAnalyzer for tests or alternate logic.
        max_chunk_tokens: Approximate token budget for analyzer chunking.

    Returns:
        A tool-safe dictionary with a ranked `results` list. Each result includes
        result type, statement, significance, location, evidence, and importance
        score. Tool-level errors are returned as structured failure payloads so
        agent workflows can continue after a bad paper extraction.
    """
    active_analyzer = analyzer or PaperAnalyzer(
        llm_client=llm_client,
        max_chunk_tokens=max_chunk_tokens,
    )
    try:
        result = active_analyzer.extract_key_results(
            paper_text=paper_text,
            paper_metadata=paper_metadata or {},
            domain=domain,
        )
        return {
            "status": "completed",
            **result,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "results": [],
            "result_count": 0,
            "domain": domain or "unknown",
            "confidence": "none",
            "source": "tool_error",
            "sections_used": [],
            "chunks_analyzed": 0,
            "error": str(exc),
        }


def get_results_extraction_tool() -> AgentTool:
    """
    Return the key-results extraction tool registered with specialist agents.

    The tool supports domain-aware extraction for mathematical theorems, machine
    learning empirical results, and cryptographic security guarantees while
    preserving one stable output contract for Julius and specialist agents.
    """
    return AgentTool(
        name="extract_key_results_tool",
        description=(
            "Extract and rank a paper's key results. For math papers, prioritize "
            "theorems and formal guarantees; for ML papers, prioritize empirical "
            "findings, benchmarks, architectures, ablations, and theory; for "
            "crypto papers, prioritize security guarantees, attacks, protocols, "
            "and assumptions."
        ),
        function=extract_key_results_tool,
        required_parameters=["paper_text"],
    )
