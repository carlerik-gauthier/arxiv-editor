"""Agent-callable quality validation tool."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from src.agents.base_agent import AgentTool
from src.generation.validator import DocumentValidator


def validate_quality_tool(
    document: Any,
    summary_request: Any,
    source_papers: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate a formatted document against the request and source papers."""
    validator = DocumentValidator()
    report = validator.validate(document, summary_request, source_papers)
    report["suggestions"] = validator.generate_improvement_suggestions(report)
    return report


def get_quality_check_tool() -> AgentTool:
    """Return the quality check tool schema."""
    return AgentTool(
        name="validate_quality_tool",
        description="Validate formatted draft completeness, citations, and audience fit.",
        function=validate_quality_tool,
        required_parameters=["document", "summary_request"],
    )
