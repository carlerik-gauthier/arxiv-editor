"""Agent-callable document formatting tool."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base_agent import AgentTool
from src.generation.formatter import DocumentFormatter


def format_document_tool(
    content: Dict[str, Any],
    output_format: str = "markdown",
    style: str = "professional",
    template_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Format a draft as Markdown, HTML, or PDF-ready bytes."""
    formatter = DocumentFormatter()
    request_format = content.get("summary_request", {}).get("format", "one_pager")
    template = formatter.apply_template(content, template_name or request_format)
    rendered = formatter.render_to_format(template, output_format)
    return {
        "output_format": output_format,
        "style": style,
        "template_name": template_name or request_format,
        "document": rendered,
        "is_binary": isinstance(rendered, bytes),
    }


def get_formatting_tool() -> AgentTool:
    """Return the formatting tool schema."""
    return AgentTool(
        name="format_document_tool",
        description="Format an approved draft as markdown, html, or pdf.",
        function=format_document_tool,
        required_parameters=["content"],
    )
