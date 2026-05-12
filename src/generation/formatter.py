"""Document formatting for approved Julius drafts.

The formatter renders deterministic Markdown first, plus simple HTML and PDF
outputs. It keeps templates small so later richer templates can replace them
without changing the public methods.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any, Dict

from src.generation.user_request import SummaryFormat, SummaryRequest


class DocumentFormatter:
    """Render draft content into Markdown, HTML, or a minimal PDF payload."""

    def apply_template(self, content: Dict[str, Any], template_name: str = "one_pager") -> str:
        """Apply a SummaryRequest-aware Markdown template to draft content."""
        draft = dict(content or {})
        request = _coerce_request(draft.get("summary_request"))
        title = draft.get("title") or _title_for(request)
        body = draft.get("content", "")
        provenance = draft.get("provenance", {})
        selected_topics = ", ".join(provenance.get("selected_topics", [])) or "not recorded"
        papers = provenance.get("selected_papers", [])
        agents = provenance.get("agent_callbacks", [])
        generated_at = datetime.now(timezone.utc).isoformat()

        lines = [
            f"# {title}",
            f"Generated at: {generated_at}",
            f"Date range: {request.date_range.start_date} to {request.date_range.end_date}",
            f"Format: {request.format.value}",
            f"Selected topics: {selected_topics}",
            "",
            body,
            "",
            "## Representative Papers",
            *_paper_lines(papers),
            "",
            "## Agent Credits",
            *_agent_lines(agents),
        ]
        if template_name == SummaryFormat.BULLET_DIGEST.value:
            return "\n".join(_to_bullets(lines))
        if template_name == SummaryFormat.PAPER_RANKINGS.value:
            return "\n".join(lines)
        return "\n".join(lines)

    def render_to_format(self, template: str, output_format: str = "markdown") -> Any:
        """Render a templated document as markdown, html, or pdf bytes."""
        normalized = output_format.lower()
        if normalized in {"markdown", "md"}:
            return template
        if normalized == "html":
            escaped = html.escape(template)
            return f"<html><body><pre>{escaped}</pre></body></html>"
        if normalized == "pdf":
            payload = template.replace("(", "[").replace(")", "]")
            return (
                b"%PDF-1.4\n"
                b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
                b"2 0 obj << /Type /Pages /Count 0 >> endobj\n"
                + f"% Julius draft\n{payload}\n%%EOF\n".encode("utf-8")
            )
        raise ValueError("output_format must be markdown, html, or pdf")


def _coerce_request(value: Any) -> SummaryRequest:
    """Normalize request payloads embedded in drafts."""
    if isinstance(value, SummaryRequest):
        return value
    if isinstance(value, dict):
        return SummaryRequest.model_validate(value)
    return SummaryRequest()


def _title_for(request: SummaryRequest) -> str:
    """Create a fallback title from the request format and topic."""
    topic = request.topic_query or "ArXiv Research Brief"
    if request.format == SummaryFormat.PAPER_RANKINGS:
        return f"{topic}: Paper Rankings"
    if request.format == SummaryFormat.BULLET_DIGEST:
        return f"{topic}: Bullet Digest"
    return topic


def _paper_lines(papers: Any) -> list[str]:
    """Render representative paper metadata."""
    if not papers:
        return ["- None recorded"]
    return [
        f"- {paper.get('title', 'Untitled paper')} ({paper.get('arxiv_id', 'no arxiv id')})"
        for paper in papers
    ]


def _agent_lines(agents: Any) -> list[str]:
    """Render agent contribution metadata."""
    if not agents:
        return ["- Julius"]
    return [
        f"- {agent.get('agent', 'Unknown agent')}: {agent.get('status', 'unknown')}"
        for agent in agents
    ]


def _to_bullets(lines: list[str]) -> list[str]:
    """Convert non-heading content to compact bullet lines."""
    converted: list[str] = []
    for line in lines:
        if not line:
            continue
        converted.append(line if line.startswith("#") or line.startswith("-") else f"- {line}")
    return converted
