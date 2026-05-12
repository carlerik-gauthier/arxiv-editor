"""Content synthesizer for Julius's first multi-agent draft.

The synthesizer is deterministic by default. It converts selected papers,
specialist callbacks, and analysis records into a structured draft controlled
by SummaryRequest preferences. An LLM client can later replace individual text
generation steps without changing the output contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.agents.tools.synthesis_tools import (
    create_paper_summary_tool,
    create_topic_overview_tool,
    rank_summary_items_tool,
    review_and_refine_tool,
)
from src.generation.user_request import Audience, SummaryFormat, SummaryRequest


class ContentSynthesizer:
    """Build structured draft sections from agent outputs and user preferences."""

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        self.llm_client = llm_client

    def synthesize_draft(
        self,
        summary_request: Any,
        selected_papers: Optional[Iterable[Dict[str, Any]]] = None,
        analyses: Optional[Iterable[Dict[str, Any]]] = None,
        agent_results: Optional[Iterable[Dict[str, Any]]] = None,
        previous_feedback: Optional[Iterable[Dict[str, Any]]] = None,
        draft_version: int = 1,
    ) -> Dict[str, Any]:
        """Create a first draft with provenance for topics, papers, and agents."""
        request = _coerce_request(summary_request)
        papers = list(selected_papers or [])
        analysis_list = list(analyses or [])
        callbacks = list(agent_results or [])
        topic = request.topic_query or _topic_from_categories(request) or "Recent ArXiv research"

        ranked_papers = rank_summary_items_tool(
            [
                {**paper, "score": paper.get("score", 1.0 / (index + 1))}
                for index, paper in enumerate(papers[: request.max_papers])
            ],
            ranking_goal="relevance",
        )["items"]
        paper_summaries = [
            create_paper_summary_tool(
                paper=paper,
                analysis=analysis_list[index] if index < len(analysis_list) else None,
                summary_request=request,
            )
            for index, paper in enumerate(ranked_papers)
        ]
        overview = create_topic_overview_tool(
            topic=topic,
            papers=ranked_papers,
            analyses=analysis_list,
            summary_request=request,
        )
        sections = self._build_sections(request, overview, paper_summaries, callbacks)
        content = self._render_content(request, topic, sections, draft_version)
        review = review_and_refine_tool(
            content,
            criteria=["audience", "citation"] if papers else ["audience"],
        )

        return {
            "version": draft_version,
            "title": self._title_for(request, topic),
            "content": review["refined_content"],
            "sections": sections,
            "summary_request": request.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "selected_topics": [topic],
                "selected_papers": [
                    {
                        "title": summary["title"],
                        "arxiv_id": summary.get("arxiv_id"),
                        "rank": index + 1,
                        "inclusion_reason": "Representative for the requested topic and ranking goal.",
                    }
                    for index, summary in enumerate(paper_summaries)
                ],
                "agent_callbacks": [
                    {
                        "agent": callback.get("to_agent") or callback.get("agent"),
                        "status": callback.get("status"),
                    }
                    for callback in callbacks
                ],
                "inclusion_reason": (
                    "The draft follows the user's current topic, date range, audience, "
                    "depth, format, category filters, and selected specialist callbacks."
                ),
                "confidence_notes": overview["confidence_notes"],
                "review_warnings": review["warnings"],
                "previous_feedback": list(previous_feedback or []),
            },
            "status": "drafted",
        }

    def _build_sections(
        self,
        request: SummaryRequest,
        overview: Dict[str, Any],
        paper_summaries: List[Dict[str, Any]],
        callbacks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create reusable draft sections before rendering."""
        sections = [
            {
                "title": "Topic Overview",
                "content": overview["overview"],
                "confidence_notes": overview["confidence_notes"],
            }
        ]
        if paper_summaries:
            sections.append(
                {
                    "title": "Representative Papers",
                    "content": paper_summaries[: request.max_papers],
                }
            )
        if callbacks:
            sections.append(
                {
                    "title": "Specialist Notes",
                    "content": [
                        {
                            "agent": callback.get("to_agent") or callback.get("agent"),
                            "status": callback.get("status"),
                        }
                        for callback in callbacks
                    ],
                }
            )
        if request.audience in {Audience.NON_EXPERT, Audience.MIXED}:
            sections.append(
                {
                    "title": "Accessible Explanation",
                    "content": "Michel review requested for intuition and terminology.",
                }
            )
        return sections

    def _render_content(
        self,
        request: SummaryRequest,
        topic: str,
        sections: List[Dict[str, Any]],
        draft_version: int,
    ) -> str:
        """Render sections into the requested markdown-style format."""
        heading = f"# Draft v{draft_version}: {self._title_for(request, topic)}"
        if request.format == SummaryFormat.BULLET_DIGEST:
            lines = [heading]
            for section in sections:
                lines.append(f"- {section['title']}: {_stringify_section(section['content'])}")
            return "\n".join(lines)
        if request.format == SummaryFormat.PAPER_RANKINGS:
            papers = next(
                (section["content"] for section in sections if section["title"] == "Representative Papers"),
                [],
            )
            lines = [heading]
            for index, paper in enumerate(papers, start=1):
                lines.append(f"{index}. {paper['title']}: {paper['main_result']}")
            return "\n".join(lines)

        lines = [heading]
        for section in sections:
            lines.append(f"## {section['title']}")
            lines.append(_stringify_section(section["content"]))
        return "\n\n".join(lines)

    def _title_for(self, request: SummaryRequest, topic: str) -> str:
        """Return a concise title aligned with the requested format."""
        if request.format == SummaryFormat.PAPER_RANKINGS:
            return f"{topic}: Representative Paper Rankings"
        if request.format == SummaryFormat.BULLET_DIGEST:
            return f"{topic}: Bullet Digest"
        return f"{topic}: Research Brief"


def _coerce_request(summary_request: Any) -> SummaryRequest:
    """Normalize request input for the synthesizer."""
    if isinstance(summary_request, SummaryRequest):
        return summary_request
    if isinstance(summary_request, dict) and "summary_request" in summary_request:
        return SummaryRequest.model_validate(summary_request["summary_request"])
    return SummaryRequest.model_validate(summary_request)


def _topic_from_categories(request: SummaryRequest) -> Optional[str]:
    """Use included categories as a fallback topic label."""
    if request.must_include_categories:
        return ", ".join(request.must_include_categories)
    return None


def _stringify_section(content: Any) -> str:
    """Render structured section content compactly."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered: List[str] = []
        for item in content:
            if isinstance(item, dict):
                title = item.get("title") or item.get("agent") or "Item"
                detail = item.get("main_result") or item.get("status") or item.get("significance") or ""
                rendered.append(f"{title}: {detail}".strip())
            else:
                rendered.append(str(item))
        return "\n".join(f"- {line}" for line in rendered)
    return str(content)
