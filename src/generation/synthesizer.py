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
from src.generation.user_request import SummaryFormat, SummaryRequest
from src.processing.topic_modeler import MAX_REPRESENTATIVE_PAPERS_PER_TOPIC


MAX_TOPICS_PER_RESPONSE = 7


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
        specialist_topics = _extract_specialist_topics(
            callbacks,
            min(request.max_topics, MAX_TOPICS_PER_RESPONSE),
        )
        if not specialist_topics:
            specialist_topics = _fallback_topic_summaries(
                request=request,
                overview=overview,
                paper_summaries=paper_summaries,
            )
        return _topic_sections(specialist_topics)

    def _render_content(
        self,
        request: SummaryRequest,
        topic: str,
        sections: List[Dict[str, Any]],
        draft_version: int,
    ) -> str:
        """Render sections into the requested markdown-style format."""
        heading = f"# Draft v{draft_version}: {self._title_for(request, topic)}"
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


def _extract_specialist_topics(
    callbacks: List[Dict[str, Any]],
    max_topics: int,
) -> List[Dict[str, Any]]:
    """Collect topic summaries from completed specialist callbacks."""
    topics: List[Dict[str, Any]] = []
    for callback in callbacks:
        if callback.get("status") not in {"COMPLETED", "completed"}:
            continue
        result = callback.get("result") or {}
        response = result.get("response") if isinstance(result, dict) else None
        if not isinstance(response, dict):
            continue
        for topic in response.get("topic_summaries") or []:
            normalized_topic = dict(topic)
            representative_papers = list(normalized_topic.get("representative_papers") or [])
            normalized_topic["representative_papers"] = representative_papers[
                :MAX_REPRESENTATIVE_PAPERS_PER_TOPIC
            ]
            topics.append(normalized_topic)
            if len(topics) >= max_topics:
                return topics
    return topics


def _topic_sections(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Render Julius's final output as repeated topic blocks."""
    sections: List[Dict[str, Any]] = []
    for index, topic in enumerate(topics, start=1):
        name = str(topic.get("topic") or topic.get("title") or f"Topic {index}")
        sections.extend(
            [
                {
                    "title": f"Topic {index} Title",
                    "content": name,
                },
                {
                    "title": f"Topic {index} Description",
                    "content": topic.get("description") or _summary_text(topic),
                },
                {
                    "title": f"Topic {index} Main Results and Importance",
                    "content": topic.get("main_results_and_importance") or _summary_text(topic),
                },
                {
                    "title": f"Topic {index} Reference",
                    "content": _topic_references(topic),
                },
            ]
        )
    return sections


def _fallback_topic_summaries(
    request: SummaryRequest,
    overview: Dict[str, Any],
    paper_summaries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a Julius-owned topic block when specialists did not return topics."""
    topic_name = str(
        overview.get("topic")
        or request.topic_query
        or _topic_from_categories(request)
        or "Recent ArXiv research"
    )
    representative_papers = [
        {
            "title": paper.get("title"),
            "arxiv_id": paper.get("arxiv_id"),
            "id": paper.get("id"),
        }
        for paper in paper_summaries[:MAX_REPRESENTATIVE_PAPERS_PER_TOPIC]
    ]
    return [
        {
            "topic": topic_name,
            "description": overview.get("overview") or f"Recent work clusters around {topic_name}.",
            "main_results_and_importance": _fallback_main_results_text(
                topic_name=topic_name,
                paper_summaries=paper_summaries,
                confidence_notes=overview.get("confidence_notes") or [],
            ),
            "representative_papers": representative_papers,
        }
    ]


def _fallback_main_results_text(
    topic_name: str,
    paper_summaries: List[Dict[str, Any]],
    confidence_notes: List[str],
) -> str:
    """Summarize the strongest paper-level findings for Julius's final topic block."""
    lines: List[str] = []
    for paper in paper_summaries[:MAX_REPRESENTATIVE_PAPERS_PER_TOPIC]:
        title = str(paper.get("title") or "Untitled paper")
        main_result = str(paper.get("main_result") or "Main result pending.")
        significance = str(paper.get("significance") or "").strip()
        line = f"- {title}: {main_result}"
        if significance and significance != main_result:
            line += f" Importance: {significance}"
        lines.append(line)

    if not lines:
        lines.append(f"- Julius did not receive representative paper summaries for {topic_name} yet.")

    if confidence_notes:
        lines.append(f"- Confidence notes: {'; '.join(str(note) for note in confidence_notes)}")

    return "\n".join(lines)


def _summary_text(topic: Dict[str, Any]) -> str:
    summary = topic.get("summary")
    if isinstance(summary, dict):
        return str(summary.get("summary") or summary)
    return str(summary or "Specialist summary pending.")


def _topic_references(topic: Dict[str, Any]) -> List[str]:
    references = []
    for paper in list(topic.get("representative_papers") or [])[:MAX_REPRESENTATIVE_PAPERS_PER_TOPIC]:
        title = str(paper.get("title") or "Untitled paper")
        arxiv_id = paper.get("arxiv_id") or paper.get("id")
        references.append(f"{title} ({arxiv_id})" if arxiv_id else title)
    return references or ["Representative papers pending."]
