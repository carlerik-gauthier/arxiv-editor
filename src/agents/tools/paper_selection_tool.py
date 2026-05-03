"""Agent-facing tools for representative paper selection and relevance ranking."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from src.agents.base_agent import AgentTool
from src.processing.topic_modeler import TopicModeler


def select_representative_papers_tool(
    topic_id: int,
    papers: Iterable[Any],
    n: int = 5,
    diversity_threshold: float = 0.7,
    batch_size: int = 32,
    topic_modeler: Optional[TopicModeler] = None,
) -> Dict[str, Any]:
    """
    Select the top papers that best represent one discovered topic.

    Selection balances BERTopic topic probability, semantic centrality to the
    topic centroid, and diversity so agents avoid returning several near-identical
    abstracts. The returned papers include rank, scores, and deterministic
    justifications suitable for Julius or specialist review.
    """
    modeler = topic_modeler or TopicModeler(use_openai_representation=False)
    try:
        return modeler.select_representative_papers(
            topic_id=topic_id,
            papers=papers,
            n=n,
            diversity_threshold=diversity_threshold,
            batch_size=batch_size,
        )
    except Exception as exc:
        return {
            "topic_id": topic_id,
            "selected_papers": [],
            "paper_count": 0,
            "candidate_count": 0,
            "requested_count": n,
            "diversity_threshold": diversity_threshold,
            "status": "failed",
            "error": str(exc),
        }


def rank_papers_by_relevance_tool(
    papers: Iterable[Any],
    query: str,
    n: Optional[int] = None,
    batch_size: int = 32,
    topic_modeler: Optional[TopicModeler] = None,
) -> Dict[str, Any]:
    """
    Re-rank papers by semantic similarity to a specific query or criterion.

    Use this after broad topic discovery when an agent needs to prioritize papers
    for a user interest such as "LLM agents", "security proofs", or "geometric
    intuition" rather than pure topic centrality.
    """
    modeler = topic_modeler or TopicModeler(use_openai_representation=False)
    try:
        return modeler.rank_papers_by_relevance(
            papers=papers,
            query=query,
            n=n,
            batch_size=batch_size,
        )
    except Exception as exc:
        return {
            "query": query,
            "ranked_papers": [],
            "paper_count": 0,
            "candidate_count": 0,
            "status": "failed",
            "error": str(exc),
        }


def get_paper_selection_tool() -> AgentTool:
    """Return the representative paper selector registered with research agents."""
    return AgentTool(
        name="select_representative_papers_tool",
        description=(
            "Select the top N representative papers for a BERTopic topic. Balances "
            "topic probability, distance to the topic centroid, and abstract-level "
            "diversity. Returns ranked papers with scoring details and justifications."
        ),
        function=select_representative_papers_tool,
        required_parameters=["topic_id", "papers"],
    )


def get_paper_relevance_tool() -> AgentTool:
    """Return the query-based paper relevance ranking tool."""
    return AgentTool(
        name="rank_papers_by_relevance_tool",
        description=(
            "Rank papers by semantic similarity to a query or curation criterion. "
            "Use it to re-order candidate papers around a user interest or domain "
            "focus after topic discovery."
        ),
        function=rank_papers_by_relevance_tool,
        required_parameters=["papers", "query"],
    )
