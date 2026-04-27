"""Agent-facing tools for topic discovery and topic title generation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from src.agents.base_agent import AgentTool
from src.processing.topic_modeler import TopicModeler, generate_topic_title


def discover_topics_tool(
    papers: Iterable[Any],
    min_topic_size: int = 5,
    num_topics: Optional[int] = None,
    representative_papers_per_topic: int = 5,
    batch_size: int = 32,
    representation_model_name: str = "gpt-4o-mini",
    use_openai_representation: bool = True,
    use_mmr_representation: bool = True,
    mmr_diversity: float = 0.3,
    mmr_top_n_words: int = 10,
    openai_api_key: Optional[str] = None,
    topic_modeler: Optional[TopicModeler] = None,
) -> Dict[str, Any]:
    """
    Discover coherent research themes from paper metadata.

    The tool embeds each paper title and abstract, runs BERTopic, and returns
    topic titles, keywords, representative papers, cache metadata, and progress
    stages. A TopicModeler can be injected for tests or alternate backends.
    """
    modeler = topic_modeler or TopicModeler(
        representation_model_name=representation_model_name,
        openai_api_key=openai_api_key,
        use_openai_representation=use_openai_representation,
        use_mmr_representation=use_mmr_representation,
        mmr_diversity=mmr_diversity,
        mmr_top_n_words=mmr_top_n_words,
    )
    try:
        return modeler.extract_topics(
            papers=papers,
            min_topic_size=min_topic_size,
            num_topics=num_topics,
            representative_papers_per_topic=representative_papers_per_topic,
            batch_size=batch_size,
        )
    except Exception as exc:
        return {
            "topics": [],
            "topic_count": 0,
            "paper_count": 0,
            "outlier_count": 0,
            "status": "failed",
            "error": str(exc),
            "progress": [
                {
                    "stage": "discover_topics",
                    "status": "failed",
                    "error": str(exc),
                }
            ],
        }


def generate_topic_title_tool(
    topic_keywords: Sequence[str],
    sample_papers: Sequence[Dict[str, Any]],
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate a human-readable title for a discovered topic.

    If an LLM client is supplied, it is called with keywords and sample paper
    titles. Without an LLM, the tool falls back to a deterministic keyword title
    so tests and offline workflows remain stable.
    """
    if llm_client is not None:
        prompt = _build_title_prompt(topic_keywords, sample_papers)
        title = _call_title_llm(llm_client, prompt)
        source = "llm"
    else:
        title = generate_topic_title(topic_keywords, sample_papers)
        source = "heuristic"

    return {
        "title": title,
        "source": source,
        "topic_keywords": list(topic_keywords),
        "sample_paper_count": len(sample_papers),
    }


def get_topic_discovery_tool() -> AgentTool:
    """Return the BERTopic discovery tool registered with research agents."""
    return AgentTool(
        name="discover_topics_tool",
        description=(
            "Discover research themes from paper titles and abstracts using "
            "custom embeddings plus BERTopic. Returns topic titles, keywords, "
            "representative papers, outlier counts, and progress metadata."
        ),
        function=discover_topics_tool,
        required_parameters=["papers"],
    )


def get_topic_title_tool() -> AgentTool:
    """Return the topic title generation tool registered with research agents."""
    return AgentTool(
        name="generate_topic_title_tool",
        description=(
            "Generate a concise, human-readable topic title from BERTopic keywords "
            "and representative paper metadata. Can use an injected LLM client."
        ),
        function=generate_topic_title_tool,
        required_parameters=["topic_keywords", "sample_papers"],
    )


def _build_title_prompt(
    topic_keywords: Sequence[str],
    sample_papers: Sequence[Dict[str, Any]],
) -> str:
    """Build a compact prompt for LLM-backed topic title generation."""
    paper_titles = [str(paper.get("title", "")) for paper in sample_papers[:5]]
    return (
        "Create a concise research topic title.\n"
        f"Keywords: {', '.join(topic_keywords)}\n"
        f"Representative papers: {'; '.join(paper_titles)}\n"
        "Return only the title."
    )


def _call_title_llm(llm_client: Any, prompt: str) -> str:
    """Call common LLM client shapes and normalize the returned title."""
    if callable(llm_client):
        response = llm_client(prompt)
    else:
        for method_name in ("complete", "generate", "chat", "invoke"):
            method = getattr(llm_client, method_name, None)
            if callable(method):
                response = method(prompt)
                break
        else:
            raise TypeError("llm_client must be callable or expose complete/generate/chat/invoke")

    if isinstance(response, dict):
        response = response.get("title") or response.get("content") or response.get("text")
    title = str(response).strip().strip('"')
    if not title:
        raise ValueError("llm_client returned an empty topic title")
    return title
