"""Agent-facing tools for topic discovery and topic title generation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from src.agents.base_agent import AgentTool
from src.openai_client import default_openai_model, resolve_openai_client
from src.processing.topic_modeler import (
    MAX_REPRESENTATIVE_PAPERS_PER_TOPIC,
    TopicModeler,
    generate_topic_title,
)


def discover_topics_tool(
    papers: Iterable[Any],
    min_topic_size: int = 5,
    num_topics: Optional[int] = None,
    representative_papers_per_topic: int = MAX_REPRESENTATIVE_PAPERS_PER_TOPIC,
    batch_size: int = 32,
    representation_model_name: str = "gpt-4o-mini",
    use_openai_representation: bool = True,
    use_mmr_representation: bool = True,
    mmr_diversity: float = 0.3,
    mmr_top_n_words: int = 10,
    openai_api_key: Optional[str] = None,
    openai_client: Optional[Any] = None,
    topic_modeler: Optional[TopicModeler] = None,
) -> Dict[str, Any]:
    """
    Discover coherent research themes from paper metadata.

    The tool embeds each paper title and abstract, runs BERTopic, and returns
    topic titles, keywords, representative papers, cache metadata, and progress
    stages. A TopicModeler can be injected for tests or alternate backends.
    """
    paper_list = [dict(paper) if isinstance(paper, dict) else paper for paper in papers]
    if not paper_list:
        return {
            "topics": [],
            "topic_count": 0,
            "paper_count": 0,
            "outlier_count": 0,
            "status": "failed",
            "error": "papers must contain at least one paper",
            "progress": [
                {
                    "stage": "discover_topics",
                    "status": "failed",
                    "error": "papers must contain at least one paper",
                }
            ],
        }
    if len(paper_list) < max(3, min_topic_size):
        return _small_corpus_topic_result(paper_list, representative_papers_per_topic)

    modeler = topic_modeler or TopicModeler(
        representation_model_name=representation_model_name,
        openai_api_key=openai_api_key,
        openai_client=openai_client,
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


def _small_corpus_topic_result(
    papers: Sequence[Any],
    representative_papers_per_topic: int,
) -> Dict[str, Any]:
    """Return a deterministic topic result when BERTopic has too little data."""
    normalized_papers = [paper if isinstance(paper, dict) else {"title": str(paper)} for paper in papers]
    representative_papers_per_topic = min(
        representative_papers_per_topic,
        MAX_REPRESENTATIVE_PAPERS_PER_TOPIC,
    )
    representative_papers = [
        {
            "arxiv_id": paper.get("arxiv_id"),
            "title": paper.get("title", "Untitled paper"),
            "summary": paper.get("summary") or paper.get("abstract", ""),
            "categories": paper.get("categories", []),
            "rank": index + 1,
        }
        for index, paper in enumerate(normalized_papers[:representative_papers_per_topic])
    ]
    keywords = _small_corpus_keywords(normalized_papers)
    title = generate_topic_title(keywords, representative_papers)
    return {
        "topics": [
            {
                "topic_id": 0,
                "title": title,
                "keywords": keywords,
                "paper_count": len(normalized_papers),
                "representative_papers": representative_papers,
            }
        ] if normalized_papers else [],
        "topic_count": 1 if normalized_papers else 0,
        "paper_count": len(normalized_papers),
        "outlier_count": 0,
        "status": "completed",
        "source": "small_corpus_fallback",
        "progress": [
            {
                "stage": "small_corpus_topic_discovery",
                "status": "completed",
                "paper_count": len(normalized_papers),
            }
        ],
    }


def _small_corpus_keywords(papers: Sequence[Dict[str, Any]]) -> list[str]:
    """Extract simple keywords from a tiny paper set."""
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "paper", "study",
        "studies", "introduce", "evaluate", "using", "into", "large", "model",
        "models", "research",
    }
    counts: Dict[str, int] = {}
    for paper in papers:
        text = f"{paper.get('title', '')} {paper.get('summary') or paper.get('abstract', '')}"
        for raw_token in text.lower().replace("-", " ").split():
            token = "".join(character for character in raw_token if character.isalnum())
            if len(token) < 4 or token in stopwords:
                continue
            counts[token] = counts.get(token, 0) + 1
    return [
        token
        for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    ]


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
    paper_titles = [
        str(paper.get("title", ""))
        for paper in sample_papers[:MAX_REPRESENTATIVE_PAPERS_PER_TOPIC]
    ]
    return (
        "Create a concise research topic title.\n"
        f"Keywords: {', '.join(topic_keywords)}\n"
        f"Representative papers: {'; '.join(paper_titles)}\n"
        "Return only the title."
    )


def _call_title_llm(llm_client: Any, prompt: str) -> str:
    """Call common LLM client shapes and normalize the returned title."""
    client = resolve_openai_client(llm_client, required=True)
    responses_api = getattr(client, "responses", None)
    create_method = getattr(responses_api, "create", None)
    if callable(create_method):
        response = create_method(
            model=getattr(client, "model", default_openai_model()),
            input=prompt,
        )
    else:
        chat_api = getattr(client, "chat", None)
        completions_api = getattr(chat_api, "completions", None)
        create_method = getattr(completions_api, "create", None)
        if not callable(create_method):
            raise TypeError(
                "llm_client must be an OpenAI client exposing responses.create or "
                "chat.completions.create"
            )
        response = create_method(
            model=getattr(client, "model", default_openai_model()),
            messages=[{"role": "user", "content": prompt}],
        )

    if isinstance(response, dict):
        response = response.get("title") or response.get("content") or response.get("text")
    else:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            response = output_text
    title = str(response).strip().strip('"')
    if not title:
        raise ValueError("llm_client returned an empty topic title")
    return title
