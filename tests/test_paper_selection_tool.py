"""Tests for the phase-4.3 representative paper selection tools."""

from src.agents import ChrisAgent
from src.agents.tools import (
    rank_papers_by_relevance_tool,
    select_representative_papers_tool,
)
from src.processing import TopicModeler


class SemanticFakeEmbedder:
    """Deterministic embedder with simple semantic buckets for selection tests."""

    model_name = "semantic-fake"

    def embed_texts(self, texts, batch_size=32):
        texts = list(texts)
        return {
            "embeddings": [self._embedding_for(text) for text in texts],
            "embedding_count": len(texts),
            "dimension": 2,
            "model_name": self.model_name,
            "cache_hits": 0,
            "cache_misses": len(texts),
            "cache_path": "fake-cache.pkl",
        }

    @staticmethod
    def _embedding_for(text):
        lowered = text.lower()
        if "markov" in lowered or "stochastic" in lowered:
            return [1.0, 0.0]
        if "geometry" in lowered or "curvature" in lowered:
            return [0.0, 1.0]
        if "language" in lowered or "agent" in lowered:
            return [0.75, 0.25]
        return [0.5, 0.5]


def _topic_papers():
    return [
        {
            "arxiv_id": "p1",
            "title": "Markov chain mixing",
            "summary": "A stochastic proof for Markov chains.",
            "topic_id": 3,
            "topic_probability": 0.99,
            "categories": ["math.PR"],
        },
        {
            "arxiv_id": "p2",
            "title": "Markov chain coupling",
            "summary": "A stochastic coupling theorem for Markov chains.",
            "topic_id": 3,
            "topic_probability": 0.98,
            "categories": ["math.PR"],
        },
        {
            "arxiv_id": "p3",
            "title": "Curvature and geometric flows",
            "summary": "A geometry view of related limiting behavior.",
            "topic_id": 3,
            "topic_probability": 0.78,
            "categories": ["math.DG"],
        },
        {
            "arxiv_id": "p4",
            "title": "Language agents for theorem search",
            "summary": "Agent planning for mathematical discovery.",
            "topic_id": 7,
            "topic_probability": 0.91,
            "categories": ["cs.CL"],
        },
    ]


def test_topic_modeler_selects_representative_papers_with_diversity():
    """Representative selection balances probability, centroid fit, and diversity."""
    modeler = TopicModeler(embedder=SemanticFakeEmbedder())

    result = modeler.select_representative_papers(
        topic_id=3,
        papers=_topic_papers(),
        n=2,
        diversity_threshold=0.7,
    )

    assert result["topic_id"] == 3
    assert result["candidate_count"] == 3
    assert result["paper_count"] == 2
    assert [paper["arxiv_id"] for paper in result["selected_papers"]] == ["p1", "p3"]
    assert result["selected_papers"][0]["scores"]["representativeness"] > 0
    assert "Selected for high topic fit" in result["selected_papers"][0]["justification"]


def test_select_representative_papers_tool_returns_structured_results():
    """Agent tool wrapper exposes rankings and failure payloads."""
    modeler = TopicModeler(embedder=SemanticFakeEmbedder())

    result = select_representative_papers_tool(
        topic_id=3,
        papers=_topic_papers(),
        n=2,
        diversity_threshold=0.7,
        topic_modeler=modeler,
    )
    failure = select_representative_papers_tool(
        topic_id=99,
        papers=[],
        topic_modeler=modeler,
    )

    assert result["selected_papers"][0]["rank"] == 1
    assert result["selected_papers"][0]["scores"]["combined"] > 0
    assert failure["status"] == "failed"
    assert "No papers" in failure["error"]


def test_rank_papers_by_relevance_tool_orders_by_query_similarity():
    """Query ranking uses semantic similarity between query and paper abstracts."""
    modeler = TopicModeler(embedder=SemanticFakeEmbedder())

    result = rank_papers_by_relevance_tool(
        papers=_topic_papers(),
        query="geometry curvature",
        n=2,
        topic_modeler=modeler,
    )

    assert result["query"] == "geometry curvature"
    assert result["paper_count"] == 2
    assert result["ranked_papers"][0]["arxiv_id"] == "p3"
    assert result["ranked_papers"][0]["relevance_score"] == 1.0


def test_specialized_agents_have_paper_selection_tools():
    """Paper curation tools are registered through the default specialist tool set."""
    agent = ChrisAgent()

    assert "select_representative_papers_tool" in agent.list_tools()
    assert "rank_papers_by_relevance_tool" in agent.list_tools()
