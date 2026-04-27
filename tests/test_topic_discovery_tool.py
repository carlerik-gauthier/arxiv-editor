"""Tests for the phase-4.2 BERTopic discovery wrapper and agent tools."""

from src.agents import ChrisAgent
from src.agents.tools import discover_topics_tool, generate_topic_title_tool
import pytest

import src.processing.topic_modeler as topic_modeler_module
from src.processing import TopicModeler, generate_topic_title, normalize_papers


class FakeEmbedder:
    """Deterministic embedder compatible with TopicModeler tests."""

    model_name = "fake-embedding-model"

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts, batch_size=32):
        texts = list(texts)
        self.calls.append({"texts": texts, "batch_size": batch_size})
        return {
            "embeddings": [[float(index), float(len(text))] for index, text in enumerate(texts)],
            "embedding_count": len(texts),
            "dimension": 2,
            "model_name": self.model_name,
            "cache_hits": 0,
            "cache_misses": len(texts),
            "cache_path": "fake-cache.pkl",
        }


class FakeTopicModel:
    """Small BERTopic-compatible model with two coherent topics and outliers."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def fit_transform(self, documents, embeddings=None):
        self.documents = list(documents)
        self.embeddings = list(embeddings or [])
        topics = []
        probabilities = []
        for index, document in enumerate(documents):
            lowered = document.lower()
            if "noise" in lowered:
                topics.append(-1)
                probabilities.append([0.05, 0.05])
            elif "markov" in lowered or "stochastic" in lowered:
                topics.append(1)
                probabilities.append([0.1, 0.92 - index * 0.001])
            else:
                topics.append(0)
                probabilities.append([0.95 - index * 0.001, 0.1])
        return topics, probabilities

    def get_topic(self, topic_id):
        if topic_id == 0:
            return [("language", 0.8), ("models", 0.7), ("agents", 0.5)]
        if topic_id == 1:
            return [("markov", 0.9), ("chains", 0.7), ("mixing", 0.5)]
        return []


def _make_papers(count=100):
    """Create a mixed paper set large enough to exercise topic grouping."""
    papers = []
    language_cutoff = int(count * 0.55)
    markov_cutoff = int(count * 0.95)
    for index in range(count):
        if index < language_cutoff:
            papers.append(
                {
                    "arxiv_id": f"cs-{index}",
                    "title": f"Language agents benchmark {index}",
                    "summary": "We evaluate language models and agent planning.",
                    "categories": ["cs.CL"],
                }
            )
        elif index < markov_cutoff:
            papers.append(
                {
                    "arxiv_id": f"math-{index}",
                    "title": f"Markov chain mixing theorem {index}",
                    "summary": "We prove stochastic convergence for Markov chains.",
                    "categories": ["math.PR"],
                }
            )
        else:
            papers.append(
                {
                    "arxiv_id": f"noise-{index}",
                    "title": f"Noise paper {index}",
                    "summary": "This isolated submission is hard to cluster.",
                    "categories": ["math.GM"],
                }
            )
    return papers


def test_topic_modeler_extracts_topics_with_representative_papers():
    """TopicModeler returns structured topics, keywords, representatives, and progress."""
    modeler = TopicModeler(embedder=FakeEmbedder(), topic_model=FakeTopicModel())

    result = modeler.extract_topics(
        papers=_make_papers(100),
        min_topic_size=5,
        representative_papers_per_topic=5,
        batch_size=25,
    )

    assert result["paper_count"] == 100
    assert result["topic_count"] == 2
    assert result["outlier_count"] == 5
    assert result["representation_model"] == "gpt-4o-mini"
    assert result["representation_models"] == ["MaximalMarginalRelevance", "gpt-4o-mini"]
    assert result["embedding"]["cache_misses"] == 100
    assert [stage["stage"] for stage in result["progress"]] == [
        "normalize_papers",
        "embed_texts",
        "fit_bertopic",
    ]
    assert result["topics"][0]["title"] == "Language / Models / Agents"
    assert len(result["topics"][0]["representative_papers"]) == 5
    assert result["topics"][1]["keywords"] == ["markov", "chains", "mixing"]


def test_topic_modeler_get_topic_info_after_extraction():
    """get_topic_info exposes the latest modeled topic details."""
    modeler = TopicModeler(embedder=FakeEmbedder(), topic_model=FakeTopicModel())
    modeler.extract_topics(_make_papers(20), min_topic_size=3)

    topic_info = modeler.get_topic_info(1)

    assert topic_info["topic_id"] == 1
    assert topic_info["paper_count"] > 0
    assert topic_info["keywords"] == ["markov", "chains", "mixing"]


def test_discover_topics_tool_uses_injected_modeler():
    """Agent tool wrapper delegates to TopicModeler and preserves structured output."""
    modeler = TopicModeler(embedder=FakeEmbedder(), topic_model=FakeTopicModel())

    result = discover_topics_tool(
        papers=_make_papers(30),
        min_topic_size=3,
        topic_modeler=modeler,
    )

    assert result["topic_count"] == 2
    assert result["topics"][0]["representative_papers"][0]["arxiv_id"].startswith("cs-")


def test_discover_topics_tool_returns_failure_payload_for_bad_input():
    """Tool-level error handling returns a structured failure instead of raising."""
    result = discover_topics_tool(papers=[], topic_modeler=TopicModeler(embedder=FakeEmbedder()))

    assert result["status"] == "failed"
    assert result["topic_count"] == 0
    assert "papers must contain" in result["error"]


def test_generate_topic_title_tool_uses_heuristic_and_llm_paths():
    """Topic title tool supports deterministic and injected LLM generation."""
    heuristic = generate_topic_title_tool(
        topic_keywords=["markov", "chains", "mixing"],
        sample_papers=[{"title": "Markov Chain Mixing"}],
    )
    llm = generate_topic_title_tool(
        topic_keywords=["language", "agents"],
        sample_papers=[{"title": "Planning with Language Agents"}],
        llm_client=lambda prompt: "Language Agent Planning",
    )

    assert heuristic == {
        "title": "Markov / Chains / Mixing",
        "source": "heuristic",
        "topic_keywords": ["markov", "chains", "mixing"],
        "sample_paper_count": 1,
    }
    assert llm["title"] == "Language Agent Planning"
    assert llm["source"] == "llm"


def test_specialized_agents_have_topic_tools():
    """Topic tools are registered through the default specialist tool set."""
    agent = ChrisAgent()

    assert "discover_topics_tool" in agent.list_tools()
    assert "generate_topic_title_tool" in agent.list_tools()


def test_normalize_papers_and_fallback_title_generation():
    """Normalization and fallback titles are documented deterministic helpers."""
    records = normalize_papers(
        [{"title": "Spectral Geometry", "abstract": "Eigenvalues on manifolds"}]
    )
    title = generate_topic_title([], [records[0].to_dict()])

    assert records[0].summary == "Eigenvalues on manifolds"
    assert title == "Spectral / Geometry / Eigenvalues"


def test_openai_representation_model_uses_gpt_4o_mini():
    """TopicModeler configures BERTopic's OpenAI representation with gpt-4o-mini."""
    captured = {}

    class FakeOpenAIRepresentation:
        def __init__(self, client, **kwargs):
            captured["client"] = client
            captured.update(kwargs)

    fake_client = object()
    modeler = TopicModeler(openai_client=fake_client)

    representation = modeler._build_openai_representation_model(FakeOpenAIRepresentation)

    assert isinstance(representation, FakeOpenAIRepresentation)
    assert captured["client"] is fake_client
    assert captured["model"] == "gpt-4o-mini"
    assert captured["chat"] is True
    assert captured["generator_kwargs"] == {"temperature": 0}
    assert "[DOCUMENTS]" in captured["prompt"]
    assert "[KEYWORDS]" in captured["prompt"]


def test_representation_models_use_mmr_before_openai():
    """TopicModeler composes MMR before OpenAI for BERTopic representations."""
    created = []

    class FakeMMR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(("mmr", kwargs))

    class FakeOpenAIRepresentation:
        def __init__(self, client, **kwargs):
            self.client = client
            self.kwargs = kwargs
            created.append(("openai", kwargs))

    fake_client = object()
    modeler = TopicModeler(
        openai_client=fake_client,
        mmr_diversity=0.55,
        mmr_top_n_words=12,
    )

    representations = modeler._build_representation_models(
        maximal_marginal_relevance_class=FakeMMR,
        openai_representation_class=FakeOpenAIRepresentation,
    )

    assert len(representations) == 2
    assert [name for name, _kwargs in created] == ["mmr", "openai"]
    assert created[0][1] == {"diversity": 0.55, "top_n_words": 12}
    assert created[1][1]["model"] == "gpt-4o-mini"


def test_representation_models_can_disable_openai_or_mmr():
    """Offline/debug configuration can independently disable representation models."""
    class FakeMMR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOpenAIRepresentation:
        def __init__(self, client, **kwargs):
            self.client = client
            self.kwargs = kwargs

    mmr_only = TopicModeler(use_openai_representation=False)
    openai_only = TopicModeler(
        openai_client=object(),
        use_mmr_representation=False,
    )

    assert len(
        mmr_only._build_representation_models(FakeMMR, FakeOpenAIRepresentation)
    ) == 1
    assert mmr_only._representation_model_names() == ["MaximalMarginalRelevance"]
    assert len(
        openai_only._build_representation_models(FakeMMR, FakeOpenAIRepresentation)
    ) == 1
    assert openai_only._representation_model_names() == ["gpt-4o-mini"]


def test_openai_representation_requires_api_key_when_no_client(monkeypatch):
    """Real OpenAI representation setup fails clearly when no API key is available."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(topic_modeler_module, "_settings_openai_api_key", lambda: "")
    modeler = TopicModeler()

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        modeler._build_openai_representation_model(lambda client, **kwargs: None)
