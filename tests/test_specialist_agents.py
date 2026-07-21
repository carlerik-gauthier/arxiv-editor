"""Unit tests for the shared specialist-agent workflow."""

from datetime import datetime
from pathlib import Path

from src import specialist_agent
from src.alain_agent import ALAIN_CATEGORIES, CONFIG as ALAIN_CONFIG, build_alain_agent
from src.abdoulaye_agent import ABDOULAYE_CATEGORIES, build_abdoulaye_agent
from src.bruno_agent import BRUNO_CATEGORIES, build_bruno_agent
from src.chris_agent import CHRIS_CATEGORIES, CONFIG as CHRIS_CONFIG, build_chris_agent
from src.elisa_agent import ELISA_CATEGORIES, build_elisa_agent
from src.felix_agent import FELIX_CATEGORIES, build_felix_agent
from src.jean_baptiste_agent import JEAN_BAPTISTE_CATEGORIES, build_jean_baptiste_agent
from src.data_object import Paper


def _paper(arxiv_id: str, category: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title="A paper",
        authors=["Author"],
        summary="Summary",
        published=datetime(2026, 1, 1),
        updated=datetime(2026, 1, 2),
        categories=[category],
        primary_category=category,
        pdf_url="https://example.test/paper.pdf",
        entry_id="entry",
        comment=None,
        journal_ref=None,
        doi=None,
    )


def test_fetch_papers_defaults_to_specialist_categories(monkeypatch, tmp_path):
    calls = []

    class FakeFetcher:
        def fetch_by_category(self, category, start_date, end_date, max_results):
            calls.append(category)
            return [_paper(category, category)]

    monkeypatch.setattr(specialist_agent, "ArxivFetcher", FakeFetcher)
    monkeypatch.setenv("ARXIV_FETCH_OUTPUT_DIR", str(tmp_path))

    result = specialist_agent.fetch_papers(CHRIS_CONFIG, "2026-01-01", [], "2026-01-31", min_threshold=1)

    assert calls == list(CHRIS_CATEGORIES)
    assert result["status"] == "success"
    assert result["categories"] == list(CHRIS_CATEGORIES)
    assert Path(result["csv_path"]).exists()


def test_fetch_papers_returns_no_csv_below_threshold(monkeypatch):
    class FakeFetcher:
        def fetch_by_category(self, *args):
            return []

    monkeypatch.setattr(specialist_agent, "ArxivFetcher", FakeFetcher)
    result = specialist_agent.fetch_papers(ALAIN_CONFIG, "2026-01-01", ["math.GR"], "2026-01-31", min_threshold=1)

    assert result["status"] == "failure"
    assert result["rows_saved"] == 0
    assert result["csv_path"] is None


def test_find_topics_clamps_topic_count(monkeypatch, tmp_path):
    csv_path = tmp_path / "papers.csv"
    csv_path.write_text("arxiv_id,title,summary\n1,t,s\n", encoding="utf-8")
    captured = {}

    def fake_compute_topics(path, n_topics, n_papers_per_topic):
        captured.update(path=path, n_topics=n_topics, n_papers_per_topic=n_papers_per_topic)
        return [{"topic_title": "Topic"}]

    monkeypatch.setattr(specialist_agent, "compute_topics", fake_compute_topics)
    result = specialist_agent.find_topics(str(csv_path), n_topics=9, n_papers_per_topic=0)

    assert result["status"] == "success"
    assert captured == {"path": str(csv_path), "n_topics": 5, "n_papers_per_topic": 1}


def test_extract_main_result_returns_download_failure(monkeypatch):
    class FakeFetcher:
        def fetch_paper_markdown(self, **kwargs):
            raise RuntimeError("unavailable")

    monkeypatch.setattr(specialist_agent, "ArxivFetcher", FakeFetcher)
    result = specialist_agent.extract_main_result(CHRIS_CONFIG, "2601.12345")

    assert result["status"] == "failure"
    assert result["main_result"] is None


def test_classify_categories_defaults_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert specialist_agent.classify_categories(ALAIN_CONFIG, "group theory") == list(ALAIN_CATEGORIES)


def test_classify_categories_includes_configured_category_descriptions(monkeypatch):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"output_text": '{"categories":["math.AG"]}'})()

    class FakeOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(specialist_agent, "OpenAI", FakeOpenAI)

    assert specialist_agent.classify_categories(ALAIN_CONFIG, "algebraic geometry") == ["math.AG"]
    assert "math.AG: Algebraic Geometry papers." in captured["input"]
    assert "math.GR: Group Theory papers." in captured["input"]


def test_specialist_agents_register_the_same_shared_tools():
    chris_tools = {tool.name for tool in build_chris_agent().tools}
    alain_tools = {tool.name for tool in build_alain_agent().tools}
    expected = {"get_arxiv_categories_tool", "check_paper_tool", "arxiv_fetcher_tool", "find_topic_tool", "extract_main_result_tool"}

    assert chris_tools == expected
    assert alain_tools == expected


def test_specialist_prompt_instructs_the_full_tool_workflow():
    instructions = build_chris_agent().instructions

    for tool_name in (
        "get_arxiv_categories_tool",
        "check_paper_tool",
        "arxiv_fetcher_tool",
        "find_topic_tool",
        "extract_main_result_tool",
    ):
        assert tool_name in instructions
    assert "before fetching" in instructions
    assert "too few papers" in instructions


def test_phase_eight_specialists_register_shared_tools_and_categories():
    agents_and_categories = (
        (build_bruno_agent, BRUNO_CATEGORIES),
        (build_elisa_agent, ELISA_CATEGORIES),
        (build_felix_agent, FELIX_CATEGORIES),
        (build_abdoulaye_agent, ABDOULAYE_CATEGORIES),
        (build_jean_baptiste_agent, JEAN_BAPTISTE_CATEGORIES),
    )
    expected_tools = {"get_arxiv_categories_tool", "check_paper_tool", "arxiv_fetcher_tool", "find_topic_tool", "extract_main_result_tool"}

    for builder, categories in agents_and_categories:
        agent = builder()
        assert {tool.name for tool in agent.tools} == expected_tools
        assert agent.name in agent.instructions
        assert all(category in agent.instructions for category in categories)
