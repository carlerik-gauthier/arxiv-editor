"""Unit tests for the shared specialist-agent workflow."""

from datetime import datetime
from pathlib import Path

from src import specialist_agent
from src.alain_agent import ALAIN_CATEGORIES, CONFIG as ALAIN_CONFIG, build_alain_agent
from src.abdoulaye_agent import ABDOULAYE_CATEGORIES, CONFIG as ABDOULAYE_CONFIG, build_abdoulaye_agent
from src.bruno_agent import BRUNO_CATEGORIES, CONFIG as BRUNO_CONFIG, build_bruno_agent
from src.chris_agent import CHRIS_CATEGORIES, CONFIG as CHRIS_CONFIG, build_chris_agent
from src.elisa_agent import ELISA_CATEGORIES, CONFIG as ELISA_CONFIG, build_elisa_agent
from src.felix_agent import FELIX_CATEGORIES, CONFIG as FELIX_CONFIG, build_felix_agent
from src.jean_baptiste_agent import JEAN_BAPTISTE_CATEGORIES, CONFIG as JEAN_BAPTISTE_CONFIG, build_jean_baptiste_agent
from src.data_object import Paper


def _paper(arxiv_id: str, category: str) -> Paper:
    """Build a minimal valid paper for specialist-workflow tests.

    Args:
        arxiv_id: Identifier assigned to the synthetic paper.
        category: Primary and only arXiv category for the synthetic paper.

    Returns:
        Paper: Valid paper model with deterministic test metadata.
    """
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
    """Verify empty requested categories default to the full specialty.

    Args:
        monkeypatch: Pytest fixture used to replace the fetcher and environment.
        tmp_path: Temporary directory used for generated CSV output.

    Returns:
        None: Asserts every configured category is fetched and persisted.
    """
    calls = []

    class FakeFetcher:
        def fetch_by_category(self, category, start_date, end_date, max_results):
            """Record one category request and return a synthetic paper.

            Args:
                category: Requested arXiv category.
                start_date: Inclusive start date passed by the workflow.
                end_date: Inclusive end date passed by the workflow.
                max_results: Maximum result count passed by the workflow.

            Returns:
                list[Paper]: One paper in the requested category.
            """
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
    """Verify insufficient fetched papers produce no CSV output.

    Args:
        monkeypatch: Pytest fixture used to replace the arXiv fetcher.

    Returns:
        None: Asserts the workflow reports a failure without a CSV path.
    """
    class FakeFetcher:
        def fetch_by_category(self, *args):
            """Return no papers for any mocked category request.

            Args:
                *args: Positional fetch arguments ignored by this test double.

            Returns:
                list[Paper]: Empty result set used to trigger the threshold path.
            """
            return []

    monkeypatch.setattr(specialist_agent, "ArxivFetcher", FakeFetcher)
    result = specialist_agent.fetch_papers(ALAIN_CONFIG, "2026-01-01", ["math.GR"], "2026-01-31", min_threshold=1)

    assert result["status"] == "failure"
    assert result["rows_saved"] == 0
    assert result["csv_path"] is None


def test_find_topics_clamps_topic_count(monkeypatch, tmp_path):
    """Verify requested topic and paper counts are clamped to valid limits.

    Args:
        monkeypatch: Pytest fixture used to replace topic computation.
        tmp_path: Temporary directory used for the input CSV.

    Returns:
        None: Asserts the topic helper receives normalized count values.
    """
    csv_path = tmp_path / "papers.csv"
    csv_path.write_text("arxiv_id,title,summary\n1,t,s\n", encoding="utf-8")
    captured = {}

    def fake_compute_topics(path, n_topics, n_papers_per_topic):
        """Capture normalized topic parameters and return a synthetic topic.

        Args:
            path: CSV path passed by the workflow.
            n_topics: Normalized maximum topic count.
            n_papers_per_topic: Normalized representative-paper count.

        Returns:
            list[dict[str, str]]: Single synthetic topic record.
        """
        captured.update(path=path, n_topics=n_topics, n_papers_per_topic=n_papers_per_topic)
        return [{"topic_title": "Topic"}]

    monkeypatch.setattr(specialist_agent, "compute_topics", fake_compute_topics)
    result = specialist_agent.find_topics(str(csv_path), n_topics=9, n_papers_per_topic=0)

    assert result["status"] == "success"
    assert captured == {"path": str(csv_path), "n_topics": 5, "n_papers_per_topic": 1}


def test_extract_main_result_returns_download_failure(monkeypatch):
    """Verify extraction download errors become recoverable result payloads.

    Args:
        monkeypatch: Pytest fixture used to replace the arXiv fetcher.

    Returns:
        None: Asserts download failure does not expose a main result.
    """
    class FakeFetcher:
        def fetch_paper_markdown(self, **kwargs):
            """Raise a deterministic download error for any extraction request.

            Args:
                **kwargs: Keyword arguments ignored by this failure test double.

            Returns:
                str: This method does not return because it always raises.

            Raises:
                RuntimeError: Always, to simulate an unavailable paper source.
            """
            raise RuntimeError("unavailable")

    monkeypatch.setattr(specialist_agent, "ArxivFetcher", FakeFetcher)
    result = specialist_agent.extract_main_result(CHRIS_CONFIG, "2601.12345")

    assert result["status"] == "failure"
    assert result["main_result"] is None


def test_classify_categories_defaults_without_api_key(monkeypatch):
    """Verify category inference uses all allowed categories without an API key.

    Args:
        monkeypatch: Pytest fixture used to remove the API key.

    Returns:
        None: Asserts the entire configured category set is returned.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert specialist_agent.classify_categories(ALAIN_CONFIG, "group theory") == list(ALAIN_CATEGORIES)


def test_classify_categories_includes_configured_category_descriptions(monkeypatch):
    """Verify model-based category prompts include configured descriptions.

    Args:
        monkeypatch: Pytest fixture used to substitute the OpenAI client.

    Returns:
        None: Asserts selected categories and prompt descriptions are correct.
    """
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            """Capture model request parameters and return a category response.

            Args:
                **kwargs: OpenAI response-creation arguments to record.

            Returns:
                object: Response-shaped object with a selected-category JSON body.
            """
            captured.update(kwargs)
            return type("Response", (), {"output_text": '{"categories":["math.AG"]}'})()

    class FakeOpenAI:
        def __init__(self, api_key):
            """Create a fake OpenAI client exposing fake response operations.

            Args:
                api_key: API key accepted for interface compatibility.

            Returns:
                None: Initializes the fake ``responses`` attribute.
            """
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(specialist_agent, "OpenAI", FakeOpenAI)

    assert specialist_agent.classify_categories(ALAIN_CONFIG, "algebraic geometry") == ["math.AG"]
    assert "math.AG: Algebraic Geometry papers." in captured["input"]
    assert "math.GR: Group Theory papers." in captured["input"]


def test_specialist_agents_register_the_same_shared_tools():
    """Verify base specialists register the same shared tool set.

    Returns:
        None: Asserts Chris and Alain expose identical required tool names.
    """
    chris_tools = {tool.name for tool in build_chris_agent().tools}
    alain_tools = {tool.name for tool in build_alain_agent().tools}
    expected = {"get_arxiv_categories_tool", "check_paper_tool", "arxiv_fetcher_tool", "find_topic_tool", "extract_main_result_tool"}

    assert chris_tools == expected
    assert alain_tools == expected


def test_specialist_prompt_instructs_the_full_tool_workflow():
    """Verify specialist prompts document the complete shared tool workflow.

    Returns:
        None: Asserts every tool and key workflow condition appears in the prompt.
    """
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
    """Verify Phase 8 specialists expose shared tools and own categories.

    Returns:
        None: Asserts each agent prompt contains its name and categories.
    """
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


def test_phase_nine_specialist_prompts_include_personality_guidance():
    """Verify Phase 9 configurations retain their defining personality cues.

    Returns:
        None: Asserts each specialist's prompt includes its expected cue.
    """
    expected_personality_cues = (
        (CHRIS_CONFIG, "encouraging coach"),
        (ALAIN_CONFIG, "wordplay"),
        (BRUNO_CONFIG, "exceptionally rigorous"),
        (ELISA_CONFIG, "cultural awareness"),
        (FELIX_CONFIG, "mad scientist"),
        (ABDOULAYE_CONFIG, "bridging academic research with"),
        (JEAN_BAPTISTE_CONFIG, "senior stakeholders"),
    )

    for config, cue in expected_personality_cues:
        assert cue in config.system_prompt
