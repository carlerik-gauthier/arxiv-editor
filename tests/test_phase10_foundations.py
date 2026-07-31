"""Unit tests for Phase 10 refactoring, documentation-facing helpers, and safeguards."""

from __future__ import annotations

from datetime import datetime, date
from pathlib import Path

import pytest

from src import julius_agent, specialist_agent, topic_finder
from src.alain_agent import TOOLS as ALAIN_TOOLS
from src.chris_agent import CHRIS_CATEGORIES, CONFIG as CHRIS_CONFIG, TOOLS as CHRIS_TOOLS
from src.data_object import Paper


def _paper(arxiv_id: str, category: str) -> Paper:
    """Build a minimal valid paper for workflow storage tests.

    Args:
        arxiv_id: Identifier assigned to the synthetic paper.
        category: Primary and only arXiv category for the synthetic paper.

    Returns:
        Paper: Valid paper model with deterministic metadata.
    """
    return Paper(
        arxiv_id=arxiv_id,
        title="A valid paper",
        authors=["Author"],
        summary="A sufficiently descriptive abstract.",
        published=datetime(2026, 5, 20),
        updated=datetime(2026, 5, 20),
        categories=[category],
        primary_category=category,
        pdf_url="https://example.test/paper.pdf",
        entry_id="https://example.test/abs/2605.00001",
    )


def test_shared_tool_factory_preserves_the_required_workflow_tools():
    """Verify every specialist exposes the required shared SDK tools.

    Returns:
        None: Asserts tool factories expose the same five names.
    """
    expected = {
        "get_arxiv_categories_tool",
        "check_paper_tool",
        "arxiv_fetcher_tool",
        "find_topic_tool",
        "extract_main_result_tool",
    }

    assert {tool.name for tool in CHRIS_TOOLS.as_list()} == expected
    assert {tool.name for tool in ALAIN_TOOLS.as_list()} == expected


def test_specialist_config_exposes_its_personality_as_documented_prompt():
    """Verify the public prompt property exposes the configured personality.

    Returns:
        None: Asserts the property equals its configuration source.
    """
    assert CHRIS_CONFIG.system_prompt == CHRIS_CONFIG.personality_and_communication_style


def test_explicit_date_range_is_parsed_without_an_llm(monkeypatch):
    """Verify explicit ISO ranges are parsed without a model request.

    Args:
        monkeypatch: Pytest fixture used to remove the API key.

    Returns:
        None: Asserts reversed explicit dates are returned chronologically.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert specialist_agent.extract_date_range("Papers from 2026-05-21 to 2026-05-19") == (
        "2026-05-19",
        "2026-05-21",
    )


def test_fetch_papers_interprets_an_end_date_as_the_full_calendar_day(monkeypatch, tmp_path):
    """Verify date-only end bounds include the entire final calendar day.

    Args:
        monkeypatch: Pytest fixture used to replace the fetcher and environment.
        tmp_path: Temporary directory used for generated CSV output.

    Returns:
        None: Asserts the fetcher receives the final day's last microsecond.
    """
    received_end_dates: list[datetime] = []

    class FakeFetcher:
        """Capture fetch bounds without making a network request."""

        def fetch_by_category(self, category, start_date, end_date, max_results):
            """Capture the received end date and return a synthetic paper.

            Args:
                category: Requested arXiv category.
                start_date: Inclusive start bound supplied by the workflow.
                end_date: Inclusive end bound recorded for this test.
                max_results: Maximum result count supplied by the workflow.

            Returns:
                list[Paper]: Single synthetic paper in the requested category.
            """
            received_end_dates.append(end_date)
            return [_paper(category, category)]

    monkeypatch.setattr(specialist_agent, "ArxivFetcher", FakeFetcher)
    monkeypatch.setenv("ARXIV_FETCH_OUTPUT_DIR", str(tmp_path))

    result = specialist_agent.fetch_papers(
        CHRIS_CONFIG,
        "2026-05-19",
        [CHRIS_CATEGORIES[0]],
        "2026-05-21",
        min_threshold=1,
    )

    assert result["status"] == "success"
    assert received_end_dates == [datetime(2026, 5, 21, 0, 0, 0, 0)]


def test_topic_finder_rejects_invalid_csv_before_loading_models(tmp_path):
    """Verify invalid CSV input fails before topic models are loaded.

    Args:
        tmp_path: Temporary directory used to create an incomplete CSV.

    Returns:
        None: Asserts missing summary data triggers the expected validation.
    """
    csv_path = tmp_path / "incomplete.csv"
    csv_path.write_text("arxiv_id,title\n1,Only title\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns: summary"):
        topic_finder.compute_topics(csv_path)


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        (None, [0.0, 0.0]),
        ([0.2, 0.8], [0.2, 0.8]),
        ([0.2], [0.0, 0.0]),
    ],
)
def test_topic_confidence_normalization_is_sortable(probabilities, expected):
    """Verify BERTopic confidence variants yield sortable paper scores.

    Args:
        probabilities: Fixture-provided BERTopic confidence representation.
        expected: Scores expected after normalization.

    Returns:
        None: Asserts normalizer output equals the expected score sequence.
    """
    assert topic_finder._confidence_scores(probabilities, 2) == expected


def test_topic_text_handles_bertopic_list_and_empty_values():
    """Verify topic-label normalization handles list and empty representations.

    Returns:
        None: Asserts meaningful label text or its fallback is returned.
    """
    assert topic_finder._topic_text(["A label"], "Fallback") == "A label"
    assert topic_finder._topic_text([], "Fallback") == "Fallback"


def test_paper_model_has_typed_serializable_metadata():
    """Verify paper serialization produces JSON-compatible metadata values.

    Returns:
        None: Asserts datetimes are ISO strings and authors remain a list.
    """
    serialized = _paper("2605.00001", "math.PR").to_dict()

    assert serialized["published"] == "2026-05-20T00:00:00"
    assert serialized["authors"] == ["Author"]


def test_allocation_cap_enforces_the_requested_topic_budget():
    """Verify allocation capping prevents the topic budget from being exceeded.

    Returns:
        None: Asserts the final allocation is shortened to the requested total.
    """
    allocations = [
        {"agent_name": "ChrisAgent", "topic_count": 3},
        {"agent_name": "AlainAgent", "topic_count": 3},
    ]

    assert julius_agent._cap_allocations(allocations, 5) == [
        {"agent_name": "ChrisAgent", "topic_count": 3},
        {"agent_name": "AlainAgent", "topic_count": 2},
    ]


def test_conversation_context_labels_the_new_user_message():
    """Verify current user input is labeled apart from prior conversation.

    Returns:
        None: Asserts serialized history includes the new-message label.
    """
    context = julius_agent._conversation_context(
        [{"role": "user", "content": "Cover group theory."}],
        "Extend the window by one week earlier.",
    )

    assert context == "user: Cover group theory.\nNew user message: Extend the window by one week earlier."


def test_phase_ten_app_and_workflow_documentation_are_present():
    """Verify the documented app and workflow schema are present in the repo.

    Returns:
        None: Asserts the app module and Mermaid workflow diagram exist.
    """
    root = Path(__file__).resolve().parents[1]

    assert (root / "streamlit_phase10.py").is_file()
    workflow = (root / "docs" / "WORKFLOW.md").read_text(encoding="utf-8")
    assert "```mermaid" in workflow
    assert "JuliusAgent" in workflow
