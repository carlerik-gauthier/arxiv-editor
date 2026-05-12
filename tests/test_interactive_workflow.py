"""Tests for the step-6.6 Streamlit backend workflow."""

from src.agents import JuliusSession
from src.generation.interactive_workflow import InteractiveSummaryWorkflow
from src.ui.streamlit_app import (
    append_chat_result,
    build_smoke_workflow,
    ensure_workflow_state,
    run_app,
    sync_session_state,
)


def test_interactive_workflow_happy_path_revision_and_finalization(tmp_path):
    """A mocked workflow supports intake, generation, revision, question, finalize."""
    workflow = InteractiveSummaryWorkflow(
        julius_session=JuliusSession(
            output_dir=tmp_path,
            selected_papers=[
                {
                    "title": "Agent Planning Benchmarks",
                    "summary": "LLM agents improve planning.",
                    "arxiv_id": "2605.00001",
                    "score": 1.0,
                }
            ],
        )
    )

    intake = workflow.handle_message("Give me a mixed audience summary of LLM agents")
    draft = workflow.generate_draft()
    revised = workflow.revise_draft("make it shorter")
    question = workflow.handle_message("why did you choose this paper?")
    final = workflow.finalize()

    assert intake["summary_request"]["topic_query"] == "LLM agents"
    assert "generated_first_draft" in draft["actions_taken"]
    assert "revised_draft" in revised["actions_taken"]
    assert "answered_draft_question" in question["actions_taken"]
    assert "saved_final_document" in final["actions_taken"]
    assert workflow.session.final_output_path is not None
    assert workflow.satisfaction_signals[-1]["type"] == "accepted_draft"


def test_interactive_workflow_clarification_path():
    """Blocking clarifications are surfaced as next questions."""
    workflow = InteractiveSummaryWorkflow()

    result = workflow.handle_message("Email me a summary of probability")

    assert result["next_questions"] == [
        "Which email address should I send the finished summary to?"
    ]
    assert result["state"] == "CLARIFYING"


def test_interactive_workflow_revision_scope_without_restart():
    """Preference and scope changes reuse the same JuliusSession."""
    workflow = InteractiveSummaryWorkflow()
    workflow.handle_message("Summarize cryptography")

    result = workflow.apply_preferences(
        {
            "topic_query": "LLM agents",
            "date_range": "last 14 days",
            "audience": "expert",
            "depth": "deep",
            "tone": "technical",
            "format": "paper_rankings",
            "must_include_categories": ["cs.AI"],
            "exclude_categories": ["math.AG"],
            "max_topics": 3,
            "max_papers": 4,
        }
    )

    request = result["summary_request"]
    assert request["topic_query"] == "LLM agents"
    assert request["audience"] == "expert"
    assert request["must_include_categories"] == ["cs.AI"]
    assert workflow.session.current_request.max_papers == 4


def test_interactive_workflow_failure_is_recoverable():
    """Mocked data failures produce recoverable Julius messages."""
    def failing_fetch(_request):
        raise RuntimeError("ArXiv unavailable")

    workflow = InteractiveSummaryWorkflow(fetch_papers_fn=failing_fetch)
    workflow.handle_message("Summarize LLM agents")

    result = workflow.generate_draft()

    assert result["recoverable"] is True
    assert "handled_recoverable_workflow_error" in result["actions_taken"]
    assert workflow.last_error == "ArXiv unavailable"


def test_ui_session_state_survives_rerun_and_syncs_backend_state():
    """UI helpers preserve workflow, draft versions, validation, and final path."""
    state = {}
    workflow = build_smoke_workflow()
    first = ensure_workflow_state(state, workflow)
    second = ensure_workflow_state(state)
    result = first.handle_message("Summarize LLM agents")

    append_chat_result(state, "Summarize LLM agents", result)
    first.generate_draft()
    sync_session_state(state, first)

    assert first is second
    assert state["messages"][0]["role"] == "user"
    assert "draft_v1" in state["draft_versions"]
    assert isinstance(state["validation_reports"], list)


def test_app_import_and_streamlit_adapter_smoke_build_without_api_keys():
    """The Streamlit page can be built with a fake Streamlit module."""
    import app

    fake = _FakeStreamlit()
    run_app(fake)

    assert callable(app.main)
    assert fake.page_config["page_title"] == "Julius ArXiv Editor"
    assert "workflow" in fake.session_state


class _Context:
    """Tiny context manager for fake Streamlit containers."""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeColumn(_Context):
    """Fake column where buttons are not clicked."""

    def button(self, _label):
        return False


class _FakeStreamlit:
    """Minimal Streamlit surface used by `run_app` smoke tests."""

    def __init__(self):
        self.session_state = {}
        self.sidebar = _Context()
        self.page_config = {}

    def set_page_config(self, **kwargs):
        self.page_config = kwargs

    def title(self, _text):
        pass

    def header(self, _text):
        pass

    def text_input(self, _label, value=""):
        return value

    def selectbox(self, _label, options):
        return options[0]

    def multiselect(self, _label, _options):
        return []

    def slider(self, _label, min_value, max_value, value):
        return value

    def columns(self, count):
        return [_FakeColumn() for _ in range(count)]

    def status(self, *_args, **_kwargs):
        return _Context()

    def chat_message(self, _role):
        return _Context()

    def chat_input(self, _placeholder):
        return None

    def tabs(self, labels):
        return [_Context() for _ in labels]

    def write(self, _content):
        pass

    def markdown(self, _content):
        pass

    def json(self, _content):
        pass

    def download_button(self, *_args, **_kwargs):
        pass

    def rerun(self):
        pass
