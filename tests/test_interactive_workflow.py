"""Tests for the step-6.6 Streamlit backend workflow."""

from src.agents import AgentTool, JeanBaptisteAgent, JuliusAgent, JuliusSession
from src.agents.tools import check_threshold_tool, generate_summary_tool
from src.generation.interactive_workflow import InteractiveSummaryWorkflow
from src.ui.streamlit_app import (
    append_chat_result,
    apply_output_preferences_to_message,
    build_smoke_workflow,
    collect_agent_activity,
    collect_topic_discovery_debug,
    ensure_workflow_state,
    output_preferences_message,
    process_chat_input,
    run_app,
    should_auto_generate_draft,
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


def test_interactive_workflow_agents_fetch_when_no_matching_data_is_available():
    """Selected agents check paper availability and fetch missing data before drafting."""
    fetched = []

    def fake_fetch_papers_tool(categories, start_date, end_date=None, max_results=10, min_count=None):
        fetched.append(
            {
                "categories": categories,
                "start_date": start_date,
                "end_date": end_date,
                "max_results": max_results,
                "min_count": min_count,
            }
        )
        return {
            "papers": [
                {
                    "title": "Fetched LLM Agent Paper",
                    "summary": "A paper fetched by the specialist agent.",
                    "arxiv_id": "2605.99999",
                    "categories": ["cs.AI"],
                }
            ],
            "paper_count": 1,
            "threshold_met": True,
        }

    def fake_discover_topics_tool(
        papers,
        min_topic_size=2,
        num_topics=None,
        representative_papers_per_topic=5,
        use_openai_representation=True,
    ):
        paper_list = list(papers)
        return {
            "topics": [
                {
                    "title": "Fetched Agent Topic",
                    "description": "LLM-generated fetched topic description.",
                    "description_source": "llm",
                    "keywords": ["agent"],
                    "paper_count": len(paper_list),
                    "representative_papers": paper_list,
                }
            ],
            "topic_count": 1,
            "paper_count": len(paper_list),
            "status": "completed",
        }

    specialist = JeanBaptisteAgent(
        tools=[
            AgentTool(
                name="check_threshold_tool",
                description="Check whether enough papers are available.",
                function=check_threshold_tool,
                required_parameters=["paper_count", "min_threshold"],
            ),
            AgentTool(
                name="fetch_papers_tool",
                description="Fetch papers for the request.",
                function=fake_fetch_papers_tool,
                required_parameters=["categories", "start_date"],
            ),
            AgentTool(
                name="discover_topics_tool",
                description="Discover topics.",
                function=fake_discover_topics_tool,
                required_parameters=["papers"],
            ),
            AgentTool(
                name="generate_summary_tool",
                description="Summarize representative papers.",
                function=generate_summary_tool,
                required_parameters=["papers", "topic"],
            ),
        ]
    )
    workflow = InteractiveSummaryWorkflow(
        julius_session=JuliusSession(
            julius=JuliusAgent(specialist_agents=[specialist]),
            reference_date="2026-05-12",
            run_specialists_in_preview=True,
        )
    )
    workflow.handle_message("Give me an expert summary of LLM agents from last week")

    result = workflow.generate_draft()

    tool_names = [
        call["tool_name"]
        for call in specialist.state["tool_calls"]
    ]
    assert fetched
    assert workflow.session.selected_papers[0]["title"] == "Fetched LLM Agent Paper"
    assert tool_names == [
        "check_threshold_tool",
        "fetch_papers_tool",
        "check_threshold_tool",
        "fetch_papers_tool",
        "discover_topics_tool",
        "generate_summary_tool",
    ]
    assert "generated_first_draft" in result["actions_taken"]


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


def test_ui_chat_surfaces_julius_clarification_questions():
    """Clarification questions are visible in the assistant chat turn."""
    state = {"messages": []}
    result = {
        "message": "I need one detail before generating the draft.",
        "next_questions": ["What custom structure should the summary follow?"],
    }

    append_chat_result(state, "Use a custom format for LLM agents", result)

    assert "Julius needs one clarification:" in state["messages"][-1]["content"]
    assert "What custom structure should the summary follow?" in state["messages"][-1]["content"]


def test_ui_chat_auto_generates_after_complete_initial_request():
    """A complete initial chat request starts Julius's draft work automatically."""
    state = {"messages": []}
    workflow = build_smoke_workflow()

    result = process_chat_input(
        _FakeStreamlit(),
        state,
        workflow,
        "Give me a mixed audience summary of LLM agents",
    )

    assert "generated_first_draft" in result["actions_taken"]
    assert "draft_v1" in state["draft_versions"]
    assert [message["role"] for message in state["messages"]] == [
        "user",
        "assistant",
        "assistant",
    ]


def test_ui_chat_applies_sidebar_output_preferences_to_initial_request():
    """Audience, tone, format, and delivery can come from the sidebar."""
    state = {"messages": []}
    workflow = build_smoke_workflow()

    result = process_chat_input(
        _FakeStreamlit(),
        state,
        workflow,
        "Summarize LLM agents from last week",
        {
            "audience": "expert",
            "tone": "technical",
            "format": "bullet digest",
            "custom_structure": "",
            "delivery": "email",
            "email_recipient": "reader@example.com",
        },
    )

    request = workflow.session.current_request
    assert result["state"] == "AWAITING_REVIEW"
    assert request.audience.value == "expert"
    assert request.tone.value == "technical"
    assert request.format.value == "bullet_digest"
    assert request.delivery.mode.value == "email"
    assert request.delivery.email_recipient == "reader@example.com"


def test_sidebar_output_preferences_are_not_added_to_questions():
    """Sidebar text should not disturb question intent routing."""
    workflow = build_smoke_workflow()
    workflow.handle_message("Summarize LLM agents")
    question = "why did you choose this paper?"
    augmented = apply_output_preferences_to_message(
        question,
        {
            "audience": "expert",
            "tone": "technical",
            "format": "bullet digest",
            "delivery": "preview",
        },
        workflow,
    )

    assert augmented == question


def test_output_preferences_message_formats_delivery_modes():
    """Sidebar choices are converted into phrases Julius's parser understands."""
    assert output_preferences_message(
        {
            "audience": "non-expert",
            "tone": "pedagogical",
            "format": "custom format",
            "custom_structure": "use sections for risks and opportunities",
            "delivery": "file",
        }
    ) == (
        "non-expert, pedagogical tone, custom format, "
        "use sections for risks and opportunities, save to file"
    )


def test_ui_chat_does_not_auto_generate_when_julius_needs_clarification():
    """Clarification questions pause generation until the user answers."""
    state = {"messages": []}
    workflow = build_smoke_workflow()

    result = process_chat_input(
        _FakeStreamlit(),
        state,
        workflow,
        "Email me a summary of LLM agents",
    )

    assert result["state"] == "CLARIFYING"
    assert workflow.session.drafts == []
    assert len(state["messages"]) == 2


def test_ui_collects_agent_and_tool_activity_after_generation():
    """The Streamlit adapter exposes Julius tool calls and specialist handoffs."""
    workflow = build_smoke_workflow()
    workflow.handle_message("Give me a mixed audience summary of LLM agents")
    workflow.generate_draft()

    activity = collect_agent_activity(workflow)

    assert activity["specialist_handoffs"]
    assert {handoff["agent"] for handoff in activity["specialist_handoffs"]} >= {
        "JeanBaptiste",
        "Michel",
    }
    assert "delegate_to_agent_tool" in {
        call["tool"] for call in activity["julius_tool_calls"]
    }


def test_ui_collects_topic_discovery_debug_payload_after_generation():
    """The debug panel can show raw discover_topics_tool results."""
    workflow = build_smoke_workflow()
    workflow.handle_message("Give me a mixed audience summary of LLM agents")
    workflow.generate_draft()

    debug_payload = collect_topic_discovery_debug(workflow)

    assert debug_payload["tool"] == "discover_topics_tool"
    assert debug_payload["call_count"] >= 1
    assert debug_payload["calls"][0]["content"]["topic_count"] >= 1
    assert "topics" in debug_payload["calls"][0]["content"]


def test_auto_generation_predicate_stops_after_first_draft():
    """Preference updates after a draft should use the revision flow, not auto-start intake."""
    workflow = build_smoke_workflow()
    workflow.handle_message("Summarize LLM agents")
    workflow.generate_draft()
    result = {
        "state": "PLANNING",
        "actions_taken": ["updated_summary_request"],
        "next_questions": [],
    }

    assert should_auto_generate_draft(result, workflow) is False


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
