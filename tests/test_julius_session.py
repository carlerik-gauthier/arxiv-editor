"""Tests for Julius's phase-6.2 interactive conversation session."""

from src.agents import (
    JuliusIntent,
    JuliusSession,
    JuliusSessionState,
    classify_user_intent_tool,
    explain_draft_choice_tool,
    update_summary_request_tool,
)
from src.generation.user_request import Audience, Depth, SummaryFormat


def test_intent_classifier_routes_core_session_messages():
    """The deterministic classifier covers the phase-6.2 intent set."""
    assert classify_user_intent_tool("Summarize LLM agent papers")["intent"] == JuliusIntent.NEW_SUMMARY_REQUEST.value
    assert classify_user_intent_tool("make it shorter")["intent"] == JuliusIntent.PREFERENCE_UPDATE.value
    assert classify_user_intent_tool("only cs.AI from last 14 days")["intent"] == JuliusIntent.SCOPE_UPDATE.value
    assert classify_user_intent_tool("why did you choose this paper?")["intent"] == JuliusIntent.DRAFT_QUESTION.value
    assert classify_user_intent_tool("generate draft")["intent"] == JuliusIntent.GENERATE_DRAFT.value
    assert classify_user_intent_tool("finalize this")["intent"] == JuliusIntent.FINALIZATION.value


def test_update_summary_request_tool_keeps_prior_preferences():
    """Refinement updates are sticky and do not erase prior scope."""
    initial = update_summary_request_tool(
        None,
        "Give me a technical summary of LLM agents from last month",
        reference_date="2026-05-12",
    )
    updated = update_summary_request_tool(
        initial["summary_request"],
        "make it a bullet digest and shorter",
        reference_date="2026-05-12",
    )

    request = updated["summary_request"]
    assert request["topic_query"] == "LLM agents"
    assert request["audience"] == Audience.EXPERT.value
    assert request["depth"] == Depth.BRIEF.value
    assert request["format"] == SummaryFormat.BULLET_DIGEST.value
    assert request["date_range"]["label"] == "last month"


def test_session_intake_then_generation_then_finalization():
    """A user can request, steer, generate, question, and finalize in one session."""
    progress_events = []
    session = JuliusSession(
        progress_callback=progress_events.append,
        reference_date="2026-05-12",
    )

    first = session.handle_user_message(
        "Give me a mixed audience summary of LLM agents from last week"
    )
    second = session.handle_user_message("make it shorter and only cs.AI")
    draft = session.handle_user_message("generate draft")
    answer = session.handle_user_message("why did you choose these topics?")
    final = session.handle_user_message("finalize this")

    assert first["state"] == JuliusSessionState.PLANNING.value
    assert first["summary_request"]["topic_query"] == "LLM agents"
    assert second["summary_request"]["depth"] == Depth.BRIEF.value
    assert second["summary_request"]["must_include_categories"] == ["cs.AI"]
    assert draft["state"] == JuliusSessionState.AWAITING_REVIEW.value
    assert "Draft v1" in draft["draft_preview"]
    assert progress_events == [
        "Preparing the paper search scope.",
        "Modeling candidate topics.",
        "Preparing specialist review tasks.",
        "Compiling the draft preview.",
    ]
    assert answer["actions_taken"] == ["answered_draft_question"]
    assert "current topic" in answer["message"]
    assert final["state"] == JuliusSessionState.FINALIZED.value
    assert "finalized_draft" in final["actions_taken"]
    assert len(session.conversation_history) == 10


def test_session_asks_for_blocking_email_clarification():
    """Email delivery without a recipient blocks generation with a focused question."""
    session = JuliusSession(reference_date="2026-05-12")

    response = session.handle_user_message("Email me a summary of probability")
    draft_attempt = session.handle_user_message("generate draft")

    assert response["state"] == JuliusSessionState.CLARIFYING.value
    assert response["next_questions"] == [
        "Which email address should I send the finished summary to?"
    ]
    assert draft_attempt["state"] == JuliusSessionState.CLARIFYING.value
    assert "blocked_generation_for_clarification" in draft_attempt["actions_taken"]


def test_session_revises_existing_draft_preview():
    """Feedback after a draft creates a new version and records the feedback."""
    session = JuliusSession(reference_date="2026-05-12")
    session.handle_user_message("Summarize cryptography papers")
    session.handle_user_message("generate draft")

    revised = session.handle_user_message("make it more technical")

    assert revised["state"] == JuliusSessionState.AWAITING_REVIEW.value
    assert "Draft v2" in revised["draft_preview"]
    assert revised["summary_request"]["tone"] == "technical"
    assert len(session.drafts) == 2
    assert session.user_feedback[0]["message"] == "make it more technical"


def test_explain_draft_choice_tool_reports_deferred_paper_selection():
    """Draft explanations use provenance rather than inventing paper details."""
    answer = explain_draft_choice_tool(
        {"version": 1, "provenance": {"selected_papers": []}},
        "what is the main result of the selected paper?",
    )

    assert answer["draft_version"] == 1
    assert "No representative papers" in answer["answer"]
