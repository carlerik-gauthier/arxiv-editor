"""Tests for phase-6.1 user request intake and preference persistence."""

from src.agents import JuliusAgent, MichelAgent
from src.generation.user_request import (
    Audience,
    DeliveryMode,
    Depth,
    SummaryFormat,
    SummaryRequestSession,
    Tone,
    clarify_request_tool,
    parse_user_request_tool,
)


def test_parse_user_request_extracts_complete_nontechnical_llm_request():
    """Julius can turn a natural request into a complete SummaryRequest."""
    result = parse_user_request_tool(
        "Give me a non-technical summary of last week's LLM agent papers",
        reference_date="2026-05-12",
    )

    request = result["summary_request"]
    assert result["needs_clarification"] is False
    assert request["topic_query"] == "LLM agent"
    assert request["date_range"]["label"] == "last week"
    assert request["date_range"]["start_date"] == "2026-05-05"
    assert request["date_range"]["end_date"] == "2026-05-12"
    assert request["audience"] == Audience.NON_EXPERT.value
    assert request["tone"] == Tone.PEDAGOGICAL.value
    assert request["depth"] == Depth.STANDARD.value
    assert request["format"] == SummaryFormat.ONE_PAGER.value


def test_parse_user_request_applies_defaults_when_user_is_vague():
    """Vague requests remain executable because the model has sensible defaults."""
    result = parse_user_request_tool("Give me the latest research brief", reference_date="2026-05-12")

    request = result["summary_request"]
    assert result["needs_clarification"] is False
    assert request["audience"] == Audience.MIXED.value
    assert request["date_range"]["label"] == "last week"
    assert request["delivery"]["mode"] == DeliveryMode.PREVIEW.value
    assert "No topic specified" in " ".join(result["assumptions"])


def test_parse_user_request_extracts_categories_limits_format_and_delivery():
    """Scope, limits, format, and delivery preferences are parsed deterministically."""
    result = parse_user_request_tool(
        "Email a bullet digest on cryptography with at most 3 topics and 8 papers, only cs.CR, exclude math.PR, to reader@example.com",
        reference_date="2026-05-12",
    )

    request = result["summary_request"]
    assert request["topic_query"] == "cryptography"
    assert request["format"] == SummaryFormat.BULLET_DIGEST.value
    assert request["max_topics"] == 3
    assert request["max_papers"] == 8
    assert request["must_include_categories"] == ["cs.CR"]
    assert request["exclude_categories"] == ["math.PR"]
    assert request["delivery"]["mode"] == DeliveryMode.EMAIL.value
    assert request["delivery"]["email_recipient"] == "reader@example.com"


def test_clarify_request_asks_only_for_blocking_delivery_detail():
    """Clarification questions are focused and capped."""
    parsed = parse_user_request_tool("Email me a summary of probability", reference_date="2026-05-12")

    clarification = clarify_request_tool(parsed["summary_request"])

    assert parsed["needs_clarification"] is True
    assert clarification["needs_clarification"] is True
    assert clarification["questions"] == [
        "Which email address should I send the finished summary to?"
    ]


def test_summary_request_session_preserves_preferences_across_turns():
    """Refinement turns update only mentioned fields and keep prior preferences."""
    session = SummaryRequestSession()
    first = session.apply_message(
        "Give me a technical summary of LLM agents for last month",
        reference_date="2026-05-12",
    )
    second = session.apply_message("make it shorter")

    assert first["summary_request"]["topic_query"] == "LLM agents"
    assert second["summary_request"]["topic_query"] == "LLM agents"
    assert second["summary_request"]["audience"] == Audience.EXPERT.value
    assert second["summary_request"]["depth"] == Depth.BRIEF.value
    assert second["summary_request"]["date_range"]["label"] == "last month"
    assert len(session.history) == 2


def test_julius_registers_request_intake_tools_and_persists_request():
    """Julius exposes step-6.1 tools through the normal AgentTool system."""
    julius = JuliusAgent(specialist_agents=[MichelAgent()])

    result = julius.execute_tool(
        "parse_user_request_tool",
        {
            "message": "Give me a mixed audience paper ranking on cs.AI from 2026-05-01 to 2026-05-07",
            "reference_date": "2026-05-12",
        },
    )

    assert result.success is True
    assert "parse_user_request_tool" in julius.list_tools()
    assert result.result["summary_request"]["format"] == SummaryFormat.PAPER_RANKINGS.value
    assert result.result["summary_request"]["must_include_categories"] == ["cs.AI"]
    assert julius.request_session.current_request.topic_query is None


def test_julius_compatibility_parser_includes_summary_request_fields():
    """The older Julius planning parser keeps compatibility keys plus preferences."""
    julius = JuliusAgent(specialist_agents=[MichelAgent()])

    parsed = julius.parse_user_request(
        "Summarize last week's cryptography papers for non-experts",
    )

    assert parsed["raw_request"].startswith("Summarize")
    assert parsed["topics"] == ["cryptography"]
    assert parsed["preferences"]["audience"] == Audience.NON_EXPERT.value
    assert parsed["summary_request"]["topic_query"] == "cryptography"
