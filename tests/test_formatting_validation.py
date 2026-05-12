"""Tests for phase-6.5 formatting, validation, and final output."""

from src.agents import JuliusSession
from src.agents.tools import format_document_tool, validate_quality_tool
from src.generation.formatter import DocumentFormatter
from src.generation.user_request import DeliveryMode, DeliveryPreference, SummaryRequest
from src.generation.validator import DocumentValidator


def _draft():
    """Return a draft with metadata for formatting and validation."""
    return {
        "version": 1,
        "title": "LLM agents: Research Brief",
        "content": "# Draft v1\n\n## Topic Overview\nLLM agents improve planning.\n\n## Representative Papers\n- Agent Planning Benchmarks (2605.00001)",
        "summary_request": SummaryRequest(topic_query="LLM agents").model_dump(mode="json"),
        "provenance": {
            "selected_topics": ["LLM agents"],
            "selected_papers": [{"title": "Agent Planning Benchmarks", "arxiv_id": "2605.00001"}],
            "agent_callbacks": [{"agent": "JeanBaptiste", "status": "COMPLETED"}],
        },
    }


def test_document_formatter_renders_markdown_html_and_pdf():
    """DocumentFormatter supports the required output formats."""
    formatter = DocumentFormatter()
    template = formatter.apply_template(_draft(), "one_pager")

    markdown = formatter.render_to_format(template, "markdown")
    html = formatter.render_to_format(template, "html")
    pdf = formatter.render_to_format(template, "pdf")

    assert "Generated at:" in markdown
    assert "<html>" in html
    assert pdf.startswith(b"%PDF")


def test_format_document_tool_returns_serializable_metadata():
    """The agent tool reports format metadata and rendered content."""
    result = format_document_tool(_draft(), output_format="markdown")

    assert result["output_format"] == "markdown"
    assert result["is_binary"] is False
    assert "Agent Credits" in result["document"]


def test_document_validator_checks_request_and_source_papers():
    """Validator checks topic visibility and source paper citations."""
    request = SummaryRequest(topic_query="LLM agents")
    document = format_document_tool(_draft())["document"]

    report = DocumentValidator().validate(
        document,
        request,
        [{"title": "Agent Planning Benchmarks", "arxiv_id": "2605.00001"}],
    )

    assert report["passed"] is True
    assert report["source_paper_count"] == 1
    assert report["warnings"] == []


def test_validate_quality_tool_adds_suggestions():
    """The quality tool wraps validation with improvement suggestions."""
    report = validate_quality_tool(
        "A draft without the requested topic",
        SummaryRequest(topic_query="cryptography"),
    )

    assert report["warnings"]
    assert report["suggestions"][0].startswith("Address:")


def test_session_finalization_formats_validates_and_saves(tmp_path):
    """JuliusSession saves a validated final document to outputs."""
    session = JuliusSession(
        reference_date="2026-05-12",
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
    session.handle_user_message("Summarize LLM agents")
    session.handle_user_message("generate draft")

    final = session.handle_user_message("finalize this")

    assert "saved_final_document" in final["actions_taken"]
    assert session.final_output_path is not None
    assert session.validation_reports
    saved = tmp_path.glob("*.md")
    saved_path = next(saved)
    assert "Generated at:" in saved_path.read_text(encoding="utf-8")
    assert session.drafts[-1]["immutable"] is True


def test_session_email_delivery_formats_final_as_html(tmp_path):
    """Email delivery prepares HTML for the later email workflow."""
    request = SummaryRequest(
        topic_query="LLM agents",
        delivery=DeliveryPreference(mode=DeliveryMode.EMAIL, email_recipient="reader@example.com"),
    )
    session = JuliusSession(reference_date="2026-05-12", output_dir=tmp_path)
    session.current_request = request
    session.handle_user_message("generate draft")

    final = session.handle_user_message("finalize this")

    assert "finalized_for_email_delivery" in final["actions_taken"]
    assert session.final_output_path.endswith(".html")
