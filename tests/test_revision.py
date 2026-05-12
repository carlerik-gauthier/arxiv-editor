"""Tests for phase-6.4 draft revision tools."""

from src.agents import JuliusSession, JuliusSessionState
from src.generation.revision import (
    RevisionOperation,
    RevisionTarget,
    mark_draft_final_tool,
    parse_revision_request_tool,
    revise_draft_tool,
    rollback_draft_tool,
)
from src.generation.user_request import SummaryRequest


def _draft(version=1):
    """Return a minimal draft with provenance to preserve."""
    return {
        "version": version,
        "content": "# Draft v1\n\n## Topic Overview\nCryptography and LLM agents.\n\n## Representative Papers\n- Paper A",
        "summary_request": SummaryRequest(topic_query="LLM agents").model_dump(mode="json"),
        "provenance": {
            "selected_topics": ["LLM agents"],
            "selected_papers": [{"title": "Paper A", "arxiv_id": "2605.00001"}],
        },
    }


def test_parse_revision_request_detects_operation_target_and_review_need():
    """Natural-language feedback becomes a structured RevisionRequest."""
    parsed = parse_revision_request_tool("make the first topic more intuitive", _draft())

    request = parsed["revision_request"]
    assert request["target"] == RevisionTarget.TOPIC.value
    assert request["operation"] == RevisionOperation.SIMPLIFY.value
    assert parsed["requires_agent_review"] is False
    assert parsed["current_draft_version"] == 1


def test_parse_revision_request_flags_scope_changes_for_reanalysis():
    """Adding/removing/reranking topics requires new data or specialist review."""
    parsed = parse_revision_request_tool("remove cryptography", _draft())

    request = parsed["revision_request"]
    assert request["operation"] == RevisionOperation.REMOVE_TOPIC.value
    assert request["affected_topic"] == "cryptography"
    assert parsed["requires_new_fetch"] is True
    assert parsed["requires_agent_review"] is True


def test_revise_draft_preserves_paper_provenance_and_versions():
    """Local revisions preserve selected papers and append change history."""
    parsed = parse_revision_request_tool("make it shorter", _draft())
    revised = revise_draft_tool(_draft(), parsed, draft_version=2)

    assert revised["version"] == 2
    assert revised["previous_version"] == 1
    assert revised["status"] == "revised"
    assert revised["provenance"]["selected_papers"][0]["title"] == "Paper A"
    assert revised["provenance"]["revision_history"][0]["operation"] == "shorten"
    assert "Revision note" in revised["content"]


def test_rollback_and_final_marking_tools():
    """Draft history can be restored and final drafts become immutable."""
    drafts = [_draft(1), {**_draft(2), "version": 2}]

    restored = rollback_draft_tool(drafts, 1)
    pending = mark_draft_final_tool(restored, approved=False)
    final = mark_draft_final_tool(restored, approved=True)

    assert restored["rollback"]["restored_version"] == 1
    assert pending["approval_required"] is True
    assert final["immutable"] is True
    assert final["status"] == "finalized"


def test_session_revision_versions_rollback_and_finalization():
    """JuliusSession revises drafts, records versions, rolls back, and finalizes."""
    session = JuliusSession(reference_date="2026-05-12")
    session.handle_user_message("Summarize LLM agents")
    session.handle_user_message("generate draft")

    revised = session.handle_user_message("make the first topic more intuitive")
    rollback = session.handle_user_message("rollback to v1")
    final = session.handle_user_message("finalize this")

    assert revised["state"] == JuliusSessionState.AWAITING_REVIEW.value
    assert "parsed_revision_request" in revised["actions_taken"]
    assert "draft_v2" in session.draft_versions
    assert rollback["actions_taken"] == ["rolled_back_draft"]
    assert "Draft v1" in rollback["draft_preview"]
    assert final["state"] == JuliusSessionState.FINALIZED.value
    assert session.drafts[-1]["immutable"] is True
