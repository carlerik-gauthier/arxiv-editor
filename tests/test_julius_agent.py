"""Unit tests for JuliusAgent's phase-six routing helpers."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src import julius_agent
from src.field_family import FieldFamily, family_for_agent


def test_supported_request_detection_handles_specialist_domains(monkeypatch):
    """Verify keyword routing recognizes representative specialist domains.

    Args:
        monkeypatch: Pytest fixture used to remove the API key.

    Returns:
        None: Asserts representative requests are considered in scope.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert julius_agent._is_supported_specialist_request("Two topics on stochastic processes")
    assert julius_agent._is_supported_specialist_request("One topic in group theory")
    assert julius_agent._is_supported_specialist_request("Recent Riemannian geometry papers")
    assert julius_agent._is_supported_specialist_request("Recent machine learning papers")


def test_family_helper_separates_mathematics_and_ai_agents():
    """Verify configured specialists map to their intended research families.

    Returns:
        None: Asserts one mathematics and one AI agent map correctly.
    """
    assert family_for_agent("BrunoAgent") is FieldFamily.MATHEMATICS
    assert family_for_agent("AbdoulayeAgent") is FieldFamily.AI


def test_normalize_allocation_discards_invalid_agents():
    """Verify allocation normalization removes unavailable specialist names.

    Returns:
        None: Asserts only a valid allocation is retained and enriched.
    """
    allocations = julius_agent._normalize_allocation_payload(
        {"allocations": [{"agent_name": "ChrisAgent", "topic_count": 2}, {"agent_name": "Unknown", "topic_count": 3}]},
        ["ChrisAgent", "AlainAgent"],
    )

    assert allocations == [{"agent_name": "ChrisAgent", "tool_name": "chris_agent_tool", "topic_count": 2, "reason": "Interest in probability or statistics detected."}]


def test_parse_json_object_response_accepts_fenced_json():
    """Verify response parsing accepts JSON enclosed in Markdown fences.

    Returns:
        None: Asserts the parsed mapping equals the fenced input object.
    """
    assert julius_agent._parse_json_object_response("```json\n{\"status\": \"compiled\"}\n```", "test") == {"status": "compiled"}


def test_run_julius_agent_declines_out_of_scope_request(monkeypatch):
    """Verify unsupported requests receive a safe no-delegation response.

    Args:
        monkeypatch: Pytest fixture used to remove the API key.

    Returns:
        None: Asserts the rejection names supported domains and uses no tools.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = julius_agent.run_julius_agent("Cover recent astrophysics papers.")

    assert "coordinate mathematics and AI research briefs" in result["reply"]
    assert result["tool_parameters"] == []


def test_julius_prompt_preserves_specialist_voices_with_requested_tone():
    """Verify the editorial prompt preserves specialist and user style cues.

    Returns:
        None: Asserts the instructions mention voice preservation and user style.
    """
    instructions = julius_agent.build_julius_agent().instructions

    assert "Preserve each delegated specialist's personality" in instructions
    assert "user's requested tone, audience" in instructions


def test_julius_prompt_makes_him_the_final_editorial_arbiter():
    """Verify specialists and readability reviewers cannot make the final call."""
    instructions = julius_agent.build_julius_agent().instructions

    assert "chief editor and final arbiter" in instructions
    assert "Specialist outputs are candidate material" in instructions
    assert "MichelAgent's review is advisory" in instructions
    assert "exactly the requested number of suitable topics" in instructions


def test_julius_prompt_handles_reviews_and_existing_drafts_editorially():
    """Verify Julius delivers reviewable drafts and refines prior drafts himself."""
    instructions = julius_agent.build_julius_agent().instructions

    assert "Never ask the user to choose" in instructions
    assert "deliver the complete current draft" in instructions
    assert "If a draft already exists in the conversation" in instructions
    assert "Do not restart topic allocation or specialist research" in instructions


def test_julius_prompt_uses_michel_as_an_advisory_readability_reviewer():
    """Verify Julius makes the final editorial choice after Michel's review.

    Returns:
        None: Asserts the prescribed draft, review, incorporation, and
        follow-up-review sequence is present in the coordinator instructions.
    """
    instructions = julius_agent.build_julius_agent().instructions

    first_draft = instructions.index("first draft must contain no pedagogical explanation")
    initial_review = instructions.index("After the first draft, always call `michel_agent_tool`")
    incorporation = instructions.index("Julius decides whether to apply, rephrase, or reject")
    follow_up_review = instructions.index("After every call to `finalize_editorial_one_pager_tool`")

    assert first_draft < initial_review < incorporation < follow_up_review
    assert "Treat MichelAgent's readability feedback and proposed explanations as suggestions" in instructions
    assert "After the third Michel review, Julius must wrap up and deliver" in instructions


def test_julius_prompt_uses_explicit_michel_placeholders_for_first_drafts():
    """Verify a first draft records where Michel must add pedagogy.

    Returns:
        None: Asserts the coordinator labels each potential pedagogical
        location for Julius's later editorial decision.
    """
    instructions = julius_agent.build_julius_agent().instructions

    assert "MICHEL_PEDAGOGY" in instructions
    assert 'needed="yes"` or `needed="no"' in instructions
    assert "may accept, rephrase, or reject Michel's suggestions" in instructions


def test_find_michel_placeholders_returns_only_unresolved_locations():
    """Verify unresolved Michel placeholder locations are detectable.

    Returns:
        None: Asserts identifiers are returned in their document order and
        regular prose is ignored.
    """
    draft = (
        'Technical description. [[MICHEL_PEDAGOGY id="topic-1-description" needed="yes"]]\n'
        'Result. [[MICHEL_PEDAGOGY id="topic-1-paper-1-result" needed="no"]]'
    )

    assert julius_agent._find_michel_placeholders(draft) == [
        "topic-1-description",
        "topic-1-paper-1-result",
    ]


def test_finalization_removes_michel_markers_without_forcing_his_wording():
    """Verify editorial markers never leak into Julius's delivered draft."""
    draft = (
        'Main result. [[MICHEL_PEDAGOGY id="topic-1-paper-1-result" needed="yes"]]\n'
        'Second result. [[MICHEL_PEDAGOGY id="topic-1-paper-2-result" needed="no"]]'
    )

    finalized = julius_agent._remove_michel_placeholders(draft)

    assert finalized == "Main result. \nSecond result. "
    assert "MICHEL_PEDAGOGY" not in finalized


def test_finalization_requires_michels_full_response_not_a_feedback_excerpt():
    """Verify a Michel feedback fragment is rejected before finalization.

    Returns:
        None: Asserts Julius requires Michel's assessment and rationale as
        well as the location-keyed pedagogical explanations.
    """
    with pytest.raises(RuntimeError, match="full response; missing fields"):
        julius_agent._parse_complete_michel_response(
            json.dumps({"pedagogical_explanations": []})
        )


def test_julius_prompt_requires_main_results_and_allows_editorial_explanations():
    """Verify the final one-pager keeps factual results above optional prose.

    Returns:
        None: Asserts the coordinator requires main results and lets Julius
        decide whether an explanation helps.
    """
    instructions = julius_agent.build_julius_agent().instructions

    assert "Every representative paper must include its factual main result" in instructions
    assert "Julius decides whether to apply, rephrase, or reject each suggestion" in instructions


def test_strict_json_request_uses_a_json_schema_response_format():
    """Verify editorial tools enforce JSON at the Responses API boundary.

    Returns:
        None: Asserts the helper requests strict JSON-schema output and returns
        the valid JSON body supplied by the API.
    """
    captured = {}

    class FakeResponses:
        """Capture a structured-output request without calling the API."""

        def create(self, **kwargs):
            """Store request arguments and return a JSON response."""
            captured.update(kwargs)
            return SimpleNamespace(output_text='{"status": "compiled"}')

    class FakeClient:
        """Expose the subset of the OpenAI client used by the helper."""

        responses = FakeResponses()

    result = julius_agent._request_strict_json_response(
        client=FakeClient(),
        model="test-model",
        prompt="Return a JSON object.",
        schema_name="test_editorial_schema",
        schema={"type": "object", "additionalProperties": False, "properties": {}, "required": []},
    )

    assert result == '{"status": "compiled"}'
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["strict"] is True
    assert captured["text"]["format"]["name"] == "test_editorial_schema"


def test_michel_agent_tool_output_extractor_serializes_a_json_object():
    """Verify Julius receives Michel feedback as parseable JSON text.

    Returns:
        None: Asserts the nested-agent output adapter serializes structured
        Michel output rather than relying on a free-form string conversion.
    """
    result = asyncio.run(
        julius_agent._extract_michel_agent_tool_output(
            SimpleNamespace(final_output={"satisfactory": True, "feedback": []})
        )
    )

    assert json.loads(result) == {"satisfactory": True, "feedback": []}


def test_finalization_error_explains_that_the_draft_was_not_changed():
    """Verify invalid finalization output has a safe user-facing response.

    Returns:
        None: Asserts the unchanged draft and a clear retry explanation are
        returned instead of an opaque JSON parsing exception.
    """
    draft = "Current one-pager draft"
    result = julius_agent._finalization_error(
        draft,
        "general audience",
        "professional",
        RuntimeError("finalize_editorial_one_pager_tool did not return valid JSON"),
    )

    assert result["status"] == "error"
    assert result["content"] == draft
    assert "invalid structured response" in result["user_message"]
    assert "not been changed" in result["user_message"]
