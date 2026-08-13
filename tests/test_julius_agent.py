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


def test_julius_prompt_requires_a_michel_review_loop_after_a_factual_draft():
    """Verify Julius delegates all pedagogy and rechecks Michel's revisions.

    Returns:
        None: Asserts the prescribed draft, review, incorporation, and
        follow-up-review sequence is present in the coordinator instructions.
    """
    instructions = julius_agent.build_julius_agent().instructions

    first_draft = instructions.index("first draft must contain no pedagogical explanation")
    initial_review = instructions.index("After the first draft, always call `michel_agent_tool`")
    incorporation = instructions.index("replace `needed=\"yes\"` placeholders")
    follow_up_review = instructions.index("After every call to `finalize_editorial_one_pager_tool`")

    assert first_draft < initial_review < incorporation < follow_up_review
    assert "MichelAgent is the only source of pedagogical explanations" in instructions
    assert "no more than three Michel reviews in total" in instructions


def test_julius_prompt_uses_explicit_michel_placeholders_for_first_drafts():
    """Verify a first draft records where Michel must add pedagogy.

    Returns:
        None: Asserts the coordinator labels each pedagogical location and
        requires Michel text for labels marked as needed.
    """
    instructions = julius_agent.build_julius_agent().instructions

    assert "MICHEL_PEDAGOGY" in instructions
    assert 'needed="yes"` or `needed="no"' in instructions
    assert "replace `needed=\"yes\"` placeholders with MichelAgent's matching `exact_text` verbatim" in instructions


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


def test_michel_feedback_must_cover_each_required_placeholder():
    """Verify Julius cannot replace a required placeholder without Michel text.

    Returns:
        None: Asserts matching, non-empty Michel explanations are required for
        every location flagged with ``needed=\"yes\"``.
    """
    draft = '[[MICHEL_PEDAGOGY id="topic-1-description" needed="yes"]]'
    complete_response = json.dumps(
        {
            "satisfactory": False,
            "readability_reason": "A pedagogical explanation is required.",
            "feedback": ["Add the requested explanation."],
            "pedagogical_explanations": [
                {
                    "location": "topic-1-description",
                    "exact_text": "Michel's clear explanation.",
                }
            ]
        }
    )

    julius_agent._validate_michel_response_for_placeholders(draft, complete_response)

    with pytest.raises(RuntimeError, match="topic-1-description"):
        julius_agent._validate_michel_response_for_placeholders(
            draft,
            json.dumps(
                {
                    "satisfactory": False,
                    "readability_reason": "A pedagogical explanation is required.",
                    "feedback": ["Add the requested explanation."],
                    "pedagogical_explanations": [],
                }
            ),
        )


def test_required_michel_explanations_use_labeled_italic_markdown():
    """Verify finalization preserves factual text and labels Michel's prose.

    Returns:
        None: Asserts required explanations use the mandated italic wrapper
        directly after the factual result instead of replacing it.
    """
    draft = (
        'Factual main result.\n'
        '[[MICHEL_PEDAGOGY id="topic-1-paper-1-result" needed="yes"]]'
    )
    feedback = json.dumps(
        {
            "satisfactory": False,
            "readability_reason": "A pedagogical explanation is required.",
            "feedback": ["Add the requested explanation."],
            "pedagogical_explanations": [
                {
                    "location": "topic-1-paper-1-result",
                    "exact_text": "Think of it as a compass that points to the strongest signal.",
                }
            ]
        }
    )
    finalized = (
        "Factual main result.\n"
        "***Pedagogical explanation:** Think of it as a compass that points to the strongest signal.*"
    )

    julius_agent._validate_pedagogical_explanation_format(draft, feedback, finalized)

    with pytest.raises(RuntimeError, match="topic-1-paper-1-result"):
        julius_agent._validate_pedagogical_explanation_format(
            draft,
            feedback,
            "Factual main result.\nThink of it as a compass that points to the strongest signal.",
        )


def test_michel_exact_text_is_inserted_verbatim_without_editorial_changes():
    """Verify finalization does not paraphrase Michel's explanation text.

    Returns:
        None: Asserts the replacement is assembled directly from the exact
        ``exact_text`` field, including its punctuation and wording.
    """
    draft = 'Main result. [[MICHEL_PEDAGOGY id="topic-1-paper-1-result" needed="yes"]]'
    exact_text = "Think of it as a compass—simple, but it keeps you on course!"
    feedback = json.dumps(
        {
            "satisfactory": False,
            "readability_reason": "A pedagogical explanation is required.",
            "feedback": ["Add the requested explanation."],
            "pedagogical_explanations": [
                {"location": "topic-1-paper-1-result", "exact_text": exact_text}
            ]
        }
    )

    finalized = julius_agent._apply_michel_pedagogical_explanations(draft, feedback)

    assert finalized == f"Main result. ***Pedagogical explanation:** {exact_text}*"


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


def test_julius_prompt_requires_main_results_and_labeled_explanations():
    """Verify the final one-pager keeps factual results above Michel's prose.

    Returns:
        None: Asserts the coordinator requires main results and the exact
        Markdown treatment for pedagogical explanations.
    """
    instructions = julius_agent.build_julius_agent().instructions

    assert "Every representative paper must include its factual main result" in instructions
    assert "***Pedagogical explanation:** <MichelAgent text>*" in instructions


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
