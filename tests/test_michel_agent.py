"""Unit tests for MichelAgent's deterministic fallback behavior."""

from src import michel_agent


def test_make_clearer_tool_falls_back_without_an_api_key(monkeypatch):
    """Verify the clearer-text tool returns a deterministic fallback.

    Args:
        monkeypatch: Pytest fixture used to remove the API key.

    Returns:
        None: Asserts the fallback success payload includes rewritten text.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = michel_agent.make_clearer_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents("Technical statement")
    assert result["status"] == "success"
    assert result["clearer_text"]


def test_pedagogical_explanation_tool_returns_json_compatible_fallback(monkeypatch):
    """Verify Michel's new explanation tool always returns structured data.

    Args:
        monkeypatch: Pytest fixture used to remove the API key.

    Returns:
        None: Asserts the offline fallback is JSON-compatible and includes a
        ready-to-insert pedagogical explanation.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    implementation = (
        michel_agent.get_pedagogical_explanation_tool
        .on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents
    )
    result = implementation("The estimator converges to the true value.")

    assert result["status"] == "success"
    assert result["exact_text"] == "The estimator converges to the true value."
    assert result["pedagogical_explanation"]


def test_non_expert_assessment_identifies_missing_context(monkeypatch):
    """Verify the offline assessment identifies explanation gaps.

    Args:
        monkeypatch: Pytest fixture used to remove the API key.

    Returns:
        None: Asserts a thin explanation is marked unsatisfactory with gaps.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = michel_agent.assess_non_expert_satisfaction_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents("A theorem converges.", concept="martingale")
    assert result["satisfactory"] is False
    assert result["missing_elements"]


def test_michel_prompt_includes_his_personality_and_comprehension_check():
    """Verify Michel's instructions retain required personality guidance.

    Returns:
        None: Asserts the prompt includes Michel's voice and comprehension cue.
    """
    instructions = michel_agent.build_michel_agent().instructions

    assert "upbeat, optimistic, curious, and energetic" in instructions
    assert "So far, so good?" in instructions


def test_michel_prompt_returns_ready_to_insert_feedback_for_julius():
    """Verify Michel's response contract supports Julius's revision loop.

    Returns:
        None: Asserts Michel assesses the complete draft and returns exact
        explanatory insertions without rewriting factual paper data.
    """
    instructions = michel_agent.build_michel_agent().instructions

    assert "complete draft is readable" in instructions
    assert "ready-to-insert text" in instructions
    assert "Never change paper titles, arXiv links, or technical claims" in instructions
    assert "pedagogical_explanations" in instructions


def test_michel_prompt_replaces_every_required_placeholder():
    """Verify Michel keys mandatory explanations to Julius's placeholders.

    Returns:
        None: Asserts Michel must provide exact text for each location marked
        as requiring pedagogy.
    """
    instructions = michel_agent.build_michel_agent().instructions

    assert "MICHEL_PEDAGOGY" in instructions
    assert "For every placeholder marked" in instructions
    assert "Each `location` must exactly match the `id`" in instructions


def test_michel_agent_uses_structured_output_and_the_pedagogy_tool():
    """Verify Michel's review and explanation generation are structured.

    Returns:
        None: Asserts the agent has a strict review output model and exposes
        the dedicated LLM-backed pedagogical-explanation tool.
    """
    agent = michel_agent.build_michel_agent()

    assert agent.output_type is michel_agent.MichelReviewOutput
    assert "get_pedagogical_explanation_tool" in [tool.name for tool in agent.tools]
    assert "copy the tool's returned" in agent.instructions
    assert "input_message" not in michel_agent.MichelReviewOutput.model_fields


def test_michel_prompt_applies_his_personality_to_required_explanations():
    """Verify required placeholder text is written in Michel's own voice.

    Returns:
        None: Asserts every explanation requested by a ``needed=\"yes\"``
        placeholder is explicitly subject to Michel's communication style.
    """
    instructions = michel_agent.build_michel_agent().instructions

    assert "Every `exact_text` written for a `needed=\"yes\"` placeholder" in instructions
    assert "upbeat, optimistic, curious, lively, concise, and engaging" in instructions
    assert "relatable example, anecdotal touch, or accurate metaphor" in instructions


def test_specialist_audience_fallback_does_not_require_a_metaphor(monkeypatch):
    """Verify Michel calibrates deterministic feedback to audience expertise.

    Args:
        monkeypatch: Pytest fixture used to remove the API key.

    Returns:
        None: Asserts a sufficiently detailed specialist draft is accepted
        without a general-audience analogy requirement.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = michel_agent._fallback_non_expert_assessment(
        explanation="This detailed explanation describes the theorem, its assumptions, proof strategy, and implications "
        "for the stated mathematical setting without relying on a classroom example or an informal comparison.",
        audience="graduate mathematics researchers",
        concept="theorem",
    )

    assert result["satisfactory"] is True
