"""Unit tests for MichelAgent's deterministic fallback behavior."""

from src import michel_agent


def test_make_clearer_tool_falls_back_without_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = michel_agent.make_clearer_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents("Technical statement")
    assert result["status"] == "success"
    assert result["clearer_text"]


def test_non_expert_assessment_identifies_missing_context(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = michel_agent.assess_non_expert_satisfaction_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents("A theorem converges.", concept="martingale")
    assert result["satisfactory"] is False
    assert result["missing_elements"]


def test_michel_prompt_includes_his_personality_and_comprehension_check():
    instructions = michel_agent.build_michel_agent().instructions

    assert "upbeat, optimistic, curious, and energetic" in instructions
    assert "So far, so good?" in instructions
