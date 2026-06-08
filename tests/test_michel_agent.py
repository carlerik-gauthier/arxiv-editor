"""Unit tests for Phase 5 MichelAgent helpers."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NEW = ROOT / "src_new"
if str(SRC_NEW) not in sys.path:
    sys.path.insert(0, str(SRC_NEW))

import michel_agent as phase5


def _wrapped_make_clearer_tool():
    return phase5.make_clearer_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents


def _wrapped_provide_intuition_tool():
    return phase5.provide_intuition_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents


def _wrapped_metaphor_tool():
    return phase5.metaphor_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents


def _wrapped_assess_non_expert_satisfaction_tool():
    return phase5.assess_non_expert_satisfaction_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents


def test_make_clearer_tool_uses_fallback_without_api_key(monkeypatch):
    monkeypatch.setattr(phase5.os, "getenv", lambda key, default=None: None if key == "OPENAI_API_KEY" else default)
    result = _wrapped_make_clearer_tool()("Technical martingale convergence statement.")

    assert result["status"] == "success"
    assert "main idea" in result["clearer_text"]
    assert len(result["simplifications"]) >= 1


def test_provide_intuition_tool_returns_examples_without_api_key(monkeypatch):
    monkeypatch.setattr(phase5.os, "getenv", lambda key, default=None: None if key == "OPENAI_API_KEY" else default)
    result = _wrapped_provide_intuition_tool()("mixing time", explanation="A convergence rate for Markov chains.")

    assert result["status"] == "success"
    assert "mixing time" in result["intuition"]
    assert result["examples"]


def test_metaphor_tool_returns_metaphor_without_api_key(monkeypatch):
    monkeypatch.setattr(phase5.os, "getenv", lambda key, default=None: None if key == "OPENAI_API_KEY" else default)
    result = _wrapped_metaphor_tool()("sigma-algebra")

    assert result["status"] == "success"
    assert "map" in result["metaphor"]


def test_assess_non_expert_satisfaction_tool_flags_missing_elements(monkeypatch):
    monkeypatch.setattr(phase5.os, "getenv", lambda key, default=None: None if key == "OPENAI_API_KEY" else default)
    result = _wrapped_assess_non_expert_satisfaction_tool()(
        explanation="This theorem gives a convergence result.",
        concept="martingale convergence",
    )

    assert result["status"] == "success"
    assert result["satisfactory"] is False
    assert result["missing_elements"]
    assert result["improvement_advice"]


def test_assess_non_expert_satisfaction_tool_accepts_complete_explanation(monkeypatch):
    monkeypatch.setattr(phase5.os, "getenv", lambda key, default=None: None if key == "OPENAI_API_KEY" else default)
    result = _wrapped_assess_non_expert_satisfaction_tool()(
        explanation=(
            "A martingale is like a fair game where your best prediction stays balanced over time. "
            "For example, if you keep averaging coin-flip outcomes, the theorem explains why the running behavior settles down."
        ),
        concept="martingale convergence",
    )

    assert result["status"] == "success"
    assert result["satisfactory"] is True
    assert result["missing_elements"] == []


def test_run_michel_agent_uses_runner_and_returns_tool_parameters(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResult:
        final_output = "Simplified explanation"

        class Item:
            raw_item = {
                "type": "function_call",
                "name": "make_clearer_tool",
                "arguments": '{"text":"technical content"}',
            }

        new_items = [Item()]

    def fake_runner(agent, input, max_turns):
        captured["agent_name"] = agent.name
        captured["input"] = input
        captured["max_turns"] = max_turns
        return FakeResult()

    monkeypatch.setattr(phase5, "build_michel_agent", lambda: type("A", (), {"name": "MichelAgent"})())
    monkeypatch.setattr(phase5.Runner, "run_sync", fake_runner)
    monkeypatch.setattr(phase5, "trace", lambda *args, **kwargs: nullcontext())

    result = phase5.run_michel_agent("Explain this theorem for beginners.")

    assert result["reply"] == "Simplified explanation"
    assert result["tool_parameters"] == [
        {
            "tool": "make_clearer_tool",
            "arguments": {"text": "technical content"},
        }
    ]
    assert captured["agent_name"] == "MichelAgent"
    assert captured["input"] == "Explain this theorem for beginners."
    assert captured["max_turns"] == phase5.DEFAULT_MAX_TURNS


def test_build_michel_agent_registers_assessment_tool(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(phase5, "Agent", FakeAgent)

    phase5.build_michel_agent()

    tool_names = [getattr(tool, "name", None) for tool in captured["tools"]]
    assert "assess_non_expert_satisfaction_tool" in tool_names
    assert "If the assessment says it is not satisfactory" in captured["instructions"]
