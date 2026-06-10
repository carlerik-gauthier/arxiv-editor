"""Unit tests for Phase 4 JuliusAgent helpers."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NEW = ROOT / "src_new"
if str(SRC_NEW) not in sys.path:
    sys.path.insert(0, str(SRC_NEW))

import julius_agent as phase4


def test_is_probability_or_statistics_request_accepts_supported_queries():
    assert phase4._is_probability_or_statistics_request(
        "Summarize recent stochastic process and Bayesian inference papers."
    )


def test_is_probability_or_statistics_request_rejects_unsupported_queries():
    assert not phase4._is_probability_or_statistics_request(
        "Summarize recent algebraic geometry papers."
    )


def test_is_probability_or_statistics_request_uses_llm_fallback(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        output_text = "YES"

    class FakeResponses:
        def create(self, model, input, temperature):
            captured["model"] = model
            captured["input"] = input
            captured["temperature"] = temperature
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    def fake_getenv(key, default=None):
        values = {
            "OPENAI_API_KEY": "test-api-key",
            "OPENAI_MODEL": "test-model",
        }
        return values.get(key, default)

    monkeypatch.setattr(phase4, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(phase4.os, "getenv", fake_getenv)

    assert phase4._is_probability_or_statistics_request(
        "Discuss the hidden structure of the urn experiment."
    )
    assert captured["api_key"] == "test-api-key"
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0
    assert "YES or NO" in str(captured["input"])


def test_extract_date_range_tool_wraps_helper(monkeypatch):
    monkeypatch.setattr(phase4, "_extract_date_range", lambda _message: ("2026-01-01", "2026-01-31"))

    wrapped = phase4.extract_date_range_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents
    result = wrapped("any message")

    assert result == {"start_date": "2026-01-01", "end_date": "2026-01-31"}


def test_run_julius_agent_declines_out_of_scope_request():
    result = phase4.run_julius_agent("Cover recent algebra papers.")
    assert "only coordinate probability or statistics" in result["reply"]
    assert result["tool_parameters"] == []


def test_run_julius_agent_traces_out_of_scope_request(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTrace:
        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            captured["exited"] = True

    def fake_trace(name, metadata=None, disabled=False, **_kwargs):
        captured["name"] = name
        captured["metadata"] = metadata
        captured["disabled"] = disabled
        return FakeTrace()

    monkeypatch.setattr(phase4, "trace", fake_trace)
    monkeypatch.setattr(phase4, "_is_probability_or_statistics_request", lambda _text: False)

    result = phase4.run_julius_agent("Cover recent algebra papers.")

    assert "only coordinate probability or statistics" in result["reply"]
    assert result["tool_parameters"] == []
    assert captured["entered"] is True
    assert captured["exited"] is True
    assert captured["name"] == "phase4-julius-agent-run"
    assert captured["metadata"] == {"agent": "JuliusAgent", "has_history": "False"}


def test_run_julius_agent_uses_runner_and_returns_tool_parameters(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResult:
        final_output = "Final one-pager"

        class Item:
            raw_item = {
                "type": "function_call",
                "name": "chris_agent_tool",
                "arguments": '{"request":"two topics on Markov chains"}',
            }

        new_items = [Item()]

    def fake_runner(agent, input, max_turns):
        captured["agent_name"] = agent.name
        captured["input"] = input
        captured["max_turns"] = max_turns
        return FakeResult()

    monkeypatch.setattr(phase4, "build_julius_agent", lambda: type("A", (), {"name": "JuliusAgent"})())
    monkeypatch.setattr(phase4.Runner, "run_sync", fake_runner)
    monkeypatch.setattr(phase4, "trace", lambda *args, **kwargs: nullcontext())

    result = phase4.run_julius_agent(
        "Give me two topics on Markov chains.",
        conversation_history=[{"role": "user", "content": "Focus on probability papers from 2026-05-19 to 2026-05-21."}],
    )

    assert result["reply"] == "Final one-pager"
    assert result["tool_parameters"] == [
        {
            "tool": "chris_agent_tool",
            "arguments": {"request": "two topics on Markov chains"},
        }
    ]
    assert captured["agent_name"] == "JuliusAgent"
    assert captured["max_turns"] == phase4.DEFAULT_MAX_TURNS
    assert captured["input"] == "Give me two topics on Markov chains."


def test_build_julius_agent_registers_michel_tool(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSpecialist:
        def __init__(self, tool_name):
            self.tool_name = tool_name

        def as_tool(self, tool_name, tool_description, max_turns):
            return {
                "tool_name": tool_name,
                "tool_description": tool_description,
                "max_turns": max_turns,
            }

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(phase4, "build_chris_agent", lambda: FakeSpecialist("chris"))
    monkeypatch.setattr(phase4, "build_michel_agent", lambda: FakeSpecialist("michel"))
    monkeypatch.setattr(phase4, "Agent", FakeAgent)

    phase4.build_julius_agent()

    tool_names = [tool["tool_name"] for tool in captured["tools"] if isinstance(tool, dict)]
    sdk_tool_names = [getattr(tool, "name", None) for tool in captured["tools"]]
    assert "chris_agent_tool" in tool_names
    assert "michel_agent_tool" in tool_names
    assert "finalize_editorial_one_pager_tool" in sdk_tool_names
    assert "MichelAgent" in captured["instructions"]
