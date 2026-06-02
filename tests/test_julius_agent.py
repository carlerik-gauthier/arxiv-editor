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


def test_infer_requested_topic_count_reads_digits():
    assert phase4._infer_requested_topic_count("Give me 9 topics.") == 5


def test_infer_requested_topic_count_reads_words_from_history():
    history = [{"role": "user", "content": "I need two topics for a LinkedIn post."}]
    assert phase4._infer_requested_topic_count("Use the same scope.", history=history) == 2


def test_run_julius_agent_declines_out_of_scope_request():
    result = phase4.run_julius_agent("Cover recent algebra papers.")
    assert "only coordinate probability or statistics" in result["reply"]
    assert result["tool_parameters"] == []


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
    assert "2026-05-19 to 2026-05-21" in str(captured["input"])
    assert "Requested topic count hint: 2." in str(captured["input"])
