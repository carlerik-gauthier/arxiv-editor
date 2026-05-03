"""Tests for optional OpenAI trace instrumentation hooks."""

from contextlib import contextmanager

from src.agents import openai_tracing
from src.agents.base_agent import AgentTool, BaseAgent
from src.agents.julius_agent import AgentHandoff, HandoffContext
from src.agents.specialized_agents import MichelAgent


class _TraceSpan:
    """Tiny stand-in for an OpenAI tracing span."""

    def __init__(self, name):
        self.name = name


def test_execute_tool_records_openai_function_span(monkeypatch):
    """BaseAgent tool execution opens a function span and attaches output."""
    events = []

    @contextmanager
    def fake_trace_tool_call(tool_name, parameters):
        span = _TraceSpan(tool_name)
        events.append(("tool_start", tool_name, parameters))
        yield span
        events.append(("tool_end", tool_name))

    monkeypatch.setattr(openai_tracing, "trace_tool_call", fake_trace_tool_call)
    monkeypatch.setattr(
        openai_tracing,
        "set_span_output",
        lambda span, output: events.append(("output", span.name, output)),
    )

    agent = BaseAgent(
        name="Tester",
        expertise="Testing",
        tools=[
            AgentTool(
                name="add",
                description="Add two numbers",
                function=lambda a, b: a + b,
                required_parameters=["a", "b"],
            )
        ],
    )

    result = agent.execute_tool("add", {"a": 2, "b": 3})

    assert result.success is True
    assert ("tool_start", "add", {"agent": "Tester", "parameters": {"a": 2, "b": 3}}) in events
    assert ("output", "add", 5) in events
    assert ("tool_end", "add") in events


def test_agent_handoff_records_openai_handoff_span(monkeypatch):
    """Agent handoffs open a handoff span around specialist execution."""
    events = []
    julius = BaseAgent(name="Julius", expertise="Coordination")
    michel = MichelAgent()

    @contextmanager
    def fake_trace_handoff(from_agent, to_agent):
        span = _TraceSpan(f"{from_agent}->{to_agent}")
        events.append(("handoff_start", from_agent, to_agent))
        yield span
        events.append(("handoff_end", from_agent, to_agent))

    @contextmanager
    def fake_trace_custom(name, data=None):
        events.append(("custom", name, data))
        yield _TraceSpan(name)

    monkeypatch.setattr(openai_tracing, "trace_handoff", fake_trace_handoff)
    monkeypatch.setattr(openai_tracing, "trace_custom", fake_trace_custom)

    def fake_set_span_output(span, output):
        if span is not None:
            events.append(("output", span.name, output["agent"]))

    monkeypatch.setattr(openai_tracing, "set_span_output", fake_set_span_output)

    handoff = AgentHandoff.execute_handoff(
        from_agent=julius,
        to_agent=michel,
        context=HandoffContext(task_description="Explain curvature."),
    )

    assert handoff.status.value == "COMPLETED"
    assert ("handoff_start", "Julius", "Michel") in events
    assert ("output", "Julius->Michel", "Michel") in events
    assert ("handoff_end", "Julius", "Michel") in events
