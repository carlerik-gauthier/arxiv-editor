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


class _SpanManager:
    """Context manager stand-in for SDK spans."""

    def __init__(self, span, exit_error=None):
        self.span = span
        self.exit_error = exit_error

    def __enter__(self):
        return self.span

    def __exit__(self, *_args):
        if self.exit_error:
            raise self.exit_error
        return False


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


def test_trace_generation_bounds_large_input_and_swallows_export_error(monkeypatch):
    """Oversized generation payloads are truncated and tracing close failures do not leak."""
    captured = {}

    class FakeSdk:
        def generation_span(self, **kwargs):
            captured["input"] = kwargs["input"]
            return _SpanManager(
                _TraceSpan("generation"),
                exit_error=RuntimeError("payload too large"),
            )

    monkeypatch.setattr(openai_tracing, "_load_tracing_sdk", lambda: FakeSdk())
    monkeypatch.setenv("OPENAI_AGENTS_TRACE_MAX_CHARS", "1000")
    payload = [{"role": "user", "content": "x" * 5000}]

    with openai_tracing.trace_generation(payload, model="test-model") as span:
        assert span.name == "generation"

    assert isinstance(captured["input"], list)
    assert captured["input"][0]["role"] == "user"
    assert len(captured["input"][0]["content"]) < 1200
    assert "<truncated" in captured["input"][0]["content"]


def test_trace_generation_wraps_non_message_input_into_message_array(monkeypatch):
    """Generation spans should always receive the SDK's expected array-of-objects shape."""
    captured = {}

    class FakeSdk:
        def generation_span(self, **kwargs):
            captured["input"] = kwargs["input"]
            return _SpanManager(_TraceSpan("generation"))

    monkeypatch.setattr(openai_tracing, "_load_tracing_sdk", lambda: FakeSdk())

    with openai_tracing.trace_generation("plain text prompt", model="test-model"):
        pass

    assert captured["input"] == [{"role": "user", "content": "plain text prompt"}]


def test_trace_custom_bounds_data_before_opening_span(monkeypatch):
    """Custom span data is serialized and bounded before it reaches the SDK."""
    captured = {}

    class FakeSdk:
        def custom_span(self, name, data=None):
            captured["name"] = name
            captured["data"] = data
            return _SpanManager(_TraceSpan(name))

    monkeypatch.setattr(openai_tracing, "_load_tracing_sdk", lambda: FakeSdk())
    monkeypatch.setenv("OPENAI_AGENTS_TRACE_MAX_CHARS", "1000")

    with openai_tracing.trace_custom("large-data", {"paper": "x" * 5000}):
        pass

    assert captured["name"] == "large-data"
    assert len(captured["data"]["paper"]) < 1200
    assert "<truncated" in captured["data"]["paper"]


def test_set_span_output_swallows_tracing_payload_errors():
    """A tracing setter failure should not break the caller."""
    class FailingOutputSpan:
        def set_output(self, _output):
            raise RuntimeError("payload too large")

    openai_tracing.set_span_output(FailingOutputSpan(), {"content": "x" * 5000})


def test_tracing_is_disabled_by_default(monkeypatch):
    """Local runs should not export traces unless explicitly enabled."""
    monkeypatch.delenv("OPENAI_AGENTS_ENABLE_TRACING", raising=False)
    monkeypatch.delenv("OPENAI_AGENTS_DISABLE_TRACING", raising=False)

    assert openai_tracing._is_tracing_enabled() is False


def test_disable_flag_overrides_enable_flag(monkeypatch):
    """An explicit disable flag should always win."""
    monkeypatch.setenv("OPENAI_AGENTS_ENABLE_TRACING", "true")
    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "true")

    assert openai_tracing._is_tracing_enabled() is False
