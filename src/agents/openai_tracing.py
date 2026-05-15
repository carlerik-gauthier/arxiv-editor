"""Optional OpenAI Agents SDK tracing helpers for the local agent framework."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_MAX_TRACE_CHARS = 8000
_TRACING_ENABLE_ENV_VAR = "OPENAI_AGENTS_ENABLE_TRACING"


@dataclass(frozen=True)
class _TracingSdk:
    trace: Callable[..., Any]
    get_current_trace: Callable[[], Any]
    agent_span: Callable[..., Any]
    custom_span: Callable[..., Any]
    function_span: Callable[..., Any]
    generation_span: Callable[..., Any]
    handoff_span: Callable[..., Any]


_TRACING_SDK: Optional[_TracingSdk] = None
_TRACING_SDK_LOADED = False


def is_tracing_available() -> bool:
    """Return whether the OpenAI Agents SDK tracing API is importable."""
    return _load_tracing_sdk() is not None


def include_sensitive_trace_data() -> bool:
    """
    Return whether tool/model inputs and outputs should be included in traces.

    This follows the Agents SDK environment convention. Set
    `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=false` to record span structure
    without payloads.
    """
    value = os.getenv("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA")
    return value is None or value.strip().lower() not in _FALSE_VALUES


@contextmanager
def trace_workflow(name: str, metadata: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """
    Open an OpenAI trace for a workflow, or a custom span if a trace exists.

    The Agents SDK warns when a nested `trace()` is created, so nested workflow
    boundaries become custom spans inside the active trace.
    """
    sdk = _load_tracing_sdk()
    if sdk is None:
        with nullcontext() as span:
            yield span
        return

    if _get_current_trace(sdk) is None:
        with _safe_sdk_span(sdk.trace, name, metadata=_trace_metadata(metadata)) as trace:
            yield trace
    else:
        with _safe_sdk_span(sdk.custom_span, name, data=_trace_data(metadata)) as span:
            yield span


@contextmanager
def trace_agent(
    name: str,
    tools: Optional[list[str]] = None,
    handoffs: Optional[list[str]] = None,
) -> Iterator[Any]:
    """Open an OpenAI agent span when tracing is available."""
    sdk = _load_tracing_sdk()
    if sdk is None:
        with nullcontext() as span:
            yield span
        return

    with _safe_sdk_span(sdk.agent_span, name=name, tools=tools, handoffs=handoffs) as span:
        yield span


@contextmanager
def trace_tool_call(tool_name: str, parameters: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Open a function span for an agent tool call."""
    sdk = _load_tracing_sdk()
    if sdk is None:
        with nullcontext() as span:
            yield span
        return

    trace_input = serialize_for_trace(parameters) if include_sensitive_trace_data() else None
    with _ensure_trace(sdk, f"Tool call: {tool_name}", {"tool_name": tool_name}):
        with _safe_sdk_span(sdk.function_span, name=tool_name, input=trace_input) as span:
            yield span


@contextmanager
def trace_handoff(from_agent: str, to_agent: str) -> Iterator[Any]:
    """Open a handoff span for agent-to-agent delegation."""
    sdk = _load_tracing_sdk()
    if sdk is None:
        with nullcontext() as span:
            yield span
        return

    with _ensure_trace(
        sdk,
        f"Handoff: {from_agent} to {to_agent}",
        {"from_agent": from_agent, "to_agent": to_agent},
    ):
        with _safe_sdk_span(sdk.handoff_span, from_agent=from_agent, to_agent=to_agent) as span:
            yield span


@contextmanager
def trace_custom(name: str, data: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Open a custom span when tracing is available."""
    sdk = _load_tracing_sdk()
    if sdk is None:
        with nullcontext() as span:
            yield span
        return

    with _safe_sdk_span(sdk.custom_span, name, data=_trace_data(data)) as span:
        yield span


@contextmanager
def trace_generation(
    messages: Any,
    model: Optional[str] = None,
) -> Iterator[Any]:
    """Open a generation span around an injected LLM client call."""
    sdk = _load_tracing_sdk()
    if sdk is None:
        with nullcontext() as span:
            yield span
        return

    trace_input = _generation_input(messages) if include_sensitive_trace_data() else None
    with _safe_sdk_span(sdk.generation_span, input=trace_input, model=model) as span:
        yield span


def set_span_output(span: Any, output: Any) -> None:
    """Attach output to a span when the SDK span supports it."""
    if span is None or not include_sensitive_trace_data():
        return

    try:
        span_data = getattr(span, "span_data", None)
        if _is_generation_span_data(span_data):
            span_data.output = _generation_output(output)
        elif hasattr(span, "set_output"):
            span.set_output(serialize_for_trace(output))
        elif hasattr(span_data, "output"):
            span_data.output = serialize_for_trace(output)
        elif hasattr(span_data, "data") and isinstance(span_data.data, dict):
            span_data.data["output"] = serialize_for_trace(output)
    except Exception:
        logger.exception("OpenAI tracing failed while attaching span output; continuing.")


def set_span_error(span: Any, exc: Exception, data: Optional[Dict[str, Any]] = None) -> None:
    """Attach structured error details to a span when tracing is available."""
    if span is None or not hasattr(span, "set_error"):
        return

    include_sensitive = include_sensitive_trace_data()
    error_data: Dict[str, Any] = {"error_type": type(exc).__name__}
    if include_sensitive:
        error_data["details"] = serialize_for_trace(str(exc))
        if data:
            error_data.update(_trace_data(data))
    try:
        span.set_error(
            {
                "message": serialize_for_trace(str(exc)) if include_sensitive else type(exc).__name__,
                "data": error_data,
            }
        )
    except Exception:
        logger.exception("OpenAI tracing failed while attaching span error; continuing.")


def serialize_for_trace(value: Any) -> str:
    """Serialize and bound trace payloads so large papers do not flood traces."""
    try:
        serialized = json.dumps(value, default=str, sort_keys=True)
    except TypeError:
        serialized = str(value)

    return _truncate_trace_string(serialized)


def _truncate_trace_string(value: str) -> str:
    """Trim serialized trace payloads to the configured size limit."""
    max_chars = _max_trace_chars()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... <truncated {len(value) - max_chars} chars>"


def _generation_output(output: Any) -> list[Dict[str, Any]]:
    return [{"role": "assistant", "content": serialize_for_trace(output)}]


def _generation_input(messages: Any) -> Optional[list[Dict[str, Any]]]:
    normalized = _normalize_trace_value(messages)
    if isinstance(normalized, list) and all(isinstance(item, dict) for item in normalized):
        return normalized
    if isinstance(normalized, dict):
        return [normalized]
    if normalized is None:
        return None
    return [{"role": "user", "content": str(normalized)}]


def _is_generation_span_data(span_data: Any) -> bool:
    return type(span_data).__name__ == "GenerationSpanData"


def _normalize_trace_value(value: Any) -> Any:
    """Convert trace payloads into JSON-like values while preserving list/object shape."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_trace_string(value)
    if isinstance(value, dict):
        return {str(key): _normalize_trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_trace_value(item) for item in value]
    return _truncate_trace_string(str(value))


def _trace_data(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {}
    if include_sensitive_trace_data():
        return {
            str(key): serialize_for_trace(value)
            for key, value in data.items()
        }
    return {
        str(key): serialize_for_trace(value)
        for key, value in data.items()
        if str(key).startswith("trace_")
    }


def _max_trace_chars() -> int:
    raw_value = os.getenv("OPENAI_AGENTS_TRACE_MAX_CHARS")
    if not raw_value:
        return _DEFAULT_MAX_TRACE_CHARS
    try:
        return max(1000, int(raw_value))
    except ValueError:
        return _DEFAULT_MAX_TRACE_CHARS


@contextmanager
def _ensure_trace(
    sdk: _TracingSdk,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    if _get_current_trace(sdk) is None:
        with _safe_sdk_span(sdk.trace, name, metadata=_trace_metadata(metadata)):
            yield
    else:
        yield


def _get_current_trace(sdk: _TracingSdk) -> Any:
    """Read the current trace without letting tracing failures affect workflow."""
    try:
        return sdk.get_current_trace()
    except Exception:
        logger.exception("OpenAI tracing failed while reading current trace; continuing.")
        return None


@contextmanager
def _safe_sdk_span(span_factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Iterator[Any]:
    """Run an SDK span as best-effort instrumentation only."""
    try:
        manager = span_factory(*args, **kwargs)
    except Exception:
        logger.exception("OpenAI tracing failed while creating span; continuing.")
        with nullcontext() as span:
            yield span
        return

    try:
        span = manager.__enter__()
    except Exception:
        logger.exception("OpenAI tracing failed while entering span; continuing.")
        with nullcontext() as span:
            yield span
        return

    try:
        yield span
    except BaseException as body_exc:
        try:
            manager.__exit__(type(body_exc), body_exc, body_exc.__traceback__)
        except Exception:
            logger.exception("OpenAI tracing failed while closing span after error; continuing.")
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            logger.exception("OpenAI tracing failed while closing span; continuing.")


def _trace_metadata(data: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not data:
        return {}
    return {
        str(key): serialize_for_trace(value)
        for key, value in _trace_data(data).items()
    }


def _load_tracing_sdk() -> Optional[_TracingSdk]:
    global _TRACING_SDK, _TRACING_SDK_LOADED

    if not _is_tracing_enabled():
        return None

    if _TRACING_SDK_LOADED:
        return _TRACING_SDK

    _TRACING_SDK_LOADED = True
    try:
        from agents.tracing import (  # type: ignore[import-not-found]
            agent_span,
            custom_span,
            function_span,
            generation_span,
            get_current_trace,
            handoff_span,
            trace,
        )
    except ImportError:
        logger.debug("OpenAI Agents SDK tracing is not installed; skipping trace export.")
        return None

    _TRACING_SDK = _TracingSdk(
        trace=trace,
        get_current_trace=get_current_trace,
        agent_span=agent_span,
        custom_span=custom_span,
        function_span=function_span,
        generation_span=generation_span,
        handoff_span=handoff_span,
    )
    return _TRACING_SDK


def _is_tracing_enabled() -> bool:
    """Enable OpenAI trace export only when explicitly requested."""
    if os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "").strip().lower() in _TRUE_VALUES:
        return False

    value = os.getenv(_TRACING_ENABLE_ENV_VAR)
    if value is None:
        return False
    return value.strip().lower() in _TRUE_VALUES
