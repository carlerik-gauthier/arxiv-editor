"""
Base AI agent abstraction and tool execution framework.

The classes in this module are intentionally provider-agnostic. A concrete LLM
client can be injected as long as it exposes one of the common completion
methods handled by BaseAgent.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.agents import openai_tracing

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when a tool cannot be executed successfully."""


@dataclass
class ToolCall:
    """Normalized representation of a tool call requested by an LLM."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result returned by the safe tool execution framework."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the tool result for conversation history or logs."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class AgentTool:
    """Callable tool registered with an agent."""

    name: str
    description: str
    function: Callable[..., Any]
    required_parameters: List[str] = field(default_factory=list)

    def execute(self, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Validate and execute the tool function."""
        parameters = parameters or {}
        missing = [
            parameter
            for parameter in self.required_parameters
            if parameter not in parameters or parameters[parameter] is None
        ]
        if missing:
            raise ToolExecutionError(
                f"Missing required parameter(s) for {self.name}: {', '.join(missing)}"
            )

        return self.function(**parameters)

    def to_schema(self) -> Dict[str, Any]:
        """Return a simple schema suitable for prompt construction."""
        return {
            "name": self.name,
            "description": self.description,
            "required_parameters": self.required_parameters,
        }


class BaseAgent:
    """
    Reusable agent abstraction with provider-agnostic LLM and tool support.

    Args:
        name: Human-readable agent name.
        expertise: Domain description used in the system prompt.
        categories: ArXiv categories assigned to the agent.
        llm_client: Optional LLM client for natural language reasoning.
        system_prompt: Optional role prompt. A default prompt is generated if omitted.
        tools: Tools available to the agent.
    """

    def __init__(
        self,
        name: str,
        expertise: str,
        categories: Optional[List[str]] = None,
        llm_client: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Iterable[AgentTool]] = None,
    ) -> None:
        if not name:
            raise ValueError("Agent name cannot be empty")
        if not expertise:
            raise ValueError("Agent expertise cannot be empty")

        self.name = name
        self.expertise = expertise
        self.categories = categories or []
        self.llm_client = llm_client
        self.system_prompt = system_prompt or self._build_default_system_prompt()
        self.tools: Dict[str, AgentTool] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {
            "tool_calls": [],
            "last_response": None,
            "last_error": None,
        }

        for tool in tools or []:
            self.register_tool(tool)

        self._add_message("system", self.system_prompt)

    def _build_default_system_prompt(self) -> str:
        """Create the default role prompt for the agent."""
        categories = ", ".join(self.categories) if self.categories else "no assigned categories"
        return (
            f"You are {self.name}, an AI research agent. "
            f"Expertise: {self.expertise}. "
            f"Assigned ArXiv categories: {categories}. "
            "Use available tools when they are needed, and explain results clearly."
        )

    def _add_message(self, role: str, content: Any) -> None:
        """Append a message to conversation history."""
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def register_tool(self, tool: AgentTool) -> None:
        """Register a tool with the agent."""
        if not tool.name:
            raise ValueError("Tool name cannot be empty")
        self.tools[tool.name] = tool
        logger.debug("Registered tool '%s' for agent '%s'", tool.name, self.name)

    def list_tools(self) -> List[str]:
        """Return available tool names."""
        return sorted(self.tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return simple schemas for available tools."""
        return [tool.to_schema() for tool in self.tools.values()]

    def execute_tool(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Execute a registered tool safely with logging and state tracking.

        Tool failures are captured in the returned ToolResult instead of being
        raised, which lets agents continue a workflow after partial failures.
        """
        started_at = datetime.now(timezone.utc)
        result = ToolResult(tool_name=tool_name, success=False, started_at=started_at)

        with openai_tracing.trace_tool_call(
            tool_name,
            {"agent": self.name, "parameters": parameters or {}},
        ) as trace_span:
            try:
                tool = self.tools.get(tool_name)
                if tool is None:
                    raise ToolExecutionError(f"Tool '{tool_name}' is not available to {self.name}")

                logger.info("Agent '%s' executing tool '%s'", self.name, tool_name)
                result.result = tool.execute(parameters)
                result.success = True
                openai_tracing.set_span_output(trace_span, result.result)

            except Exception as exc:
                result.error = str(exc)
                self.state["last_error"] = result.error
                openai_tracing.set_span_error(
                    trace_span,
                    exc,
                    {"agent": self.name, "tool_name": tool_name},
                )
                logger.exception("Agent '%s' failed executing tool '%s'", self.name, tool_name)

            finally:
                result.completed_at = datetime.now(timezone.utc)
                self.state["tool_calls"].append(result.to_dict())
                self._add_message("tool", result.to_dict())

        return result

    def execute_tool_calls(self, tool_calls: Iterable[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls in order."""
        return [self.execute_tool(call.name, call.parameters) for call in tool_calls]

    def respond(
        self,
        user_message: str,
        auto_execute_tools: bool = True,
    ) -> Dict[str, Any]:
        """
        Ask the LLM client for a response and optionally execute requested tools.

        The response format is intentionally simple so later specialized agents
        can build richer workflows without changing the base contract.
        """
        workflow_metadata = {
            "trace_agent": self.name,
            "auto_execute_tools": auto_execute_tools,
            "user_message": user_message,
        }
        with openai_tracing.trace_workflow(
            f"{self.name} agent response",
            metadata=workflow_metadata,
        ):
            with openai_tracing.trace_agent(
                self.name,
                tools=self.list_tools(),
            ) as trace_span:
                self._add_message("user", user_message)
                llm_response = self._call_llm(user_message)
                self.state["last_response"] = llm_response
                self._add_message("assistant", llm_response)

                tool_calls = self.parse_tool_calls(llm_response)
                tool_results: List[ToolResult] = []
                if auto_execute_tools and tool_calls:
                    tool_results = self.execute_tool_calls(tool_calls)

                response_payload = {
                    "agent": self.name,
                    "response": llm_response,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                }
                openai_tracing.set_span_output(trace_span, response_payload)
                return response_payload

    def _call_llm(self, user_message: str) -> Any:
        """
        Call an injected LLM client using common provider/client conventions.

        If no client is configured, the agent returns a deterministic response.
        This keeps unit tests and early project phases independent from paid APIs.
        """
        messages = self.conversation_history
        tool_schemas = self.get_tool_schemas()

        with openai_tracing.trace_generation(
            messages=messages,
            model=_infer_llm_model_name(self.llm_client),
        ) as trace_span:
            if self.llm_client is None:
                response = (
                    f"{self.name} received the task and is ready to use tools. "
                    f"Available tools: {', '.join(self.list_tools()) or 'none'}."
                )
                openai_tracing.set_span_output(trace_span, response)
                return response

            if callable(self.llm_client):
                response = self.llm_client(
                    messages=messages,
                    system_prompt=self.system_prompt,
                    tools=tool_schemas,
                )
                openai_tracing.set_span_output(trace_span, response)
                return response

            for method_name in ("complete", "generate", "chat", "invoke"):
                if method_name not in dir(self.llm_client):
                    continue
                method = getattr(self.llm_client, method_name, None)
                if callable(method):
                    try:
                        response = method(
                            messages=messages,
                            system_prompt=self.system_prompt,
                            tools=tool_schemas,
                        )
                    except TypeError:
                        response = method(user_message)
                    openai_tracing.set_span_output(trace_span, response)
                    return response

        raise TypeError("llm_client must be callable or expose complete/generate/chat/invoke")

    @staticmethod
    def parse_tool_calls(llm_response: Any) -> List[ToolCall]:
        """
        Extract tool calls from common LLM response shapes.

        Supported shapes include:
        - {"tool_calls": [{"name": "...", "parameters": {...}}]}
        - {"tool": "...", "parameters": {...}}
        - JSON strings or fenced JSON with either of the above
        - Provider-like objects with a tool_calls attribute
        """
        if llm_response is None:
            return []

        if hasattr(llm_response, "tool_calls"):
            return BaseAgent._normalize_tool_calls(getattr(llm_response, "tool_calls"))

        if isinstance(llm_response, dict):
            return BaseAgent._extract_tool_calls_from_dict(llm_response)

        if isinstance(llm_response, list):
            return BaseAgent._normalize_tool_calls(llm_response)

        if not isinstance(llm_response, str):
            return []

        parsed_payloads = BaseAgent._parse_json_payloads(llm_response)
        for payload in parsed_payloads:
            calls = BaseAgent.parse_tool_calls(payload)
            if calls:
                return calls

        return []

    @staticmethod
    def _extract_tool_calls_from_dict(payload: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from a dictionary payload."""
        if "tool_calls" in payload:
            return BaseAgent._normalize_tool_calls(payload["tool_calls"])

        if "tool_call" in payload:
            return BaseAgent._normalize_tool_calls([payload["tool_call"]])

        if "tool" in payload or "tool_name" in payload or "name" in payload:
            name = payload.get("tool") or payload.get("tool_name") or payload.get("name")
            parameters = payload.get("parameters") or payload.get("args") or {}
            if name:
                return [ToolCall(name=str(name), parameters=dict(parameters))]

        return []

    @staticmethod
    def _normalize_tool_calls(raw_calls: Any) -> List[ToolCall]:
        """Convert provider-specific tool call objects into ToolCall instances."""
        if raw_calls is None:
            return []
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]
        elif not isinstance(raw_calls, (list, tuple, set)):
            raw_calls = [raw_calls]

        normalized: List[ToolCall] = []
        for raw_call in raw_calls:
            if isinstance(raw_call, ToolCall):
                normalized.append(raw_call)
                continue

            if isinstance(raw_call, dict):
                name = (
                    raw_call.get("name")
                    or raw_call.get("tool")
                    or raw_call.get("tool_name")
                    or raw_call.get("function", {}).get("name")
                )
                parameters = (
                    raw_call.get("parameters")
                    or raw_call.get("args")
                    or raw_call.get("arguments")
                    or raw_call.get("function", {}).get("arguments")
                    or {}
                )
            else:
                name = getattr(raw_call, "name", None) or getattr(raw_call, "tool_name", None)
                parameters = (
                    getattr(raw_call, "parameters", None)
                    or getattr(raw_call, "args", None)
                    or getattr(raw_call, "arguments", None)
                    or {}
                )

            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    parameters = {"raw": parameters}

            if name:
                normalized.append(ToolCall(name=str(name), parameters=dict(parameters)))

        return normalized

    @staticmethod
    def _parse_json_payloads(text: str) -> List[Any]:
        """Parse JSON objects from plain or fenced response text."""
        candidates = [text.strip()]
        candidates.extend(
            match.strip()
            for match in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        )

        payloads: List[Any] = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                payloads.append(json.loads(candidate))
            except json.JSONDecodeError:
                continue

        return payloads


def _infer_llm_model_name(llm_client: Any) -> Optional[str]:
    """Best-effort model name extraction for tracing injected clients."""
    if llm_client is None:
        return "deterministic"

    for attribute_name in ("model", "model_name", "deployment", "engine"):
        value = getattr(llm_client, attribute_name, None)
        if value:
            return str(value)

    return type(llm_client).__name__
