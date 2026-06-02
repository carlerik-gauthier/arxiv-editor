"""Phase 4 JuliusAgent implementation with OpenAI Agents SDK."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from agents import Agent, Runner, trace

from src_new.chris_agent import build_chris_agent


JULIUS_SYSTEM_PROMPT = (
    "Editor and coordinator role, responsible for planning, delegation and generating the one-pager. "
    "The one-pager must meet the user request, including tone. "
    "The one-pager must be engaging. You can use emojis or speech elevator techniques to make it appealing. "
    "You must remain professional. Unless stated otherwise by the user, the one-pager is aimed for a LinkedIn post. "
    "The post must contain between 1 and 5 topics.\n"
    "You own the editorial workflow:\n"
    "- Parse the user request, including date range, topics, and preferences.\n"
    "- Create a concise execution plan before writing the final one-pager.\n"
    "- Determine how many topics are needed from ChrisAgent.\n"
    "- Delegate probability/statistics content requests to ChrisAgent.\n"
    "- When you call ChrisAgent, make the request self-contained and include the date range, topic count, "
    "audience, tone, and whether main results are required.\n"
    "- Coordinate parallel execution where possible, but do not claim parallelism if only one specialist is used.\n"
    "- Synthesize the delegated material into one coherent one-pager.\n"
    "- If the request is outside probability/statistics, reply politely that you do not have knowledge about it.\n"
)

DEFAULT_MAX_TOPICS = 5
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TURNS = 8

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

_SUPPORTED_KEYWORDS = (
    "probability",
    "probabilistic",
    "statistics",
    "statistical",
    "stochastic",
    "random variable",
    "random process",
    "markov",
    "brownian",
    "martingale",
    "bayesian",
    "inference",
    "estimation",
    "hypothesis testing",
    "monte carlo",
    "regression",
    "sampling",
    "limit theorem",
    "math.pr",
    "math.st",
)


def build_julius_agent() -> Agent:
    """Create JuliusAgent for phase 4."""
    chris_tool = build_chris_agent().as_tool(
        tool_name="chris_agent_tool",
        tool_description=(
            "Probability/statistics specialist. Use it for math.PR and math.ST requests, "
            "topic extraction, and representative-paper main results."
        ),
        max_turns=6,
    )
    return Agent(
        name="JuliusAgent",
        instructions=JULIUS_SYSTEM_PROMPT,
        tools=[chris_tool],
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    )


def run_julius_agent(
    message: str,
    conversation_history: Optional[Iterable[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run one JuliusAgent turn with SDK tracing enabled."""
    history = list(conversation_history or [])
    combined_context = _conversation_context(history, message)
    if not _is_probability_or_statistics_request(combined_context):
        return {
            "reply": (
                "I can only coordinate probability or statistics requests for now. "
                "Please reformulate the request around those domains."
            ),
            "tool_parameters": [],
        }

    topic_count = _infer_requested_topic_count(message, history=history)
    enriched_message = _build_editorial_prompt(
        message=message,
        conversation_history=history,
        topic_count=topic_count,
    )
    agent = build_julius_agent()

    with trace(
        "phase4-julius-agent-run",
        metadata={
            "agent": "JuliusAgent",
            "requested_topic_count": topic_count,
            "has_history": bool(history),
        },
        disabled=os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "0") == "1",
    ):
        result = Runner.run_sync(agent, enriched_message, max_turns=DEFAULT_MAX_TURNS)

    return {
        "reply": str(getattr(result, "final_output", "")),
        "tool_parameters": _extract_tool_parameters(getattr(result, "new_items", [])),
    }


def _build_editorial_prompt(
    message: str,
    conversation_history: List[Dict[str, str]],
    topic_count: Optional[int],
) -> str:
    """Build a self-contained JuliusAgent input with session context."""
    history_text = _serialize_conversation(conversation_history)
    topic_line = (
        f"Requested topic count hint: {topic_count}."
        if topic_count is not None
        else "Requested topic count hint: infer it from the request and keep it between 1 and 5."
    )
    return (
        "You are handling the phase 4 editorial workflow.\n"
        f"{topic_line}\n"
        "If the user did not specify an audience, assume LinkedIn.\n"
        "Always restate the execution plan briefly before the final one-pager.\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Latest user request:\n{message}"
    )


def _conversation_context(
    conversation_history: List[Dict[str, str]],
    message: str,
) -> str:
    """Collapse the session into one text block for lightweight request checks."""
    history_text = _serialize_conversation(conversation_history)
    if not history_text:
        return message
    return f"{history_text}\nuser: {message}"


def _serialize_conversation(conversation_history: Iterable[Dict[str, str]]) -> str:
    """Serialize chat history into a compact plain-text transcript."""
    lines: List[str] = []
    for item in conversation_history:
        role = str(item.get("role", "user")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No previous conversation."


def _is_probability_or_statistics_request(text: str) -> bool:
    """Return True when the request stays within JuliusAgent's current scope."""
    normalized = text.casefold()
    return any(keyword in normalized for keyword in _SUPPORTED_KEYWORDS)


def _infer_requested_topic_count(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[int]:
    """Infer an explicit topic count from the latest message, then history."""
    inferred = _extract_topic_count_from_text(message)
    if inferred is not None:
        return inferred

    for item in reversed(history or []):
        inferred = _extract_topic_count_from_text(str(item.get("content", "")))
        if inferred is not None:
            return inferred
    return None


def _extract_topic_count_from_text(text: str) -> Optional[int]:
    """Extract a requested topic count and clamp it to JuliusAgent's limits."""
    lowered = text.casefold()
    digit_match = re.search(r"\b([1-9][0-9]*)\s+(?:main\s+)?topics?\b", lowered)
    if digit_match:
        return max(1, min(int(digit_match.group(1)), DEFAULT_MAX_TOPICS))

    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+(?:main\s+)?topics?\b", lowered):
            return value
    return None


def _extract_tool_parameters(new_items: List[Any]) -> List[Dict[str, Any]]:
    """Extract function-tool call arguments from SDK run items."""
    extracted: List[Dict[str, Any]] = []
    for item in new_items:
        raw_item = getattr(item, "raw_item", None)
        if raw_item is None:
            continue

        item_type = raw_item.get("type") if isinstance(raw_item, dict) else getattr(raw_item, "type", None)
        if item_type != "function_call":
            continue

        tool_name = raw_item.get("name") if isinstance(raw_item, dict) else getattr(raw_item, "name", None)
        arguments = raw_item.get("arguments") if isinstance(raw_item, dict) else getattr(raw_item, "arguments", None)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass

        extracted.append(
            {
                "tool": tool_name,
                "arguments": arguments,
            }
        )
    return extracted
