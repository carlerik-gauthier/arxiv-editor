"""Phase 4 JuliusAgent implementation with OpenAI Agents SDK."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from openai import OpenAI
from typing import Any, Dict, Iterable, List, Optional

from agents import Agent, Runner, function_tool, trace

from src_new.chris_agent import build_chris_agent
from src_new.michel_agent import build_michel_agent

EXPECTED_FORMAT_OUTPUT_RULE = """
    For every topic, the expected output structure is:
        # <TOPIC TITLE>: 
        <topic count> papers
        <topic description>
        <pedagogical explanation for topic description>
        **Representative papers**
        1. <paper title>, <paper arxiv_id>
        <Representative paper main results>
        <pedagogical explanation for reprensative main result> 
        2. <paper title>, <paper arxiv_id>
        <Representative paper main results>
        <pedagogical explanation for reprensative main result>
    Repeat as many times as there are representative papers.
    Mandatory elements are <TOPIC TITLE>, <topic count>, <topic description> or <pedagogical explanation for topic description>, <paper title> and <paper arxiv_id>.
    <pedagogical explanation for reprensative main result> should be provided if MichelAgent is called.
    <pedagogical explanation for topic description> should be provided if MichelAgent is called. If necessary it can replace <topic description> 
    They must be returned regardless of user request
"""

JULIUS_SYSTEM_PROMPT = (
    "Editor and coordinator role, responsible for planning, delegation and generating the one-pager. "
    "The one-pager must meet the user request, including tone. "
    "The one-pager must be engaging. You can use emojis or speech elevator techniques to make it appealing. "
    "You must remain professional. Unless stated otherwise by the user, the one-pager is aimed for a LinkedIn post. "
    "The post must contain between 1 and 5 topics. By default, unless stated otherwise, assume 3 topics.\n\n"
    "You own the editorial workflow:\n"
    "- Parse the user request, including date range, topics, and preferences.\n"
    "- Create a concise execution plan before writing the final one-pager.\n"
    "- Determine how many topics are needed from ChrisAgent.\n"
    "- Delegate probability/statistics content requests to ChrisAgent.\n"
    "- Delegate clarity, intuition, and metaphor work to MichelAgent when the audience is general or when the user asks for simpler explanations.\n"
    "- When you call ChrisAgent, make the request self-contained and include the date range, topic count, "
    "  and whether main results are required.\n"
    "- When you call MichelAgent, pass the exact concept or draft text that needs to be made clearer.\n"
    "- You must never clarity, intuition, and metaphor work by yourself\n"
    "- Coordinate parallel execution where possible, but do not claim parallelism if only one specialist is used.\n"
    "- Synthesize the delegated material into one coherent one-pager. Topic title, topic description, represensative papers are mandatory.\n"
    "- If the request is outside probability/statistics, reply politely that you do not have knowledge about it.\n"
)

DEFAULT_MAX_TOPICS = 5
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TURNS = 8

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

_SUPPORTED_PR_ST_KEYWORDS = (
    "probability",
    "probabilistic",
    "probabilities",
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
            "Probability/statistics specialist. Use it for requests related to probability and statistics, "
            "topic extraction and description, and representative-paper main results."
        ),
        max_turns=6,
    )
    michel_tool = build_michel_agent().as_tool(
        tool_name="michel_agent_tool",
        tool_description=(
            "General-audience explainer. Use it to simplify technical mathematics, add intuition, "
            "or create metaphors for non-experts."
        ),
        max_turns=6,
    )
    instructions = (
        f"{JULIUS_SYSTEM_PROMPT}\n"
        "Use `extract_date_range_tool` to find the date range requested by the user.\n"
        "Use `chris_agent_tool` to delegate probability/statistics work and collect topic titles, descriptions, "
        "representative papers, and main results when needed.\n"
        "Use `michel_agent_tool` to provide a general-audience explanation, vulgarization, intuition, "
        "examples, metaphors, or simpler phrasing.\n"
        "Use `editorial_one_pager_tool` to make the editorial work and build a first draft.\n"
        "Use `finalize_editorial_one_pager_tool` to finalize the one pager\n"
        "If the user did not specify an audience, assume LinkedIn.\n"
        "Always restate the execution plan briefly before the final one-pager.\n"
        "When calling `editorial_one_pager_tool`, provide the specialist outputs as a dictionary, the tone and targeted audience.\n"
        "When calling `finalize_editorial_one_pager_tool`, provide the first draft, MichelAgent suggestions, the tone and targeted audience.\n"
        "When delegating to ChrisAgent, include the date range, inferred categories, topic count, audience, tone, "
        "and whether main results are required.\n"
        "When delegating to MichelAgent, include the target audience and the exact topic text or result that needs simplification.\n"
        "If you delegated to MichelAgent, you must use `finalize_editorial_one_pager_tool` to finalize the one-pager with MichelAgent proposals\n"
        "MichelAgent and `finalize_editorial_one_pager_tool` can be called multiple times (no more than 3 times) to refine the one-pager"
        f"Never forget the expected format rule : {EXPECTED_FORMAT_OUTPUT_RULE}"
    )

    return Agent(
        name="JuliusAgent",
        instructions=instructions,
        tools=[
            extract_date_range_tool,
            chris_tool,
            michel_tool,
            editorial_one_pager_tool,
            finalize_editorial_one_pager_tool,
        ],
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    )


def run_julius_agent(
    message: str,
    conversation_history: Optional[Iterable[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run one JuliusAgent turn with SDK tracing enabled."""
    history = list(conversation_history or [])
    with trace(
        "phase4-julius-agent-run",
        metadata={
            "agent": "JuliusAgent",
            "has_history": 'True' if bool(history) else 'False',
        },
        disabled=os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "0") == "1",
    ):
        combined_context = _conversation_context(history, message)
        if not _is_probability_or_statistics_request(combined_context):
            return {
                "reply": (
                    "I can only coordinate probability or statistics requests for now. "
                    "Please reformulate the request around those domains."
                ),
                "tool_parameters": [],
            }

        # enriched_message = _enrich_message_for_michel(message, history)
        agent = build_julius_agent()
        result = Runner.run_sync(agent, message, max_turns=DEFAULT_MAX_TURNS)

    return {
        "reply": str(getattr(result, "final_output", "")),
        "tool_parameters": _extract_tool_parameters(getattr(result, "new_items", [])),
    }

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
    if any(keyword in normalized for keyword in _SUPPORTED_PR_ST_KEYWORDS):
        return True
    return _is_probability_or_statistics_request_with_llm(text)


def _is_probability_or_statistics_request_with_llm(text: str) -> bool:
    """Use a tiny LLM fallback when the keyword check is inconclusive."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    prompt = (
        "Decide whether the user text is about probability or statistics, including "
        "implicit or closely related requests.\n"
        "Return exactly YES or NO.\n"
        f"Text: {text}"
    )
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
        )
    except Exception:
        return False

    content = (response.output_text or "").strip().casefold()
    return content == "yes"

@function_tool(name_override="editorial_one_pager_tool")
def editorial_one_pager_tool(
    specialized_agent_input: List[Any],
    title: str = "ArXiv Research Brief",
    audience: str = "LinkedIn",
    tone: str = "professional",
) -> Dict[str, Any]:
    """
    Synthesize specialist outputs into a one-pager draft.

    This tool is responsible for adapting the tone and structure to the target
    audience and for turning specialist results into a coherent editorial brief.
    Example = specialized_agent_input = [{'ChrisAgent': <output>}, {'MichelAgent': <output>}]
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for editorial_one_pager_tool")

    payload = _normalize_editorial_handoff(specialized_agent_input)

    prompt = (
        f"You are an editorial assistant for a one-pager called {title}.\n"
        "Use the specialist handoff to select the best topics, decide whether general-audience "
        "explanation, vulgarization, intuition, or metaphors are needed, and write a coherent first draft.\n"
        "Return JSON only with keys: status, title, audience, tone, topic_count, selected_topics, "
        "omitted_topics, needs_michel, clarity_review, editorial_summary, content.\n"
        "Do not invent unsupported facts.\n"
        "Use concise editorial prose.\n"
        f"Target Audience is {audience} and the one-pager tone is expected to be {tone}\n"
        f"The title is {title}\n"
        f"Specialist handoff JSON:\n{json.dumps(payload, ensure_ascii=False, default=str, indent=2)}"
    )

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.2,
        )
    except Exception as exc:
        raise RuntimeError(f"editorial_one_pager_tool failed: {exc}") from exc

    content = (response.output_text or "").strip()
    if not content:
        raise RuntimeError("editorial_one_pager_tool returned an empty response")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("editorial_one_pager_tool did not return valid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("editorial_one_pager_tool must return a JSON object")

    result.setdefault("status", "compiled")
    result.setdefault("title", title)
    result.setdefault("audience", audience)
    result.setdefault("tone", tone)
    result.setdefault("selected_topics", [])
    result.setdefault("omitted_topics", [])
    result.setdefault("needs_michel", False)
    result.setdefault("clarity_review", {})
    result.setdefault("editorial_summary", {})
    result.setdefault("content", "")
    result["topic_count"] = len(result.get("selected_topics") or [])
    return result


def _normalize_editorial_handoff(specialized_agent_input: List[Any]) -> Dict[str, Any]:
    """Normalize the list-based specialist handoff into a single payload."""
    payload: Dict[str, Any] = {}
    for item in specialized_agent_input or []:
        if isinstance(item, dict):
            payload.update(item)
    return payload

@function_tool(name_override="finalize_editorial_one_pager_tool")
def finalize_editorial_one_pager_tool(
    one_pager: str,
    michel_agent_output: str,
    audience: str = "LinkedIn",
    tone: str = "professional",
) -> Dict[str, Any]:
    """Finalize a one-pager draft using MichelAgent's editorial suggestions."""
    prompt = (
        "You are finalizing an editorial one-pager.\n"
        "Review the draft and MichelAgent suggestions.\n"
        "Decide if the one-pager can be delivered or if more editorial work is needed.\n"
        "Return JSON only with keys: status, final_decision, reason, content, title, audience, tone, "
        "needs_further_revision, michel_assessment, editorial_summary.\n"
        "Use status=ready_to_deliver and final_decision=deliver only if the draft is clear enough for the "
        "target audience and MichelAgent feedback has been integrated.\n"
        f"Target audience: {audience}\n"
        f"Tone: {tone}\n"
        f"Draft:\n{one_pager}\n"
        f"MichelAgent output:\n{michel_agent_output}"
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for finalize_editorial_one_pager_tool")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.2,
        )
    except Exception as exc:
        raise RuntimeError(f"finalize_editorial_one_pager_tool failed: {exc}") from exc

    content = (response.output_text or "").strip()
    if not content:
        raise RuntimeError("finalize_editorial_one_pager_tool returned an empty response")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("finalize_editorial_one_pager_tool did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("finalize_editorial_one_pager_tool must return a JSON object")

    payload.setdefault("status", "needs_editorial_revision")
    payload.setdefault("final_decision", "revise")
    payload.setdefault("needs_further_revision", payload.get("status") != "ready_to_deliver")
    payload.setdefault("title", "ArXiv Research Brief")
    payload.setdefault("audience", audience)
    payload.setdefault("tone", tone)
    payload.setdefault("michel_assessment", {})
    payload.setdefault("editorial_summary", {})
    payload.setdefault("content", one_pager)
    return payload

@function_tool(name_override="extract_date_range_tool")
def extract_date_range_tool(message: str) -> Dict[str, Any]:
    """Tool wrapper that extracts an ISO date range from the user message."""
    start_date, end_date = _extract_date_range(message)
    return {
        "start_date": start_date,
        "end_date": end_date,
    }


def _extract_date_range(message: str) -> tuple[str, str]:
    extracted = _extract_date_range_with_llm(message)
    if extracted:
        start_raw = extracted.get("start_date")
        end_raw = extracted.get("end_date")
        try:
            start = _parse_iso_date(str(start_raw))
            end = _parse_iso_date(str(end_raw)) if end_raw else datetime.now()
        except (ValueError, TypeError):
            pass
        else:
            if start > end:
                start, end = end, start
            return start.date().isoformat(), end.date().isoformat()
    end = datetime.now()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start.date().isoformat(), end.date().isoformat()


def _parse_iso_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc

def _extract_date_range_with_llm(message: str) -> Dict[str, Optional[str]]:
    """Use an LLM to extract ISO date bounds from a user message."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    today = datetime.now().date()
    day_name = today.strftime("%A")
    month_name = today.strftime("%B")
    prompt = (
        "You are a specialist in finding dates of all format.\n"
        "Extract an explicit date range from the user message.\n"
        "Return JSON only with this exact schema: "
        "{\"start_date\":\"YYYY-MM-DD or null\",\"end_date\":\"YYYY-MM-DD or null\"}.\n"
        f"Today is {day_name}, {month_name} {today.day}, {today.year}.\n"
        "#Rule:#\n"
        f"- Read *very carefully* the message. \n"
        f"- if there is a relative date, then end_date is {today.isoformat()}\n"
        "- if there is not any date intent in the message, then return null for start_date and end_date\n"
        f"#User message:\n {message}"
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    content = (response.output_text or "").strip()
    if not content:
        return {
            "start_date": None,
            "end_date": None,
        }

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {
            "start_date": None,
            "end_date": None,
        }

    return {
        "start_date": payload.get("start_date"),
        "end_date": payload.get("end_date"),
    }


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
