"""Phase 6 JuliusAgent implementation with OpenAI Agents SDK."""

from __future__ import annotations

import json
import os
import re
import logging
from datetime import datetime, timedelta
from openai import OpenAI
from typing import Any, Dict, Iterable, List, Optional

from agents import Agent, Runner, function_tool, trace

from src_new.alain_agent import build_alain_agent
from src_new.chris_agent import build_chris_agent
from src_new.michel_agent import build_michel_agent

logger = logging.getLogger(__name__)

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
    "- You are responsible to allocate the number of topics to the different specialized agents. If an agent cannot return you requested, you pick another topic from another agent.\n"
    "- Use ChrisAgent for probability/statistics content and AlainAgent for algebra content.\n"
    "- Delegate clarity, intuition, and metaphor work to MichelAgent when the audience is general or when the user asks for simpler explanations.\n"
    "- When you call ChrisAgent or AlainAgent, make the request self-contained and include the date range, topic count, "
    "  and whether main results are required.\n"
    "- When you call MichelAgent, pass the exact concept or draft text that needs to be made clearer.\n"
    "- You must never clarity, intuition, and metaphor work by yourself\n"
    "- Coordinate parallel execution where possible, but do not claim parallelism if only one specialist is used.\n"
    "- Synthesize the delegated material into one coherent one-pager. Topic title, topic description, represensative papers are mandatory.\n"
    "- If the request is outside probability/statistics or algebra, reply politely that you do not have knowledge about it.\n"
)

DEFAULT_MAX_TOPICS = 5
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TURNS = 20

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

_SUPPORTED_ALGEBRA_KEYWORDS = (
    "algebra",
    "algebraic",
    "algebraic geometry",
    "algebraic topology",
    "commutative algebra",
    "ring",
    "rings",
    "algebra",
    "algebras",
    "ideal",
    "ideals",
    "module",
    "modules",
    "group",
    "groups",
    "group theory",
    "representation theory",
    "galois",
    "math.ag",
    "math.ra",
    "math.gr",
    "math.at"
)

_GENERIC_RESEARCH_KEYWORDS = (
    "paper",
    "papers",
    "topic",
    "topics",
    "arxiv",
    "research",
    "one-pager",
    "summary",
    "summarize",
)

_UNSUPPORTED_DOMAIN_KEYWORDS = (
    "geometry",
    "dynamical systems",
    "symplectic",
    "machine learning",
    "ml",
    "nlp",
    "llm",
    "language model",
    "cryptography",
    "data science",
    "artificial intelligence",
    "agentic ai",
)

_SUPPORTED_AGENT_ORDER = ("ChrisAgent", "AlainAgent")


def build_julius_agent() -> Agent:
    """Create JuliusAgent for phase 6."""
    chris_tool = build_chris_agent().as_tool(
        tool_name="chris_agent_tool",
        tool_description=(
            "Probability/statistics specialist. Use it for requests related to probability and statistics, "
            "topic extraction and description, and representative-paper main results."
        ),
        max_turns=6,
    )
    alain_tool = build_alain_agent().as_tool(
        tool_name="alain_agent_tool",
        tool_description=(
            "Algebra specialist. Use it for requests related to algebraic geometry, rings and algebras, "
            "group theory, topic extraction and description, and representative-paper main results."
        ),
        max_turns=6,
    )
    michel_tool = build_michel_agent().as_tool(
        tool_name="michel_agent_tool",
        tool_description=(
            "General-audience explainer. Use it to get feedback about how to simplify technical mathematics, add intuition, "
            "or create metaphors for non-experts."
        ),
        max_turns=6,
    )
    instructions = (
        f"{JULIUS_SYSTEM_PROMPT}\n"
        "Use `extract_date_range_tool` to find the date range requested by the user.\n"
        "Use `allocate_topics_tool` to decide how many topics each specialized agent should cover before delegating.\n"
        "Use `chris_agent_tool` to delegate probability/statistics work and collect topic titles, descriptions, "
        "representative papers, and main results when needed.\n"
        "Use `alain_agent_tool` to delegate algebra work and collect topic titles, descriptions, "
        "representative papers, and main results when needed.\n"
        "Use `michel_agent_tool` to get feedbacks about improvement to address non advanced experts by vulgarizing, providing intuitions and metaphors."
        "examples, metaphors, or simpler phrasing.\n"
        "Use `editorial_one_pager_tool` to make the editorial work and build a first draft.\n"
        "Use `finalize_editorial_one_pager_tool` to finalize the one pager\n"
        "Use `revise_one_pager_tool` to decide if the one pager needs further improvement\n"
        "If the user has not specified an audience, assume LinkedIn.\n"
        "If the user has not specified a tone, aussume professional.\n"
        "Always restate the execution plan briefly before the final one-pager.\n"
        "When calling `editorial_one_pager_tool`, provide the specialist outputs serialized as a JSON string, an appealing title, the target audience and the tone to use.\n"
        "When calling `finalize_editorial_one_pager_tool`, provide the first draft, MichelAgent feedback, the tone and targeted audience.\n"
        "When calling `allocate_topics_tool`, pass the full user request so the allocation can reflect the domain mix and requested number of topics.\n"
        "When delegating to ChrisAgent, include the date range, inferred categories, topic count, audience, tone, "
        "and whether main results are required.\n"
        "When delegating to AlainAgent, include the date range, inferred categories, topic count, audience, tone, "
        "and whether main results are required.\n"
        "When delegating to MichelAgent, include the target audience and the exact topic text or result that needs simplification.\n"
        "If you delegated to MichelAgent, you must use `finalize_editorial_one_pager_tool` to finalize the one-pager with MichelAgent's feedbacks\n"
        "MichelAgent's feedbacks must be considered if you call `michel_agent_tool`"
        "`michel_agent_tool` and `finalize_editorial_one_pager_tool` can be called multiple times (no more than 3 times) to refine the one-pager"
        "When calling `revise_one_pager_tool`, provide the latest one-pager draft, the expected tone and the target audience"
        f"Never forget the expected format rule : {EXPECTED_FORMAT_OUTPUT_RULE}"
    )

    return Agent(
        name="JuliusAgent",
        instructions=instructions,
        tools=[
            extract_date_range_tool,
            allocate_topics_tool,
            chris_tool,
            alain_tool,
            michel_tool,
            editorial_one_pager_tool,
            finalize_editorial_one_pager_tool,
            revise_one_pager_tool
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
        "phase6-julius-agent-run",
        metadata={
            "agent": "JuliusAgent",
            "has_history": 'True' if bool(history) else 'False',
        },
        disabled=os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "0") == "1",
    ):
        combined_context = _conversation_context(history, message)
        if not _is_supported_specialist_request(combined_context):
            return {
                "reply": (
                    "I can only coordinate probability, statistics, or algebra requests for now. "
                    "Please reformulate the request around those domains."
                ),
                "tool_parameters": [],
            }

        # enriched_message = _enrich_message_for_michel(message, history)
        agent = build_julius_agent()
        result = Runner.run_sync(agent, combined_context, max_turns=DEFAULT_MAX_TURNS)

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
    return f"Past messages: {history_text}\nNew user message: {message}"


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


def _is_supported_specialist_request(text: str) -> bool:
    """Return True when Julius can route the request to its current specialists."""
    normalized = text.casefold()
    if any(keyword in normalized for keyword in _SUPPORTED_PR_ST_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _SUPPORTED_ALGEBRA_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _GENERIC_RESEARCH_KEYWORDS):
        return _is_supported_specialist_request_with_llm(text)
    return _is_supported_specialist_request_with_llm(text)


def _is_supported_specialist_request_with_llm(text: str) -> bool:
    """Use an LLM fallback to decide whether Julius can serve the request."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        normalized = text.casefold()
        return any(keyword in normalized for keyword in _GENERIC_RESEARCH_KEYWORDS) and not any(
            keyword in normalized for keyword in _UNSUPPORTED_DOMAIN_KEYWORDS
        )

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    prompt = (
        "Decide whether the user text can be handled by the currently available specialists, including implicit or closely related requests.\n"
        "Supported domains: probability, statistics, algebra.\n"
        "Also answer YES for a general arXiv topic-summary request when no other domain is specified.\n"
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


@function_tool(name_override="allocate_topics_tool")
def allocate_topics_tool(
    message: str,
    available_agents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Allocate the requested number of topics across JuliusAgent's specialists."""
    agent_order = [agent for agent in (available_agents or list(_SUPPORTED_AGENT_ORDER)) if agent in _SUPPORTED_AGENT_ORDER]
    if not agent_order:
        agent_order = list(_SUPPORTED_AGENT_ORDER)

    payload = _allocate_topics_with_llm(message, agent_order)
    allocations = _normalize_allocation_payload(payload, agent_order)
    requested_topic_count = payload.get("requested_topic_count")
    if not isinstance(requested_topic_count, int):
        requested_topic_count = sum(item["topic_count"] for item in allocations)
    if requested_topic_count <= 0:
        requested_topic_count = DEFAULT_MAX_TOPICS

    return {
        "status": "success",
        "requested_topic_count": requested_topic_count,
        "allocations": allocations,
        "selection_reason": str(payload.get("selection_reason") or "").strip(),
        "fallback_order": list(agent_order),
    }


def _allocate_topics_with_llm(message: str, agent_order: List[str]) -> Dict[str, Any]:
    """Use an LLM to decide which specialists to call and how many topics each should cover."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for allocate_topics_tool")

    agent_descriptions = {
        "ChrisAgent": "probability and statistics specialist",
        "AlainAgent": "algebra specialist",
    }
    available_agents = [
        {
            "agent_name": agent_name,
            "specialty": agent_descriptions.get(agent_name, "specialized mathematics agent"),
        }
        for agent_name in agent_order
    ]
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    prompt = (
        "You are JuliusAgent's planning assistant.\n"
        "Your job is to decide which specialized agents to call and how many topics to request from each.\n"
        "Use only the available agents.\n"
        "The total requested topics must stay between 1 and 5.\n"
        "If the user names a supported domain explicitly, prefer the matching agent.\n"
        "If the user does not specify a supported domain, you may spread the topics across the available agents.\n"
        "The total number of topics you allocate must be greater or equal to the total number of topics required by the user.\n"
        "If you cannot find out the total number of topics required by the user, assume it is 5."
        "Return JSON only with this schema:\n"
        "{"
        "\"requested_topic_count\": <int>, "
        "\"selection_reason\": <string>, "
        "\"allocations\": ["
        "{\"agent_name\": <string>, \"topic_count\": <int>, \"reason\": <string>}"
        "]"
        "}.\n"
        f"Available agents: {json.dumps(available_agents, ensure_ascii=False)}\n"
        f"User request: {message}"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    content = (response.output_text or "").strip()
    if not content:
        raise RuntimeError("allocate_topics_tool returned an empty response")
    return _parse_json_object_response(content, "allocate_topics_tool")


def _normalize_allocation_payload(payload: Dict[str, Any], agent_order: List[str]) -> List[Dict[str, Any]]:
    """Normalize LLM allocation output into JuliusAgent's expected tool payload."""
    raw_allocations = payload.get("allocations") or []
    if not isinstance(raw_allocations, list):
        raise RuntimeError("allocate_topics_tool must return a list of allocations")

    normalized_allocations: List[Dict[str, Any]] = []
    seen_agents: set[str] = set()
    for item in raw_allocations:
        if not isinstance(item, dict):
            continue
        agent_name = str(item.get("agent_name") or "").strip()
        if agent_name not in agent_order or agent_name in seen_agents:
            continue
        topic_count = item.get("topic_count")
        if not isinstance(topic_count, int):
            continue
        topic_count = max(1, min(topic_count, DEFAULT_MAX_TOPICS))
        normalized_allocations.append(
            {
                "agent_name": agent_name,
                "tool_name": _agent_tool_name(agent_name),
                "topic_count": topic_count,
                "reason": str(item.get("reason") or _allocation_reason(agent_name)).strip(),
            }
        )
        seen_agents.add(agent_name)

    if not normalized_allocations:
        raise RuntimeError("allocate_topics_tool did not return usable allocations")
    return normalized_allocations


def _agent_tool_name(agent_name: str) -> str:
    """Return the JuliusAgent tool name associated with a specialist."""
    mapping = {
        "ChrisAgent": "chris_agent_tool",
        "AlainAgent": "alain_agent_tool",
    }
    return mapping[agent_name]


def _allocation_reason(agent_name: str) -> str:
    """Explain why a specialist received topic allocation. Default reason if a proper reason cannot be found"""
    if agent_name == "ChrisAgent":
        return "Interest in probability or statistics detected."
    if agent_name == "AlainAgent":
        return "Interest in algebra detected."
    return "Allocated by JuliusAgent."

@function_tool(name_override="editorial_one_pager_tool")
def editorial_one_pager_tool(
    specialized_agent_input: str,
    requested_topic_count: int,
    title: str="ArXiv Research Brief",
    audience: str = "LinkedIn",
    tone: str = "professional",
) -> Dict[str, Any]:
    """
    Synthesize specialist outputs into a one-pager draft.

    This tool is responsible for adapting the tone and structure to the target
    audience and for turning specialist results into a coherent editorial brief.
    Example = specialized_agent_input = '{"ChrisAgent": [...], "MichelAgent": {...}}'
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for editorial_one_pager_tool")

    payload = _normalize_editorial_handoff(specialized_agent_input)

    prompt = (
        f"You are an editorial assistant for a one-pager called {title}.\n"
        "Use the specialist handoff to select the best topics, decide whether general-audience "
        "explanation, vulgarization, intuition, or metaphors are needed, and write a coherent first draft.\n"
        "Return JSON only with keys: status, title, topic_count, "
        "editorial_summary, content.\n"
        "Do not invent unsupported facts.\n"
        "Use concise editorial prose.\n"
        f"You must return **exactly** {min(requested_topic_count, 5)} topics.\n"
        f"If you get less than {requested_topic_count} topics, return all of them\n"
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
    
    result = _parse_json_object_response(content, "editorial_one_pager_tool")

    result.setdefault("status", "compiled")
    result.setdefault("title", title)
    result.setdefault("selected_topics", [])
    result.setdefault("omitted_topics", [])
    result.setdefault("editorial_summary", {})
    result.setdefault("content", "")
    result.setdefault("topic_count", len(result.get("selected_topics") or []))
    return result


def _normalize_editorial_handoff(specialized_agent_input: Any) -> Dict[str, Any]:
    """Normalize specialist handoff data into one dictionary payload."""
    if isinstance(specialized_agent_input, str):
        raw = specialized_agent_input.strip()
        if not raw:
            return {}
        try:
            specialized_agent_input = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_input": raw}

    if isinstance(specialized_agent_input, dict):
        return specialized_agent_input

    payload: Dict[str, Any] = {}
    for item in specialized_agent_input or []:
        if isinstance(item, dict):
            payload.update(item)
    return payload


@function_tool(name_override="finalize_editorial_one_pager_tool")
def finalize_editorial_one_pager_tool(
    one_pager: str,
    michel_agent_feedback: str,
    audience: str = "LinkedIn",
    tone: str = "professional",
) -> Dict[str, Any]:
    """Finalize a one-pager draft using MichelAgent's editorial suggestions."""
    prompt = (
        "You are finalizing an editorial one-pager.\n"
        "Review the draft and take into account MichelAgent feedbacks.\n"
        f"Remember you addressing to a {audience} audience and your tone must be {tone}.\n"
        "Return JSON only with keys: status, final_decision, reason, content, title"
        "needs_further_revision, michel_assessment, editorial_summary.\n"
        "Use status=ready_to_deliver and final_decision=deliver only if the draft is clear enough for the "
        "target audience and MichelAgent feedback has been integrated.\n"
        f"Target audience: {audience}\n"
        f"Tone: {tone}\n"
        f"Latest Draft:\n{one_pager}\n"
        f"MichelAgent feedback:\n{michel_agent_feedback}"
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

    payload = _parse_json_object_response(content, "finalize_editorial_one_pager_tool")

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

@function_tool(name_override="revise_one_pager_tool")
def revise_one_pager_tool(
    one_pager: str,
    audience: str = "LinkedIn",
    tone: str = "professional",
) -> Dict[str, Any]:
    """Assess whether a one-pager fits the requested tone and audience."""
    prompt = (
        "You are reviewing a one-pager draft.\n"
        "Judge whether it matches the requested tone and audience.\n"
        "If it does not, decide whether the main issue is clarity, metaphor, or intuition.\n"
        "Return JSON only with keys: status, appropriate, reason, issue_type, recommendation.\n"
        "Use issue_type=none when the draft is appropriate.\n"
        "Use issue_type=clarity when the draft is too vague or hard to follow.\n"
        "Use issue_type=metaphor when the draft needs a metaphor to make the idea accessible.\n"
        "Use issue_type=intuition when the draft needs more intuitive explanation.\n"
        f"Target audience: {audience}\n"
        f"Tone: {tone}\n"
        f"Draft:\n{one_pager}"
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for revise_one_pager_tool")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.2,
        )
    except Exception as exc:
        raise RuntimeError(f"revise_one_pager_tool failed: {exc}") from exc

    content = (response.output_text or "").strip()
    if not content:
        raise RuntimeError("revise_one_pager_tool returned an empty response")

    payload = _parse_json_object_response(content, "revise_one_pager_tool")

    payload.setdefault("status", "ok")
    payload.setdefault("appropriate", True)
    payload.setdefault("reason", "")
    payload.setdefault("issue_type", "")
    payload.setdefault("recommendation", "")
    payload["audience"] = audience
    payload["tone"] = tone
    payload["one_pager"] = one_pager
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


def _parse_json_object_response(content: str, tool_name: str) -> Dict[str, Any]:
    """Parse a JSON object even when the model wraps it in fences or short prose."""
    candidates = [content.strip()]

    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(match.strip() for match in fenced_matches if match.strip())

    extracted_object = _extract_first_json_object(content)
    if extracted_object:
        candidates.append(extracted_object)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise RuntimeError(f"{tool_name} did not return valid JSON")


def _extract_first_json_object(content: str) -> Optional[str]:
    """Return the first balanced JSON object substring found in free-form text."""
    start = content.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(content)):
            char = content[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start:index + 1]
        start = content.find("{", start + 1)
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
