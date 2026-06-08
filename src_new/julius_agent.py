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
        "Use `michel_agent_tool` when the user wants a general-audience explanation, vulgarization, intuition, "
        "examples, metaphors, or simpler phrasing.\n"
        "Use `editorial_one_pager_tool` to turn the specialist output into the final one-pager.\n"
        "If the user did not specify an audience, assume LinkedIn.\n"
        "Always restate the execution plan briefly before the final one-pager.\n"
        "When delegating to ChrisAgent, include the date range, inferred categories, topic count, audience, tone, "
        "and whether main results are required.\n"
        "When delegating to MichelAgent, include the target audience and the exact topic text or result that needs simplification.\n"
    )

    return Agent(
        name="JuliusAgent",
        instructions=instructions,
        tools=[extract_date_range_tool, chris_tool, michel_tool, editorial_one_pager_tool],
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

        enriched_message = _enrich_message_for_michel(message, history)
        agent = build_julius_agent()
        result = Runner.run_sync(agent, enriched_message, max_turns=DEFAULT_MAX_TURNS)

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
    specialized_agent_input: str,
    title: str = "ArXiv Research Brief",
    audience: str = "LinkedIn",
    tone: str = "professional",
) -> Dict[str, Any]:
    """
    Synthesize specialist outputs into a one-pager draft.

    This tool is responsible for adapting the tone and structure to the target
    audience and for turning specialist results into a coherent editorial brief.
    """
    payload = _parse_editorial_payload(specialized_agent_input)
    audience = (payload.get("audience") or audience or "LinkedIn").strip() or "LinkedIn"
    tone = (payload.get("tone") or tone or "professional").strip() or "professional"
    title = (payload.get("title") or title or "ArXiv Research Brief").strip() or "ArXiv Research Brief"

    topics = _normalize_editorial_topics(payload)
    if not topics:
        return {
            "status": "needs_topics",
            "title": title,
            "audience": audience,
            "tone": tone,
            "topic_count": 0,
            "content": (
                f"# {title}\n\n"
                "No specialist topics were provided yet. Collect the delegate outputs "
                "before drafting the one-pager."
            ),
        }

    execution_plan = _normalize_execution_plan(payload.get("execution_plan"))
    date_range = _normalize_date_range(payload)
    intro = _build_editorial_intro(
        audience=audience,
        tone=tone,
        topic_count=len(topics),
        date_range=date_range,
    )
    content_parts = [f"# {title}", intro]

    if execution_plan:
        content_parts.append(
            "## Execution Plan\n" + "\n".join(f"- {step}" for step in execution_plan)
        )

    for index, topic in enumerate(topics, start=1):
        content_parts.append(_render_editorial_topic(topic, index))

    return {
        "status": "compiled",
        "title": title,
        "audience": audience,
        "tone": tone,
        "topic_count": len(topics),
        "date_range": date_range,
        "content": "\n\n".join(content_parts),
    }


def _normalize_editorial_topics(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect topic-like entries from specialist output."""
    agent_results = payload.get("agent_results")
    if isinstance(agent_results, dict):
        collected: List[Dict[str, Any]] = []
        for agent_name, agent_output in agent_results.items():
            if isinstance(agent_output, dict):
                nested_topics = _normalize_editorial_topics(agent_output)
                if nested_topics:
                    collected.extend(nested_topics)
                    continue
                collected.append({**agent_output, "source_agent": agent_name})
        if collected:
            return collected

    raw_topics = (
        payload.get("topic_summaries")
        or payload.get("topics")
        or payload.get("specialist_results")
        or payload.get("results")
        or []
    )
    if isinstance(raw_topics, dict):
        raw_topics = [raw_topics]

    normalized: List[Dict[str, Any]] = []
    for item in raw_topics:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def _parse_editorial_payload(specialized_agent_input: Any) -> Dict[str, Any]:
    """Parse the tool input into a normalized payload."""
    if isinstance(specialized_agent_input, dict):
        return dict(specialized_agent_input)
    if isinstance(specialized_agent_input, str):
        text = specialized_agent_input.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"raw_input": text}
        if isinstance(parsed, dict):
            return parsed
        return {"raw_input": text}
    return {}


def _normalize_execution_plan(execution_plan: Any) -> List[str]:
    """Convert a plan payload into a short list of readable steps."""
    if not execution_plan:
        return []
    if isinstance(execution_plan, str):
        return [line.strip("- ").strip() for line in execution_plan.splitlines() if line.strip()]
    if isinstance(execution_plan, dict):
        execution_plan = execution_plan.get("steps") or execution_plan.get("items") or []

    steps: List[str] = []
    for item in execution_plan if isinstance(execution_plan, list) else []:
        if isinstance(item, str):
            step = item.strip()
        elif isinstance(item, dict):
            step = str(item.get("step") or item.get("description") or item.get("title") or "").strip()
        else:
            step = str(item).strip()
        if step:
            steps.append(step)
    return steps


def _normalize_date_range(payload: Dict[str, Any]) -> Optional[str]:
    """Extract a compact date-range summary for the editorial intro."""
    date_range = payload.get("date_range")
    if isinstance(date_range, dict):
        start = str(date_range.get("start_date") or date_range.get("start") or "").strip()
        end = str(date_range.get("end_date") or date_range.get("end") or "").strip()
        if start and end:
            return f"{start} to {end}"
        if start:
            return start
    start = str(payload.get("start_date") or "").strip()
    end = str(payload.get("end_date") or "").strip()
    if start and end:
        return f"{start} to {end}"
    if start:
        return start
    return None


def _build_editorial_intro(
    audience: str,
    tone: str,
    topic_count: int,
    date_range: Optional[str],
) -> str:
    """Create the short editorial framing for the one-pager."""
    audience_text = audience or "LinkedIn"
    tone_text = tone or "professional"
    range_text = f" Date range: {date_range}." if date_range else ""
    topic_text = "topic" if topic_count == 1 else "topics"
    return (
        f"This one-pager is written for {audience_text} in a {tone_text} tone."
        f" It synthesizes {topic_count} {topic_text} from the specialist handoff."
        f"{range_text}"
    )


def _render_editorial_topic(topic: Dict[str, Any], index: int) -> str:
    """Render one topic section from the specialist payload."""
    title = (
        str(
            topic.get("title")
            or topic.get("topic")
            or topic.get("topic_title")
            or f"Topic {index}"
        ).strip()
        or f"Topic {index}"
    )
    description = str(
        topic.get("description")
        or topic.get("topic_description")
        or topic.get("summary")
        or ""
    ).strip()
    importance = str(
        topic.get("importance")
        or topic.get("main_results_and_importance")
        or topic.get("main_results")
        or ""
    ).strip()
    representative_papers = topic.get("representative_papers") or topic.get("papers") or []
    clearer_text = str(
        topic.get("clearer_text")
        or topic.get("clear_explanation")
        or ""
    ).strip()
    intuition = str(topic.get("intuition") or "").strip()
    metaphor = str(topic.get("metaphor") or "").strip()

    section_lines = [f"## {index}. {title}"]
    if description:
        section_lines.append(description)
    if importance:
        section_lines.append(f"Main results and importance: {importance}")
    if clearer_text:
        section_lines.append(f"Clear explanation: {clearer_text}")
    if intuition:
        section_lines.append(f"Intuition: {intuition}")
    if metaphor:
        section_lines.append(f"Metaphor: {metaphor}")
    if representative_papers:
        section_lines.append("Representative papers:")
        for paper_index, paper in enumerate(representative_papers, start=1):
            section_lines.append(f"{paper_index}. {_render_paper_reference(paper)}")
    else:
        section_lines.append("Representative papers: not provided.")
    return "\n".join(section_lines)


def _render_paper_reference(paper: Any) -> str:
    """Format a representative paper reference for the one-pager."""
    if isinstance(paper, dict):
        title = str(paper.get("title") or paper.get("paper_title") or "Untitled paper").strip()
        arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or paper.get("paper_id") or "").strip()
        main_result = str(paper.get("main_result") or paper.get("result") or "").strip()
        parts = [title]
        if arxiv_id:
            parts.append(f"({arxiv_id})")
        if main_result:
            parts.append(f"- {main_result}")
        return " ".join(parts)
    return str(paper).strip() or "Untitled paper"


def _enrich_message_for_michel(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Add an explicit simplification requirement when the request implies vulgarization."""
    if not _should_delegate_to_michel(message, history):
        return message

    return (
        f"{message}\n\n"
        "General-audience support is required. Use MichelAgent to simplify technical ideas, "
        "provide intuition, examples, or metaphors where the explanation would otherwise be too technical."
    )


def _should_delegate_to_michel(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """Detect requests that need MichelAgent's vulgarization support."""
    text_parts = [message]
    for item in history or []:
        content = str(item.get("content", "")).strip()
        if content:
            text_parts.append(content)
    normalized = " ".join(text_parts).casefold()
    triggers = (
        "general audience",
        "non-expert",
        "non expert",
        "beginner",
        "linkedin",
        "simple",
        "simpler",
        "plain english",
        "accessible",
        "intuitive",
        "intuition",
        "metaphor",
        "example",
        "examples",
        "vulgarize",
        "clarify",
        "clearer",
        "explain simply",
    )
    return any(trigger in normalized for trigger in triggers)




# def _infer_requested_topic_count(
#     message: str,
#     history: Optional[List[Dict[str, str]]] = None,
# ) -> Optional[int]:
#     """Infer an explicit topic count from the latest message, then history."""
#     inferred = _extract_topic_count_from_text(message)
#     if inferred is not None:
#         return inferred

#     for item in reversed(history or []):
#         inferred = _extract_topic_count_from_text(str(item.get("content", "")))
#         if inferred is not None:
#             return inferred
#     return None


# def _extract_topic_count_from_text(text: str) -> Optional[int]:
#     """Extract a requested topic count and clamp it to JuliusAgent's limits."""
#     lowered = text.casefold()
#     digit_match = re.search(r"\b([1-9][0-9]*)\s+(?:main\s+)?topics?\b", lowered)
#     if digit_match:
#         return max(1, min(int(digit_match.group(1)), DEFAULT_MAX_TOPICS))

#     for word, value in _NUMBER_WORDS.items():
#         if re.search(rf"\b{word}\s+(?:main\s+)?topics?\b", lowered):
#             return value
#     return None


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
        "Extract an explicit date range from the user message.\n"
        "Return JSON only with this exact schema: "
        "{\"start_date\":\"YYYY-MM-DD or null\",\"end_date\":\"YYYY-MM-DD or null\"}.\n"
        f"Today is {day_name}, {month_name} {today.day}, {today.year}.\n"
        "Rule:\n"
        f"- if there is a relative date, then end_date is {today.isoformat()}\n"
        "- if there is not any date intent in the message, then return null for start_date and end_date\n"
        f"User message: {message}"
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
