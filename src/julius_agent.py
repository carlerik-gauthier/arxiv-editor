"""JuliusAgent: the editorial coordinator for ArXiv research one-pagers."""

from __future__ import annotations

import json
import os
import re
from openai import OpenAI
from typing import Any, Dict, Iterable, List, Optional

from agents import Agent, Runner, function_tool, trace

from src.abdoulaye_agent import build_abdoulaye_agent
from src.alain_agent import build_alain_agent
from src.bruno_agent import build_bruno_agent
from src.chris_agent import build_chris_agent
from src.elisa_agent import build_elisa_agent
from src.felix_agent import build_felix_agent
from src.field_family import family_for_agent
from src.jean_baptiste_agent import build_jean_baptiste_agent
from src.michel_agent import build_michel_agent
from src.specialist_agent import extract_date_range as extract_specialist_date_range

EXPECTED_FORMAT_OUTPUT_RULE = """
    1. For every topic, the expected output structure **MUST BE**:
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

    2. Mandatory elements are <TOPIC TITLE>, <topic count>, <topic description>, <paper title>, <Representative paper main results> and <paper arxiv_id>.
    3. Optional elements are
    <pedagogical explanation for reprensative main result> and  <pedagogical explanation for topic description>. Both should be provided if MichelAgent is called.  
    <pedagogical explanation for reprensative main result> and <pedagogical explanation for topic description> **MUST** be provided when the one-pager is for a general or non-expert audience
    
    **Description**
    - <TOPIC TITLE> is an informative title describing the topic
    - <topic count> is the number of papers the topic covers.
    - <topic description> is a short description about the topic content
    - <pedagogical explanation for topic description> provides a simplified and more intuitive description description of the topic when addressing to a general audience
    - <Representative paper main results> is a short description of the paper's main results
    - <pedagogical explanation for reprensative main result> provides a simplified and more intuitive description of the main results when addressing to a general audience
    - <paper title> is the paper title
    - <paper arxiv_id> is the paper ID in ArXiv. It links to the online paper.

    They must be returned regardless of user request
"""

JULIUS_SYSTEM_PROMPT = (
    "Editor and coordinator role, responsible for planning, delegation and generating the one-pager. "
    "The one-pager must meet the user request, including tone. "
    "The one-pager must be engaging. You can use emojis or speech elevator techniques to make it appealing. "
    "You must remain professional. Unless stated otherwise by the user, the one-pager is aimed for a LinkedIn post. "
    "The post must contain between 1 and 5 topics. By default, unless stated otherwise, assume 3 topics.\n\n"
    "Preserve each delegated specialist's personality and communication style when representing their insight. "
    "Use those voices as editorial texture, but reconcile them with the user's requested tone, audience, and "
    "professional one-pager format. Do not imitate a personality in a way that reduces clarity, rigor, or factual accuracy.\n\n"
    "You own the editorial workflow:\n"
    "- Parse the user request, including date range, topics, and preferences.\n"
    "- Create a concise execution plan before writing the final one-pager.\n"
    "- You are responsible to allocate the number of topics to the different specialized agents. If an agent cannot return you requested, you pick another topic from another agent.\n"
    "- Route mathematics requests to ChrisAgent (probability/statistics), AlainAgent (algebra), BrunoAgent "
    "(geometry), ElisaAgent (applied mathematics/cryptography), or FelixAgent (dynamical systems/symplectic geometry).\n"
    "- Route AI requests to AbdoulayeAgent (machine learning) or JeanBaptisteAgent (data science, NLP, LLMs, and agentic AI).\n"
    "- Delegate clarity, intuition, and metaphor work to MichelAgent when the audience is general or when the user asks for simpler explanations.\n"
    "- When you call ChrisAgent or AlainAgent, make the request self-contained and include the date range, topic count, "
    "  and whether main results are required.\n"
    "- When you call MichelAgent, pass the exact concept or draft text that needs to be made clearer.\n"
    "- When pedagogical explanation is needed, call MichelAgent for review.\n"
    "- Your are prohibited from providing clarity, intuition, or metaphor material yourself; you must delegate it to MichelAgent.\n"
    "- Coordinate parallel execution where possible, but do not claim parallelism if only one specialist is used.\n"
    "- Synthesize the delegated material into one coherent one-pager. Topic title, topic description, represensative papers are mandatory.\n"
    "- If the request is outside the supported mathematics and AI specialties, reply politely and state the supported scope.\n"
)

DEFAULT_MAX_TOPICS = 5
DEFAULT_MODEL = "gpt-5.4-nano" # "gpt-4.1-mini"
DEFAULT_MAX_TURNS = 20


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

_SUPPORTED_GEOMETRY_KEYWORDS = (
    "geometry", "riemannian", "spectral", "manifold", "curvature", "math.dg", "math.sp"
)

_SUPPORTED_APPLIED_MATH_KEYWORDS = (
    "cryptography", "cryptographic", "security", "optimization", "numerical analysis", "applied mathematics", "control", "math.oc", "cs.cr", "math.na"
)
_SUPPORTED_HAMILTONIAN_DYNAMIC_KEYWORDS = (
    "dynamical systems", "dynamical", "symplectic", "hamiltonian", "chaos", "math.ds", "math.sg"
)
_SUPPORTED_DATA_SCIENCE_KEYWORDS = (
    "data science", "machine learning", "deep learning", "ml", "learning algorithm", "cs.lg", "stat.ml"
)


_SUPPORTED_AGENTIC_AI_KEYWORDS = (
    "natural language processing", "nlp", "large language model", "llm", "agentic ai",
    "artificial intelligence", "multi-agent", "cs.cl", "cs.ai", "cs.ma", "cs.ce"
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
    "astrophysics",
    "quantum physics",
    "economics",
    "biology",
)

_SUPPORTED_AGENT_ORDER = (
    "ChrisAgent", "AlainAgent", "BrunoAgent", "ElisaAgent", "FelixAgent",
    "AbdoulayeAgent", "JeanBaptisteAgent",
)


def build_julius_agent() -> Agent:
    """Build the editorial coordinator with delegation and drafting tools.

    Returns:
        Agent: Configured ``JuliusAgent`` instance ready to coordinate research
        specialists and assemble one-pagers.
    """
    chris_tool = build_chris_agent().as_tool(
        tool_name="chris_agent_tool",
        tool_description=(
            "Probability and Statistics specialist. Use for probability, stochastic processes, or statistics."
        ),
        max_turns=6,
    )
    alain_tool = build_alain_agent().as_tool(
        tool_name="alain_agent_tool",
        tool_description=(
            "Algebra specialist. Use for algebraic geometry, rings and algebras, group theory, and algebraic topology."
        ),
        max_turns=6,
    )
    bruno_tool = build_bruno_agent().as_tool(
        tool_name="bruno_agent_tool",
        tool_description="Spectral and Riemannian geometry specialist. Use for geometry, curvature, manifolds, or spectral theory.",
        max_turns=6,
    )
    elisa_tool = build_elisa_agent().as_tool(
        tool_name="elisa_agent_tool",
        tool_description="Applied mathematics and cryptography specialist. Use for optimization, numerical analysis, cryptography, or security.",
        max_turns=6,
    )
    felix_tool = build_felix_agent().as_tool(
        tool_name="felix_agent_tool",
        tool_description="Dynamical systems and symplectic geometry specialist. Use for Dynamical sytems, Hamiltonian systems, or symplectic geometry.",
        max_turns=6,
    )
    abdoulaye_tool = build_abdoulaye_agent().as_tool(
        tool_name="abdoulaye_agent_tool",
        tool_description="Data science amd Machine-learning specialist. Use for ML algorithms, learning theory, and AI applications.",
        max_turns=6,
    )
    jean_baptiste_tool = build_jean_baptiste_agent().as_tool(
        tool_name="jean_baptiste_agent_tool",
        tool_description="NLP, LLM, and agentic-AI specialist with production deployment expertise.",
        max_turns=6,
    )
    michel_tool = build_michel_agent().as_tool(
        tool_name="michel_agent_tool",
        tool_description=(
            "General-audience explainer. Use for feedback about simplifying technical mathematics, providing more intuition, "
            "or generating metaphors when addressing to a general audience."
        ),
        max_turns=6,
    )
    instructions = (
        f"{JULIUS_SYSTEM_PROMPT}\n"
        "Use `extract_date_range_tool` to find the date range requested by the user.\n"
        "Use `allocate_topics_tool` to decide how many topics each specialized agent should cover before delegating and the requested topic count.\n"
        "For example, if the user request mathematics, only allocate anything to agents specialized in mathematics\n"
        "Use `chris_agent_tool` to delegate probability or statistics work and collect topic titles, descriptions, "
        "representative papers, and main results when needed.\n"
        "Use `alain_agent_tool` to delegate algebra work and collect topic titles, descriptions, "
        "representative papers, and main results when needed.\n"
        "Use `bruno_agent_tool` for spectral or Riemannian geometry; use `elisa_agent_tool` for applied mathematics (e.g. numerical analysis)"
        "or cryptography; and use `felix_agent_tool` for dynamical systems or symplectic geometry.\n"
        "Use `abdoulaye_agent_tool` for data science or machine learning and `jean_baptiste_agent_tool` for NLP, "
        "LLMs, or agentic AI. Each specialist delegation must include the date range, topic count, audience, tone, "
        "and whether main results are required.\n"
        "Use `michel_agent_tool` to get feedbacks about improvement to address non-experts by vulgarizing, providing intuitions and metaphors."
        "examples, metaphors, or simpler phrasing.\n"
        "Use `editorial_one_pager_tool` to make the editorial work and build a first draft.\n"
        "Use `finalize_editorial_one_pager_tool` to finalize the one pager\n"
        "Use `revise_one_pager_tool` to decide if the one pager needs further improvement\n"
        "If the user has not specified an audience, assume LinkedIn.\n"
        "If the user has not specified a tone, assume professional.\n"
        "Always restate the execution plan briefly before the final one-pager.\n"
        "When calling `editorial_one_pager_tool`, you must provide the outputs of **all** specialized agents that have been called serialized as a JSON string, the requested topic count\n"
        "and an appealing title, the target audience and the tone to use. Remember, **all specialized agents** that got allocated topics MUST be provided!\n"
        "When calling `finalize_editorial_one_pager_tool`, provide the first draft, MichelAgent feedback, the tone and targeted audience.\n"
        "When calling `allocate_topics_tool`, pass the full user request so the allocation can reflect the domain mix and requested number of topics.\n"
        f"When using 'chris_agent_tool' you **must** provide in the instruction the date range, the inferred categories, the number of allocated topic to collect, the audience, the tone,\n"
        "and and whether main results are required.\n"
        f"When using 'alain_agent_tool' you **must** provide in the instruction the date range, the inferred categories, the nnumber of allocated topic to collect, the audience, the tone,\n"
        "and and whether main results are required.\n"
        f"When using 'abdoulaye_agent_tool' you **must** provide in the instruction the date range, the inferred categories, the number of allocated topic to collect, the audience, the tone,\n"
        "and and whether main results are required.\n"
        f"When using 'bruno_agent_tool' you **must** provide in the instruction the date range, the inferred categories, the number of allocated topic to collect, the audience, the tone,\n"
        "and and whether main results are required.\n"
        f"When using 'elisa_agent_tool' you **must** provide in the instruction the date range, the inferred categories, the number of allocated topic to collect, the audience, the tone,\n"
        "and and whether main results are required.\n"
        f"When using 'felix_agent_tool' you **must** provide in the instruction the date range, the inferred categories, the number of allocated topic to collect, the audience, the tone,\n"
        "and and whether main results are required.\n"
        f"When using 'jean_baptiste_agent_tool' you **must** provide in the instruction the date range, the inferred categories, the number of allocated topic count, the audience, the tone,\n"
        "and and whether main results are required.\n"
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
            bruno_tool,
            elisa_tool,
            felix_tool,
            abdoulaye_tool,
            jean_baptiste_tool,
            michel_tool,
            editorial_one_pager_tool,
            finalize_editorial_one_pager_tool,
            revise_one_pager_tool,
        ],
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    )


def run_julius_agent(
    message: str,
    conversation_history: Optional[Iterable[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run one traced editorial-coordinator turn.

    Args:
        message: Current user request for a research brief.
        conversation_history: Optional preceding chat messages with ``role`` and
            ``content`` fields.

    Returns:
        Dict[str, Any]: Reply text and parameters passed to invoked tools.

    Raises:
        Exception: If the OpenAI Agents SDK cannot complete a supported request.
    """
    history = list(conversation_history or [])
    with trace(
        "arxiv-editor-run",
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
                    "I can coordinate mathematics and AI research briefs (including probability, algebra, geometry, "
                    "cryptography, machine learning, data science, NLP, LLMs, and agentic AI). Please reformulate "
                    "the request around those domains."
                ),
                "tool_parameters": [],
            }

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
    """Combine prior messages and the current request into routeable text.

    Args:
        conversation_history: Prior chat messages with role and content fields.
        message: Current user message to append to the history.

    Returns:
        str: Plain-text context with the current request clearly labeled.
    """
    history_text = _serialize_conversation(conversation_history)
    if not history_text:
        return message
    return f"{history_text}\nNew user message: {message}"


def _serialize_conversation(conversation_history: Iterable[Dict[str, str]]) -> str:
    """Serialize usable chat history into a compact plain-text transcript.

    Args:
        conversation_history: Messages containing optional role and content keys.

    Returns:
        str: Newline-delimited transcript excluding blank messages.
    """
    lines: List[str] = []
    for item in conversation_history:
        role = str(item.get("role", "user")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _is_supported_specialist_request(text: str) -> bool:
    """Determine whether a request can be routed to current specialists.

    Args:
        text: User request or compiled conversation context to classify.

    Returns:
        bool: ``True`` when a supported domain or general research request is
        detected; otherwise ``False``.
    """
    normalized = text.casefold()
    if any(keyword in normalized for keyword in _SUPPORTED_PR_ST_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _SUPPORTED_ALGEBRA_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _SUPPORTED_GEOMETRY_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _SUPPORTED_AGENTIC_AI_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _SUPPORTED_APPLIED_MATH_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _SUPPORTED_DATA_SCIENCE_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _SUPPORTED_HAMILTONIAN_DYNAMIC_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _GENERIC_RESEARCH_KEYWORDS):
        return _is_supported_specialist_request_with_llm(text)
    return _is_supported_specialist_request_with_llm(text)


def _is_supported_specialist_request_with_llm(text: str) -> bool:
    """Use deterministic or model-based fallback routing for a request.

    Args:
        text: Request whose relationship to supported domains is uncertain.

    Returns:
        bool: Whether the fallback considers the request in scope. API failures
        are treated as unsupported requests.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        normalized = text.casefold()
        return any(keyword in normalized for keyword in _GENERIC_RESEARCH_KEYWORDS) and not any(
            keyword in normalized for keyword in _UNSUPPORTED_DOMAIN_KEYWORDS
        )

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    prompt = (
        "Decide whether the user text is linked to one of the supported domains, including closely related requests.\n"
        "Help yourself by using the lexicons related to the supported domains.\n"
        "Supported domains: probability, statistics, algebra, geometry, applied mathematics, cryptography, "
        "dynamical systems, symplectic geometry, machine learning, data science, NLP, LLMs, and agentic AI.\n"
        f"Lexicon related to the probability and statisfics speciality is: {', '.join(_SUPPORTED_PR_ST_KEYWORDS)}.\n"
        f"Lexicon related to the algebra speciality is: {', '.join(_SUPPORTED_ALGEBRA_KEYWORDS)}.\n"
        f"Lexicon related to the geometry speciality is: {', '.join(_SUPPORTED_GEOMETRY_KEYWORDS)}.\n"
        f"Lexicon related to the applied mathematic and cyptography speciality is: {', '.join(_SUPPORTED_APPLIED_MATH_KEYWORDS)}.\n"
        f"Lexicon related to the Hamiltonian dynamical system speciality is: {', '.join(_SUPPORTED_HAMILTONIAN_DYNAMIC_KEYWORDS)}.\n"
        f"Lexicon related to the Data Science and Machine Learning speciality is: {', '.join(_SUPPORTED_DATA_SCIENCE_KEYWORDS)}.\n"
        f"Lexicon related to the NLP and Agentic AI speciality is: {', '.join(_SUPPORTED_AGENTIC_AI_KEYWORDS)}.\n"
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
    """Allocate a request's topic budget across available specialists.

    Args:
        message: Full user request used to infer domains and topic count.
        available_agents: Optional subset of supported specialist names.

    Returns:
        Dict[str, Any]: Allocation payload with capped topic counts, reasoning,
        and fallback routing order.

    Raises:
        RuntimeError: If no usable allocation is returned by the model.
    """
    agent_order = [agent for agent in (available_agents or list(_SUPPORTED_AGENT_ORDER)) if agent in _SUPPORTED_AGENT_ORDER]
    if not agent_order:
        agent_order = list(_SUPPORTED_AGENT_ORDER)

    payload = _allocate_topics_with_llm(message, agent_order)
    allocations = _normalize_allocation_payload(payload, agent_order)
    requested_topic_count = payload.get("requested_topic_count")
    requested_topic_count = max(1, min(requested_topic_count, DEFAULT_MAX_TOPICS))
    allocations = _cap_allocations(allocations, requested_topic_count)
    if not isinstance(requested_topic_count, int):
        requested_topic_count = sum(item["topic_count"] for item in allocations)

    return {
        "status": "success",
        "requested_topic_count": requested_topic_count,
        "allocations": allocations,
        "selection_reason": str(payload.get("selection_reason") or "").strip(),
        "fallback_order": list(agent_order),
    }


def _allocate_topics_with_llm(message: str, agent_order: List[str]) -> Dict[str, Any]:
    """Ask the model to select specialists and their topic counts.

    Args:
        message: User request that defines the desired coverage.
        agent_order: Supported specialist names allowed in the allocation.

    Returns:
        Dict[str, Any]: Parsed allocation object returned by the model.

    Raises:
        RuntimeError: If no API key, model response, or valid JSON object is
            available.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for allocate_topics_tool")

    agent_descriptions = {
        "ChrisAgent": "Probability and statistical theory (math.PR, math.ST): stochastic processes, limit theorems, SDEs, inference, regression, time series, and Monte Carlo.",
        "AlainAgent": "Algebra (math.AG, math.RA, math.GR, math.AT): algebraic geometry, rings and algebras, group and representation theory, and algebraic topology.",
        "BrunoAgent": "Differential/Riemannian geometry and spectral theory (math.DG, math.SP): manifolds, curvature, geometric analysis, and spectra of operators.",
        "ElisaAgent": "Applied mathematics and cryptography (cs.CR, math.OC, math.NA): security protocols and privacy, optimization, control, operations research, and numerical algorithms.",
        "FelixAgent": "Dynamical systems and symplectic geometry (math.DS, math.SG): differential-equation flows, mechanics, complex dynamics, Hamiltonian systems, and symplectic flows.",
        "AbdoulayeAgent": "Machine learning (cs.LG, stat.ML): learning algorithms and theory, reinforcement learning, bandits, robustness, fairness, explainability, and applications.",
        "JeanBaptisteAgent": "NLP, LLMs, agentic and multi-agent AI, and production-oriented data science (cs.CL, cs.AI, cs.MA, cs.CE).",
    }
    available_agents = [
        {
            "agent_name": agent_name,
            "specialty": agent_descriptions.get(agent_name, "not implemented agent"),
            "specialty_domain": family_for_agent(agent_name).value
        }
        for agent_name in agent_order
    ]
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    prompt = (
        "You are JuliusAgent's planning assistant.\n"
        "Your job is to decide which specialized agents to call and how many topics to request from each from the user request.\n"
        "Use only the available agents.\n"
        "The total requested topics must stay between 1 and 5.\n"
        "**ANALYZE CAREFULLY** the user request to determine the agent allocation based on the available agents. Follow the following principle:\n"
        "- If the user names a supported domain explicitly, pick the matching agent.\n"
        "- If the user request closely related to supported domains, pick the matching agents.\n"
        "- If you cannot infer which agents to select, you may spread the topics across the available agents.\n"
        "- If the user provides an allocation, you must meet the given allocation. In particular you are prohibited to allocate less.\n"
        "- The total number of topics you allocate must be greater or equal to the total number of topics required by the user.\n"
        "- If you can infer the total number of topics requested by the user, you must be satisfy that number.\n"
        "- If you cannot find out the total number of topics requested by the user, assume it is 5.\n"
        "- Return JSON only with this schema:\n"
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
    """Normalize model allocation data into delegate-tool payloads.

    Args:
        payload: Raw model JSON containing candidate allocations.
        agent_order: Ordered allowlist of eligible specialist names.

    Returns:
        List[Dict[str, Any]]: Valid, unique allocations with tool names and
        default reasons filled in.

    Raises:
        RuntimeError: If the allocation field is invalid or has no usable item.
    """
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


def _cap_allocations(
    allocations: List[Dict[str, Any]],
    requested_topic_count: int,
) -> List[Dict[str, Any]]:
    """Trim allocations so delegation never exceeds the requested budget.

    Args:
        allocations: Normalized specialist allocations in routing order.
        requested_topic_count: Maximum combined topic count to retain.

    Returns:
        List[Dict[str, Any]]: Allocations shortened or removed after the budget
        is exhausted.
    """
    remaining = requested_topic_count
    capped: List[Dict[str, Any]] = []
    for allocation in allocations:
        if remaining == 0:
            break
        topic_count = min(int(allocation["topic_count"]), remaining)
        capped.append({**allocation, "topic_count": topic_count})
        remaining -= topic_count
    return capped


def _agent_tool_name(agent_name: str) -> str:
    """Return the delegation tool registered for a specialist.

    Args:
        agent_name: Canonical specialist name.

    Returns:
        str: JuliusAgent tool name that invokes the specialist.

    Raises:
        KeyError: If ``agent_name`` has no registered delegation tool.
    """
    mapping = {
        "ChrisAgent": "chris_agent_tool",
        "AlainAgent": "alain_agent_tool",
        "BrunoAgent": "bruno_agent_tool",
        "ElisaAgent": "elisa_agent_tool",
        "FelixAgent": "felix_agent_tool",
        "AbdoulayeAgent": "abdoulaye_agent_tool",
        "JeanBaptisteAgent": "jean_baptiste_agent_tool",
    }
    return mapping[agent_name]


def _allocation_reason(agent_name: str) -> str:
    """Provide a default explanation for allocating work to a specialist.

    Args:
        agent_name: Canonical specialist name to explain.

    Returns:
        str: Domain-specific allocation rationale or a general fallback reason.
    """
    if agent_name == "ChrisAgent":
        return "Interest in probability or statistics detected."
    if agent_name == "AlainAgent":
        return "Interest in algebra detected."
    if agent_name == "BrunoAgent":
        return "Interest in spectral or Riemannian geometry detected."
    if agent_name == "ElisaAgent":
        return "Interest in applied mathematics or cryptography detected."
    if agent_name == "FelixAgent":
        return "Interest in dynamical systems or symplectic geometry detected."
    if agent_name == "AbdoulayeAgent":
        return "Interest in data science or machine learning detected."
    if agent_name == "JeanBaptisteAgent":
        return "Interest in NLP, LLMs, or agentic AI detected."
    return "Allocated by JuliusAgent."

@function_tool(name_override="editorial_one_pager_tool")
def editorial_one_pager_tool(
    specialized_agent_inputs: str,
    requested_topic_count: int,
    title: str="ArXiv Research Brief",
    audience: str = "LinkedIn",
    tone: str = "professional",
) -> Dict[str, Any]:
    """Synthesize specialist outputs into an audience-specific one-pager draft.

    Args:
        specialized_agent_inputs: JSON string or compatible handoff containing
            specialist findings and optional explanation feedback.
        requested_topic_count: Maximum number of topics to include in the draft.
        title: Editorial title to use when the model does not supply one.
        audience: Intended readership for structure and language choices.
        tone: Desired editorial voice for the draft.

    Returns:
        Dict[str, Any]: Parsed editorial draft with default status and content
        fields supplied when absent.

    Raises:
        RuntimeError: If no API key, a model failure, empty output, or invalid
        JSON prevents the draft from being produced.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for editorial_one_pager_tool")

    payload = _normalize_editorial_handoff(specialized_agent_inputs)

    prompt = (
        f"You are an editorial assistant for a one-pager called {title}.\n"
        "Use the specialist handoff to select the best topics, decide whether general-audience "
        "explanation, vulgarization, intuition, or metaphors are needed, and write a coherent first draft.\n"
        f"Your first draft must satisfy the following format : {EXPECTED_FORMAT_OUTPUT_RULE}.\n"
        "Return JSON only with keys: status, title, topic_count, "
        "editorial_summary, content, one_pager_draft.\n"
        "Do not invent unsupported facts.\n"
        "Use concise editorial prose.\n"
        "**NEVER change** the paper titles nor the links to ArXiv\n"
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
    """Normalize JSON, mappings, or mapping collections into one handoff.

    Args:
        specialized_agent_input: Serialized JSON, a mapping, or iterable of
        mappings containing specialist results.

    Returns:
        Dict[str, Any]: Single dictionary suitable for editorial prompting;
        malformed strings are retained as ``raw_input``.
    """
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
    """Finalize a draft after incorporating MichelAgent's feedback.

    Args:
        one_pager: Latest editorial draft to revise or approve.
        michel_agent_feedback: Simplification and accessibility feedback.
        audience: Intended readers for the finalized version.
        tone: Desired editorial voice for the finalized version.

    Returns:
        Dict[str, Any]: Finalization decision, rationale, content, and metadata.

    Raises:
        RuntimeError: If no API key, a model failure, empty output, or invalid
        JSON prevents finalization.
    """
    prompt = (
        "You are finalizing an editorial one-pager.\n"
        "Review the one pager draft and take into account MichelAgent feedbacks, especially when the target audience is made of non-experts (e.g. a general audience)\n"
        f"Remember you addressing to a {audience} audience and your tone must be {tone}.\n"
        "Return JSON only with keys: status, final_decision, reason, content, title"
        "needs_further_revision, michel_assessment, editorial_summary.\n"
        "Use status=ready_to_deliver and final_decision=deliver only if the draft content complexity suits to the "
        "target audience and MichelAgent feedback has been integrated.\n"
        f"Remember the one_pager **MUST** satisfy {EXPECTED_FORMAT_OUTPUT_RULE}\n"
        "**NEVER change** the paper titles nor the links to ArXiv\n"
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
    """Assess whether a one-pager fits its intended audience and tone.

    Args:
        one_pager: Draft to evaluate for editorial fit.
        audience: Intended readers used to judge appropriateness.
        tone: Requested editorial voice used to judge appropriateness.

    Returns:
        Dict[str, Any]: Assessment status, issue type, recommendation, and the
        reviewed draft metadata.

    Raises:
        RuntimeError: If no API key, a model failure, empty output, or invalid
        JSON prevents the assessment.
    """
    prompt = (
        "You are reviewing a one-pager draft.\n"
        "Judge whether it matches the requested tone and audience.\n"
        "If it does not, decide whether the main issue is simplification, clarity, metaphor, or intuition.\n"
        "Return JSON only with keys: status, appropriate, reason, issue_type, recommendation.\n"
        "Use issue_type=none when the draft is appropriate.\n"
        "Use issue_type=clarity when the draft is too vague or hard to follow.\n"
        "Use issue_type=metaphor when the draft needs a metaphor to make the idea accessible.\n"
        "Use issue_type=intuition when the draft needs more intuitive explanation.\n"
        "Use issue_type=simplification when the draft needs simplifications.\n"
        f"Remember the one_pager **MUST** satisfy {EXPECTED_FORMAT_OUTPUT_RULE}\n"
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
    """Extract a normalized ISO date range from a user request.

    Args:
        message: User text that may contain explicit or relative dates.

    Returns:
        Dict[str, Any]: Mapping containing ISO ``start_date`` and ``end_date``.
    """
    start_date, end_date = _extract_date_range(message)
    return {
        "start_date": start_date,
        "end_date": end_date,
    }


def _extract_date_range(message: str) -> tuple[str, str]:
    """Delegate date parsing to the shared specialist date-range parser.

    Args:
        message: User text that may contain explicit or relative dates.

    Returns:
        tuple[str, str]: Inclusive ISO start and end dates.
    """
    return extract_specialist_date_range(message)


def _parse_json_object_response(content: str, tool_name: str) -> Dict[str, Any]:
    """Parse a JSON object from plain, fenced, or prose-wrapped model output.

    Args:
        content: Raw model response that should contain a JSON object.
        tool_name: Tool name included in validation errors.

    Returns:
        Dict[str, Any]: First valid JSON object found in ``content``.

    Raises:
        RuntimeError: If no valid JSON object can be extracted.
    """
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
    """Find the first balanced JSON object in free-form response text.

    Args:
        content: Text that may contain a JSON object among other prose.

    Returns:
        Optional[str]: Balanced JSON-object substring, or ``None`` if absent.
    """
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
    """Extract function-tool names and arguments from SDK run items.

    Args:
        new_items: Items emitted during an OpenAI Agents SDK run.

    Returns:
        List[Dict[str, Any]]: One tool-and-arguments mapping per function call.
    """
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
