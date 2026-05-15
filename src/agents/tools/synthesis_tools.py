"""Deterministic synthesis tools for multi-agent draft generation.

These tools are small, documented contracts for step 6.3. They accept agent
analyses, selected papers, and SummaryRequest preferences, then return
structured draft pieces that can later be replaced by LLM-backed versions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from src.agents.base_agent import AgentTool
from src.generation.user_request import Audience, Depth, SummaryRequest
from src.openai_client import default_openai_model, resolve_openai_client


def create_topic_overview_tool(
    topic: str,
    papers: Iterable[Dict[str, Any]],
    analyses: Optional[Iterable[Dict[str, Any]]] = None,
    summary_request: Optional[Any] = None,
) -> Dict[str, Any]:
    """Synthesize a topic-level overview for the requested audience and depth."""
    request = _coerce_request(summary_request)
    paper_list = list(papers or [])
    analysis_list = list(analyses or [])
    paper_titles = [_paper_title(paper) for paper in paper_list[: request.max_papers]]
    key_points = _collect_key_points(analysis_list, paper_list)
    lead = key_points[0] if key_points else f"Recent work clusters around {topic}."
    if request.audience == Audience.NON_EXPERT:
        lead = generate_layperson_explanation_tool(lead)["explanation"]
    elif request.audience == Audience.EXPERT:
        lead = generate_expert_explanation_tool(lead, domain=topic)["explanation"]

    return {
        "topic": topic,
        "title": topic.title(),
        "overview": lead,
        "paper_titles": paper_titles,
        "paper_count": len(paper_list),
        "depth": request.depth.value,
        "confidence_notes": _confidence_notes(analysis_list),
    }


def create_paper_summary_tool(
    paper: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    summary_request: Optional[Any] = None,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Summarize one representative paper according to user preferences."""
    request = _coerce_request(summary_request)
    analysis = analysis or {}
    if llm_client is not None:
        llm_summary = _create_paper_summary_with_llm(
            llm_client=llm_client,
            paper=paper,
            analysis=analysis,
            request=request,
        )
        if llm_summary:
            return llm_summary

    return _create_paper_summary_heuristic(paper, analysis, request)


def _create_paper_summary_heuristic(
    paper: Dict[str, Any],
    analysis: Dict[str, Any],
    request: SummaryRequest,
) -> Dict[str, Any]:
    """Build a paper summary without an LLM client."""
    problem = analysis.get("problem") or analysis.get("problem_statement") or paper.get("summary", "")
    results = analysis.get("main_results") or analysis.get("key_results") or []
    if isinstance(results, str):
        results = [results]
    statement = results[0] if results else _first_words(paper.get("summary", ""), 32)
    if request.depth == Depth.BRIEF:
        statement = _first_words(statement, 24)

    return {
        "title": _paper_title(paper),
        "arxiv_id": paper.get("arxiv_id") or paper.get("id"),
        "problem": _first_words(problem, 40),
        "main_result": _first_words(statement, 45),
        "significance": _first_words(
            analysis.get("impact_summary") or analysis.get("significance") or statement,
            36,
        ),
        "audience": request.audience.value,
        "source": "heuristic",
    }


def generate_expert_explanation_tool(content: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """Return a technical explanation shell for expert readers."""
    domain_text = f" in {domain}" if domain else ""
    return {
        "explanation": f"Technically{domain_text}, {content}",
        "style": "expert",
    }


def generate_layperson_explanation_tool(
    content: str,
    metaphors: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return an accessible explanation shell for non-expert readers."""
    metaphor_text = f" Think of it as {next(iter(metaphors), '')}." if metaphors else ""
    return {
        "explanation": f"In plain terms, {content}{metaphor_text}".strip(),
        "style": "layperson",
    }


def rank_summary_items_tool(
    items: Iterable[Dict[str, Any]],
    ranking_goal: str = "relevance",
) -> Dict[str, Any]:
    """Rank topics or papers by an available score while preserving stable ties."""
    ranked = sorted(
        list(items or []),
        key=lambda item: (
            item.get(f"{ranking_goal}_score")
            or item.get("score")
            or item.get("impact_score")
            or 0
        ),
        reverse=True,
    )
    return {
        "ranking_goal": ranking_goal,
        "items": [
            {**item, "rank": index + 1}
            for index, item in enumerate(ranked)
        ],
    }


def review_and_refine_tool(content: str, criteria: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Run a deterministic quality pass and return concise warnings."""
    criteria_list = list(criteria or [])
    warnings: List[str] = []
    if not content or not content.strip():
        warnings.append("Content is empty.")
    if "citation" in criteria_list and "arxiv" not in content.lower():
        warnings.append("No ArXiv citation marker found.")
    if "audience" in criteria_list and len(content.split()) < 40:
        warnings.append("Content may be too short for audience calibration.")
    return {
        "refined_content": content.strip(),
        "criteria": criteria_list,
        "warnings": warnings,
        "passed": not warnings,
    }


def _create_paper_summary_with_llm(
    llm_client: Any,
    paper: Dict[str, Any],
    analysis: Dict[str, Any],
    request: SummaryRequest,
) -> Dict[str, Any]:
    """Call an injected LLM client and normalize the stable paper-summary contract."""
    prompt = _paper_summary_prompt(paper, analysis, request)
    try:
        response = _call_summary_llm(llm_client, prompt)
        payload = _parse_summary_response(response)
    except Exception:
        return {}

    heuristic = _create_paper_summary_heuristic(paper, analysis, request)
    return {
        "title": str(payload.get("title") or heuristic["title"]),
        "arxiv_id": payload.get("arxiv_id") or heuristic.get("arxiv_id"),
        "problem": _first_words(payload.get("problem") or heuristic["problem"], 40),
        "main_result": _first_words(payload.get("main_result") or heuristic["main_result"], 45),
        "significance": _first_words(payload.get("significance") or heuristic["significance"], 36),
        "audience": request.audience.value,
        "source": "llm",
    }


def _paper_summary_prompt(
    paper: Dict[str, Any],
    analysis: Dict[str, Any],
    request: SummaryRequest,
) -> str:
    """Build a compact prompt for LLM-backed paper summaries."""
    return (
        "Summarize one representative research paper for Julius.\n"
        "Return strict JSON with keys: title, arxiv_id, problem, main_result, significance.\n"
        f"Audience: {request.audience.value}\n"
        f"Depth: {request.depth.value}\n"
        f"Paper title: {_paper_title(paper)}\n"
        f"ArXiv ID: {paper.get('arxiv_id') or paper.get('id') or ''}\n"
        f"Paper summary: {paper.get('summary') or paper.get('abstract') or ''}\n"
        f"Existing analysis: {json.dumps(analysis, default=str, sort_keys=True)}"
    )


def _call_summary_llm(llm_client: Any, prompt: str) -> Any:
    """Call an OpenAI client shape used by paper-summary tools."""
    client = resolve_openai_client(llm_client, required=True)

    responses_api = getattr(client, "responses", None)
    create_method = getattr(responses_api, "create", None)
    if callable(create_method):
        return create_method(model=getattr(client, "model", default_openai_model()), input=prompt)

    chat_api = getattr(client, "chat", None)
    completions_api = getattr(chat_api, "completions", None)
    create_method = getattr(completions_api, "create", None)
    if callable(create_method):
        return create_method(
            model=getattr(client, "model", default_openai_model()),
            messages=[{"role": "user", "content": prompt}],
        )

    raise TypeError(
        "llm_client must be an OpenAI client exposing responses.create or "
        "chat.completions.create"
    )


def _parse_summary_response(response: Any) -> Dict[str, Any]:
    """Normalize LLM output into a dictionary."""
    if isinstance(response, dict):
        return dict(response)
    text = _extract_llm_text(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"main_result": text, "significance": text}
    return payload if isinstance(payload, dict) else {"main_result": text, "significance": text}


def _extract_llm_text(response: Any) -> str:
    """Extract text from common OpenAI-compatible response shapes."""
    if isinstance(response, str):
        return response.strip()
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text.strip()
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            return content.strip()
    return str(response).strip()


def get_synthesis_tools() -> List[AgentTool]:
    """Return synthesis tools for Julius and specialist agents."""
    return [
        AgentTool(
            name="create_topic_overview_tool",
            description="Synthesize a topic overview from papers, analyses, and SummaryRequest preferences.",
            function=create_topic_overview_tool,
            required_parameters=["topic", "papers"],
        ),
        AgentTool(
            name="create_paper_summary_tool",
            description="Summarize one representative paper for the requested audience and depth.",
            function=create_paper_summary_tool,
            required_parameters=["paper"],
        ),
        AgentTool(
            name="generate_expert_explanation_tool",
            description="Convert content into a concise expert-facing explanation.",
            function=generate_expert_explanation_tool,
            required_parameters=["content"],
        ),
        AgentTool(
            name="generate_layperson_explanation_tool",
            description="Convert content into a concise non-expert explanation.",
            function=generate_layperson_explanation_tool,
            required_parameters=["content"],
        ),
        AgentTool(
            name="rank_summary_items_tool",
            description="Rank draft topics or paper summaries by a stated goal.",
            function=rank_summary_items_tool,
            required_parameters=["items"],
        ),
        AgentTool(
            name="review_and_refine_tool",
            description="Check and lightly refine synthesized content against criteria.",
            function=review_and_refine_tool,
            required_parameters=["content"],
        ),
    ]


def _coerce_request(summary_request: Optional[Any]) -> SummaryRequest:
    """Normalize optional request input for tool calls."""
    if summary_request is None:
        return SummaryRequest()
    if isinstance(summary_request, SummaryRequest):
        return summary_request
    if isinstance(summary_request, dict) and "summary_request" in summary_request:
        return SummaryRequest.model_validate(summary_request["summary_request"])
    return SummaryRequest.model_validate(summary_request)


def _paper_title(paper: Dict[str, Any]) -> str:
    """Return a stable title for paper-like dictionaries."""
    return str(paper.get("title") or paper.get("name") or "Untitled paper")


def _collect_key_points(
    analyses: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
) -> List[str]:
    """Collect result-like sentences from analyses, falling back to abstracts."""
    points: List[str] = []
    for analysis in analyses:
        for key in ("impact_summary", "significance", "problem"):
            if analysis.get(key):
                points.append(str(analysis[key]))
        results = analysis.get("main_results") or analysis.get("key_results") or []
        if isinstance(results, str):
            points.append(results)
        else:
            points.extend(str(result) for result in results[:2])
    if not points:
        points.extend(str(paper.get("summary", "")) for paper in papers if paper.get("summary"))
    return [_first_words(point, 45) for point in points if point]


def _confidence_notes(analyses: List[Dict[str, Any]]) -> List[str]:
    """Extract uncertainty notes from specialist analyses."""
    notes = [
        str(analysis.get("confidence_note") or analysis.get("confidence"))
        for analysis in analyses
        if analysis.get("confidence_note") or analysis.get("confidence")
    ]
    return notes or ["deterministic synthesis; specialist validation pending"]


def _first_words(text: str, limit: int) -> str:
    """Return the first `limit` words of text."""
    words = str(text or "").split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + "..."
