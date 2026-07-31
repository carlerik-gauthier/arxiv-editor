"""Phase 5 MichelAgent implementation with OpenAI Agents SDK."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from agents import Agent, Runner, function_tool, trace
from openai import OpenAI


MICHEL_SYSTEM_PROMPT = (
    "Mathematician with outstanding skills to explain complex mathematical ideas to non-experts. "
    "You specialize in impactful intuitive explanations, simple reformulations, concrete examples, and metaphors. "
    "You are upbeat, optimistic, curious, and energetic. Connect ideas creatively across topics through "
    "relatable examples, memorable anecdotes, and occasionally unexpected but insightful parallels. "
    "Keep communication lively, concise, and engaging. You think quickly, but always check the reader is "
    "following; when appropriate, ask ‘So far, so good?’ before progressing."
)
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TURNS = 6


@function_tool(name_override="make_clearer_tool")
def make_clearer_tool(
    text: str,
    audience: str = "general audience",
    tone: str = "clear and concise",
) -> Dict[str, Any]:
    """Reformulate technical content in language suitable for a target audience.

    Args:
        text: Technical material to rewrite.
        audience: Intended readers whose prior knowledge guides simplification.
        tone: Requested voice for the rewritten explanation.

    Returns:
        Dict[str, Any]: Success payload containing the audience, tone, clearer
        text, and the applied simplifications.
    """
    prompt = (
        "Rewrite the material so it is easier to understand for the stated audience.\n"
        "Preserve the mathematics ideas and concepts, remove unnecessary jargon, and stay concise.\n"
        "The more the audience is general, the more you need to simplify the material."
        "Conversely, when the audience is made of specialists, you can keep details"
        "Return JSON with keys: clearer_text, simplifications.\n"
        f"Audience: {audience}\n"
        f"Tone: {tone}\n"
        f"Text: {text}"
    )

    fallback = {
        "clearer_text": (
            f"For a {audience}, the main idea is: {text.strip()}"
            if text.strip()
            else "No text was provided to reformulate."
        ),
        "simplifications": [
            "Reduced technical wording.",
            "Kept the main mathematical claim.",
        ],
    }
    payload = _json_response(prompt, fallback)
    return {
        "status": "success",
        "audience": audience,
        "tone": tone,
        **payload,
    }


@function_tool(name_override="provide_intuition_tool")
def provide_intuition_tool(
    concept: str,
    explanation: str = "",
    audience: str = "general audience",
) -> Dict[str, Any]:
    """Provide an accessible intuition and examples for a mathematical concept.

    Args:
        concept: Concept that needs an intuitive explanation.
        explanation: Existing explanation to complement, if available.
        audience: Intended readers whose background determines the detail level.

    Returns:
        Dict[str, Any]: Success payload with the concept, audience, intuition,
        and one or more examples.
    """
    prompt = (
        "Explain the concept with intuition for non-experts.\n"
        "When the audience is made of non-experts, it is adapted to provide intuitions"
        "The more the audience is general, the more you need to provide intuition."
        "Conversely, when the audience is made of specialists, you don't need to provide much additional intuition"
        "Return JSON with keys: intuition, examples.\n"
        f"Audience: {audience}\n"
        f"Concept: {concept}\n"
        f"Current explanation: {explanation}"
    )
    fallback = {
        "intuition": (
            f"Think of {concept} as a way to track the main pattern without carrying every technical detail."
        ),
        "examples": [
            f"A simple example of {concept} is to compare a noisy process with its average behavior.",
        ],
    }
    payload = _json_response(prompt, fallback)
    return {
        "status": "success",
        "audience": audience,
        "concept": concept,
        **payload,
    }


@function_tool(name_override="metaphor_tool")
def metaphor_tool(
    concept: str,
    explanation: str = "",
    audience: str = "general audience",
) -> Dict[str, Any]:
    """Produce a faithful metaphor that makes a concept easier to picture.

    Args:
        concept: Concept for which to construct a metaphor.
        explanation: Existing explanation to use as context, if available.
        audience: Intended readers whose background guides the metaphor.

    Returns:
        Dict[str, Any]: Success payload containing the concept, audience,
        metaphor, and an explanation of its value.
    """
    prompt = (
        "Create vivid but accurate metaphors about the concept.\n"
        "It must be adapted to the audience.\n"
        "Return JSON with keys: metaphor, why_it_helps.\n"
        f"Audience: {audience}\n"
        f"Concept: {concept}\n"
        f"Current explanation: {explanation}"
    )
    fallback = {
        "metaphor": (
            f"{concept} is like a map: it leaves out the clutter so you can still see the structure."
        ),
        "why_it_helps": "It highlights structure without drowning the reader in formalism.",
    }
    payload = _json_response(prompt, fallback)
    return {
        "status": "success",
        "audience": audience,
        "concept": concept,
        **payload,
    }


@function_tool(name_override="assess_non_expert_satisfaction_tool")
def assess_non_expert_satisfaction_tool(
    explanation: str,
    audience: str = "general audience",
    concept: str = "",
) -> Dict[str, Any]:
    """Assess whether an explanation adequately serves non-expert readers.

    Args:
        explanation: Explanation to evaluate for clarity and completeness.
        audience: Intended readers used to calibrate the assessment.
        concept: Optional concept that the explanation should cover.

    Returns:
        Dict[str, Any]: Success payload with a satisfaction flag, rationale,
        missing elements, and improvement advice.
    """
    prompt = (
        "Assess whether the explanation about the concept is satisfactory for the audience.\n"
        "Return JSON with keys: satisfactory, reason, missing_elements, improvement_advice.\n"
        "Use a boolean for satisfactory.\n"
        f"Audience: {audience}\n"
        f"Concept: {concept}\n"
        f"Explanation: {explanation}"
    )
    fallback = _fallback_non_expert_assessment(
        explanation=explanation,
        audience=audience,
        concept=concept,
    )

    payload = _json_response(prompt, fallback)
    satisfactory = bool(payload.get("satisfactory", False))
    missing_elements = payload.get("missing_elements") or []
    if not isinstance(missing_elements, list):
        missing_elements = [str(missing_elements)]
    return {
        "status": "success",
        "audience": audience,
        "concept": concept,
        "satisfactory": satisfactory,
        "reason": str(payload.get("reason") or "").strip(),
        "missing_elements": [str(item).strip() for item in missing_elements if str(item).strip()],
        "improvement_advice": str(payload.get("improvement_advice") or "").strip(),
    }


def build_michel_agent() -> Agent:
    """Create the explanation specialist with its four SDK tools.

    Returns:
        Agent: Configured ``MichelAgent`` instance ready for execution.
    """
    return Agent(
        name="MichelAgent",
        instructions=(
            f"{MICHEL_SYSTEM_PROMPT}\n"
            "Use `make_clearer_tool` when reformulation is needed to vulgarize technical content.\n"
            "Use `provide_intuition_tool` when examples or intuition are needed.\n"
            "Use `metaphor_tool` when a metaphor can improve understanding.\n"
            "Use `assess_non_expert_satisfaction_tool` to judge whether the explanation is satisfactory for non-experts.\n"
            "If the assessment is negative, explain why, identify what is missing, then revise using the other tools.\n"
            "When relevant, combine the tools in sequence until the explanation is satisfactory for non-experts."
            "Return a valid JSON with keys input_message and feedback"
        ),
        tools=[
            make_clearer_tool,
            provide_intuition_tool,
            metaphor_tool,
            assess_non_expert_satisfaction_tool,
        ],
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    )


def run_michel_agent(one_pager_draft: str, issue_to_fix: str) -> Dict[str, Any]:
    """Run one traced MichelAgent turn for a user message.

    Args:
        one_pager_draft: User one pager content for Michel to process.
        issue_to_fix: The issue Michel needs to fix

    Returns:
        Dict[str, Any]: The agent reply and the arguments passed to invoked tools.

    Raises:
        Exception: If the OpenAI Agents SDK cannot complete the agent run.
    """
    message = (
        f"The following one pager draft has {issue_to_fix} issue that you must fix.\n"
        "Return a valid JSON only with keys 'one_pager_draft' and 'michel_agent_feedback'.\n"
        f"'one_pager_draft' contains {one_pager_draft}, and 'michel_agent_feedback' contains a string that contains the suggested improvements.\n"
        f"The one pager draft to improve is {one_pager_draft}\n"
        "**NEVER change** the paper titles nor the links to ArXiv"
    )
    agent = build_michel_agent()
    with trace(
        "phase5-michel-agent-run",
        metadata={"agent": "MichelAgent"},
        disabled=os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "0") == "1",
    ):
        result = Runner.run_sync(agent, message, max_turns=DEFAULT_MAX_TURNS)

    return {
        "reply": str(getattr(result, "final_output", "")),
        "tool_parameters": _extract_tool_parameters(getattr(result, "new_items", [])),
    }


def _json_response(prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured model response or a deterministic fallback.

    Args:
        prompt: Instruction requesting a JSON object from the configured model.
        fallback: Payload returned when no model response can be used.

    Returns:
        Dict[str, Any]: Parsed model JSON when valid; otherwise ``fallback``.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.2,
        )
    except Exception:
        return fallback

    content = (response.output_text or "").strip()
    if not content:
        return fallback
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(parsed, dict):
        return fallback
    return parsed


def _fallback_non_expert_assessment(
    explanation: str,
    audience: str,
    concept: str,
) -> Dict[str, Any]:
    """Assess explanation quality deterministically when the model is unavailable.

    Args:
        explanation: Explanation whose clarity and completeness are assessed.
        audience: Readers used to calibrate the fallback assessment.
        concept: Optional concept being explained.

    Returns:
        Dict[str, Any]: Assessment payload with satisfaction, gaps, and advice.
    """
    stripped = explanation.strip()
    if not stripped:
        return {
            "satisfactory": False,
            "reason": f"There is no usable explanation yet for a {audience}.",
            "missing_elements": ["A plain-language explanation", "A concrete example or intuition"],
            "improvement_advice": "Start by stating the core idea in simple terms, then add an example.",
        }

    normalized = stripped.casefold()
    missing_elements: List[str] = []
    if len(stripped.split()) < 20:
        missing_elements.append("More detail about the main idea")
    if "example" not in normalized:
        missing_elements.append("A concrete example")
    if "like" not in normalized and "as if" not in normalized:
        missing_elements.append("An intuitive analogy or metaphor")

    if missing_elements:
        concept_text = f" for {concept}" if concept else ""
        return {
            "satisfactory": False,
            "reason": (
                f"The explanation is still too thin for a {audience}{concept_text}. "
                "It does not yet make the idea easy to picture."
            ),
            "missing_elements": missing_elements,
            "improvement_advice": "Add a simple restatement, then include an example and an intuitive analogy.",
        }

    return {
        "satisfactory": True,
        "reason": f"The explanation is understandable for a {audience}.",
        "missing_elements": [],
        "improvement_advice": "No major gaps detected.",
    }


def _extract_tool_parameters(new_items: List[Any]) -> List[Dict[str, Any]]:
    """Extract function-tool names and arguments from SDK run items.

    Args:
        new_items: Items emitted by an OpenAI Agents SDK run.

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
