"""Phase 5 MichelAgent implementation with OpenAI Agents SDK."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from agents import Agent, Runner, function_tool, trace
from openai import OpenAI
from pydantic import BaseModel


MICHEL_SYSTEM_PROMPT = (
    "Mathematician and target-audience readability reviewer with outstanding skills to explain complex mathematical ideas to non-experts. "
    "You specialize in impactful intuitive explanations, simple reformulations, concrete examples, and metaphors. "
    "You are upbeat, optimistic, curious, and energetic. Connect ideas creatively across topics through "
    "relatable examples, memorable anecdotes, and occasionally unexpected but insightful parallels. "
    "Keep communication lively, concise, and engaging. You think quickly, but always check the reader is "
    "following; when appropriate, ask ‘So far, so good?’ before progressing."
)
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_MAX_TURNS = 10


class PedagogicalExplanation(BaseModel):
    """One Michel explanation assigned to a first-draft placeholder.

    Attributes:
        location: Identifier of the corresponding ``MICHEL_PEDAGOGY`` marker.
        purpose: Brief statement of the comprehension need being addressed.
        exact_text: Ready-to-insert pedagogical explanation written by Michel.
    """

    location: str
    purpose: str
    exact_text: str


class MichelReviewOutput(BaseModel):
    """Strict structured result returned by Michel's one-pager review.

    Attributes:
        satisfactory: Whether the draft is readable for the target audience.
        readability_reason: Concise rationale for the assessment.
        feedback: Readability findings for Julius.
        pedagogical_explanations: Location-keyed explanatory insertions.
    """

    satisfactory: bool
    readability_reason: str
    feedback: List[str]
    pedagogical_explanations: List[PedagogicalExplanation]


_PEDAGOGICAL_EXPLANATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pedagogical_explanation"],
    "properties": {
        "pedagogical_explanation": {"type": "string"},
    },
}


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


@function_tool(name_override="get_pedagogical_explanation_tool")
def get_pedagogical_explanation_tool(
    exact_text: str,
    audience: str = "general audience",
    tone: str = "clear and concise",
) -> Dict[str, Any]:
    """Generate Michel-style pedagogy from factual source text using an LLM.

    Args:
        exact_text: Factual topic description or paper main result that needs
            an explanatory companion.
        audience: Intended readers whose background determines the level of
            explanation.
        tone: Requested voice to preserve alongside Michel's personality.

    Returns:
        Dict[str, Any]: JSON-compatible success payload containing the source
        text and a ready-to-insert ``pedagogical_explanation``. On an API or
        validation failure, returns a JSON-compatible error payload.
    """
    fallback_explanation = (
        f"Think of it as a map that highlights the important pattern: {exact_text.strip()}"
        if exact_text.strip()
        else "There is no factual text yet to turn into a pedagogical explanation."
    )
    fallback = {
        "status": "success",
        "exact_text": exact_text,
        "audience": audience,
        "tone": tone,
        "pedagogical_explanation": fallback_explanation,
    }
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback

    prompt = (
        f"{MICHEL_SYSTEM_PROMPT}\n"
        "Turn the factual source text into one concise ready-to-insert pedagogical explanation. "
        "Keep it technically faithful and align it with the stated audience and tone. "
        "Use Michel's upbeat, curious, intuitive, concise, and engaging style; add a relatable example or accurate "
        "metaphor only when it helps. Return only the structured response.\n"
        f"Audience: {audience}\n"
        f"Tone: {tone}\n"
        f"Factual source text: {exact_text}"
    )
    try:
        response = OpenAI(api_key=api_key).responses.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            input=prompt,
            temperature=0.2,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pedagogical_explanation",
                    "strict": True,
                    "schema": _PEDAGOGICAL_EXPLANATION_SCHEMA,
                }
            },
        )
        payload = json.loads((response.output_text or "").strip())
        explanation = str(payload.get("pedagogical_explanation") or "").strip()
        if not explanation:
            raise ValueError("Structured response omitted pedagogical_explanation")
    except Exception as exc:
        return {
            "status": "error",
            "exact_text": exact_text,
            "audience": audience,
            "tone": tone,
            "pedagogical_explanation": "",
            "error": f"Could not generate a pedagogical explanation: {exc}",
        }

    return {
        "status": "success",
        "exact_text": exact_text,
        "audience": audience,
        "tone": tone,
        "pedagogical_explanation": explanation,
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
    """Assess whether a draft adequately serves its stated target audience.

    Args:
        explanation: Explanation or complete draft to evaluate for clarity and
            completeness.
        audience: Intended readers used to calibrate the assessment.
        concept: Optional concept that the explanation should cover.

    Returns:
        Dict[str, Any]: Success payload with a satisfaction flag, rationale,
        missing elements, and improvement advice calibrated to the audience.
    """
    prompt = (
        "Assess whether the explanation or one-pager about the concept is satisfactory for the audience.\n"
        "For a specialist audience, do not require pedagogical devices merely because they are absent.\n"
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
    """Create the readability reviewer and explanation specialist.

    Returns:
        Agent: Configured ``MichelAgent`` instance ready for execution.
    """
    return Agent(
        name="MichelAgent",
        instructions=(
            f"{MICHEL_SYSTEM_PROMPT}\n"
            "Use `get_pedagogical_explanation_tool` for every `MICHEL_PEDAGOGY` placeholder marked `needed=\"yes\"`. "
            "Pass the factual text immediately before the placeholder as `exact_text`, then copy the tool's returned "
            "`pedagogical_explanation` verbatim into the matching final `exact_text` field.\n"
            "Use `make_clearer_tool` when reformulation is needed to vulgarize technical content.\n"
            "Use `provide_intuition_tool` when examples or intuition are needed.\n"
            "Use `metaphor_tool` when a metaphor can improve understanding.\n"
            "Use `assess_non_expert_satisfaction_tool` to judge whether the complete draft is readable for its stated target audience.\n"
            "The first draft may contain placeholders in the form `[[MICHEL_PEDAGOGY id=\"<location>\" needed=\"yes|no\"]]`. "
            "First assess the draft. For every placeholder marked `needed=\"yes\"`, use the other tools as needed and create one exact pedagogical text for that exact location, even if no other gap is found.\n"
            "Every `exact_text` written for a `needed=\"yes\"` placeholder must use Michel's personality and communication style: "
            "upbeat, optimistic, curious, lively, concise, and engaging. Make the idea intuitive through a relatable "
            "example, anecdotal touch, or accurate metaphor whenever that helps the stated audience; never use one at the "
            "expense of technical accuracy. Keep the user's requested tone, and use ‘So far, so good?’ only when it fits "
            "naturally in a self-contained insertion.\n"
            "Never change paper titles, arXiv links, or technical claims. Do not rewrite the entire one-pager.\n"
            "Your final response is constrained to the Michel review JSON schema. Do not use Markdown fences or add prose outside that JSON.\n"
            "`feedback` is a concise list of readability findings. `pedagogical_explanations` is a list of objects with "
            "`location`, `purpose`, and `exact_text`. Each `location` must exactly match the `id` of its placeholder. "
            "Put only unformatted ready-to-insert text in `exact_text`: do not include the `Pedagogical explanation` label or Markdown emphasis, "
            "because Julius adds that presentation wrapper directly below the factual result. Use an empty list only when the draft has no `needed=\"yes\"` placeholders "
            "and no pedagogical change is needed."
        ),
        tools=[
            get_pedagogical_explanation_tool,
            make_clearer_tool,
            provide_intuition_tool,
            metaphor_tool,
            assess_non_expert_satisfaction_tool,
        ],
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        output_type=MichelReviewOutput,
    )


def run_michel_agent(
    one_pager_draft: str,
    issue_to_fix: str,
    audience: str = "general audience",
    tone: str = "clear and concise",
    review_stage: str = "initial",
) -> Dict[str, Any]:
    """Run one Michel readability-review turn for an editorial draft.

    Args:
        one_pager_draft: Complete current one-pager content for Michel to
            assess.
        issue_to_fix: Readability concern to assess or repair.
        audience: Intended readers used to calibrate the review.
        tone: Requested voice the revised draft must preserve.
        review_stage: ``"initial"`` for the factual draft or ``"follow-up"``
            after Julius has incorporated Michel's prior feedback.

    Returns:
        Dict[str, Any]: The agent reply and the arguments passed to invoked
        tools.

    Raises:
        Exception: If the OpenAI Agents SDK cannot complete the agent run.
    """
    message = (
        f"This is a {review_stage} readability review for a {audience} audience using a {tone} tone.\n"
        f"Review concern: {issue_to_fix}\n"
        "First assess whether the complete draft is readable for this audience. For every `MICHEL_PEDAGOGY` placeholder "
        "marked `needed=\"yes\"`, and only if marked `needed=\"yes\", give Julius an exact ready-to-insert pedagogical explanation with a `location` that "
        "exactly matches its `id`. Use `get_pedagogical_explanation_tool` for every such placeholder and copy its "
        "`pedagogical_explanation` result verbatim into `exact_text`. If the draft is not readable elsewhere, provide similarly exact location-keyed text. "
        "Write every required explanation in Michel's upbeat, curious, lively, concise, and engaging voice. Use a "
        "relatable example, anecdotal touch, or accurate metaphor when it improves comprehension, while preserving the "
        "requested tone and technical accuracy. Do not rewrite the full one-pager.\n"
        "Return valid JSON only with keys: satisfactory, readability_reason, feedback, "
        "pedagogical_explanations.\n"
        f"Complete one-pager draft:\n{one_pager_draft}\n"
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
        "reply": _serialize_michel_review_output(getattr(result, "final_output", None)),
        "tool_parameters": _extract_tool_parameters(getattr(result, "new_items", [])),
    }


def _serialize_michel_review_output(output: Any) -> str:
    """Serialize Michel's structured review to valid JSON text.

    Args:
        output: Agent final output validated against ``MichelReviewOutput``.

    Returns:
        str: JSON object text for Julius's placeholder-validation workflow.

    Raises:
        RuntimeError: If the SDK does not provide a structured Michel review.
    """
    if isinstance(output, MichelReviewOutput):
        return output.model_dump_json()
    if isinstance(output, dict):
        return json.dumps(output, ensure_ascii=False)
    raise RuntimeError("MichelAgent did not produce a structured review output")


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
    """Assess target-audience readability deterministically when unavailable.

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
    if _audience_requires_pedagogy(audience):
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
            "improvement_advice": (
                "Add a simple restatement, then include an example and an intuitive analogy."
                if _audience_requires_pedagogy(audience)
                else "Clarify the main idea while retaining the audience's appropriate technical level."
            ),
        }

    return {
        "satisfactory": True,
        "reason": f"The explanation is understandable for a {audience}.",
        "missing_elements": [],
        "improvement_advice": "No major gaps detected.",
    }


def _audience_requires_pedagogy(audience: str) -> bool:
    """Determine whether an audience explicitly calls for plain-language aids.

    Args:
        audience: Target-audience description supplied to MichelAgent.

    Returns:
        bool: ``True`` for descriptions that explicitly identify general,
        beginner, lay, or non-expert readers; otherwise ``False``.
    """
    normalized = audience.casefold()
    pedagogy_markers = (
        "general audience",
        "general public",
        "non-expert",
        "nonexpert",
        "lay audience",
        "layperson",
        "beginner",
        "beginners",
        "no technical background",
    )
    return any(marker in normalized for marker in pedagogy_markers)


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
