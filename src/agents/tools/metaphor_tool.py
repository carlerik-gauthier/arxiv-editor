"""Tools for generating non-technical mathematical intuition."""

from __future__ import annotations

import re
from typing import Dict

from src.agents.base_agent import AgentTool


def create_metaphor_tool(concept: str, audience: str = "non-experts") -> Dict[str, str]:
    """
    Create a compact metaphor for a mathematical concept.

    This deterministic tool gives Michel a local way to produce intuition before
    a future LLM-powered implementation is connected. The output contract is
    structured so callers can use the metaphor, explanation, and caveat
    separately in generated copy.
    """
    normalized_concept = " ".join(concept.split())
    if not normalized_concept:
        raise ValueError("concept cannot be empty")

    concept_lower = normalized_concept.lower()
    template = _select_template(concept_lower)

    return {
        "concept": normalized_concept,
        "audience": audience,
        "metaphor": template["metaphor"].format(concept=normalized_concept),
        "intuition": template["intuition"].format(concept=normalized_concept),
        "caveat": template["caveat"],
    }


def get_metaphor_tool() -> AgentTool:
    """Return Michel's custom metaphor-generation tool."""
    return AgentTool(
        name="create_metaphor_tool",
        description=(
            "Generate an intuitive metaphor and caveat for a mathematical concept."
        ),
        function=create_metaphor_tool,
        required_parameters=["concept"],
    )


def _select_template(concept_lower: str) -> Dict[str, str]:
    """Select a metaphor template using lightweight concept keywords."""
    if _matches(concept_lower, "random", "probability", "stochastic", "markov"):
        return {
            "metaphor": (
                "{concept} is like watching many possible paths through a city "
                "and asking which neighborhoods the paths tend to visit."
            ),
            "intuition": (
                "The key idea behind {concept} is to reason about patterns that "
                "remain visible even when individual outcomes are uncertain."
            ),
            "caveat": (
                "The metaphor hides the formal assumptions that make probabilistic "
                "statements precise."
            ),
        }

    if _matches(concept_lower, "geometry", "manifold", "curvature", "riemannian"):
        return {
            "metaphor": (
                "{concept} is like learning the shape of a landscape by walking on "
                "it and measuring how straight paths bend."
            ),
            "intuition": (
                "{concept} turns local measurements into information about the "
                "larger shape of a mathematical space."
            ),
            "caveat": (
                "Real geometric definitions depend on precise structures, not only "
                "on visual shape."
            ),
        }

    if _matches(concept_lower, "algebra", "group", "ring", "module", "symmetry"):
        return {
            "metaphor": (
                "{concept} is like a grammar for transformations: it records which "
                "moves are allowed and how they combine."
            ),
            "intuition": (
                "{concept} helps expose the hidden rules that stay stable when "
                "objects are rearranged or combined."
            ),
            "caveat": (
                "The metaphor simplifies the axioms that distinguish different "
                "algebraic structures."
            ),
        }

    if _matches(concept_lower, "learning", "model", "neural", "embedding"):
        return {
            "metaphor": (
                "{concept} is like tuning an instrument by repeatedly listening to "
                "how far the note is from the target."
            ),
            "intuition": (
                "{concept} improves through feedback, gradually shaping a useful "
                "representation of data."
            ),
            "caveat": (
                "The metaphor leaves out optimization details and the risk of "
                "learning misleading patterns."
            ),
        }

    return {
        "metaphor": (
            "{concept} is like building a map: the map leaves out details, but it "
            "keeps the relationships needed to navigate."
        ),
        "intuition": (
            "{concept} gives researchers a simpler structure for seeing what is "
            "essential in a complicated problem."
        ),
        "caveat": (
            "The metaphor is only a first intuition; the exact result still depends "
            "on the formal definitions."
        ),
    }


def _matches(text: str, *keywords: str) -> bool:
    """Return True when any keyword appears as a word-like token."""
    return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords)
