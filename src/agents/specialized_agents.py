"""Specialized research agents with domain-specific prompts and tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Type

from src.agents.base_agent import AgentTool, BaseAgent
from src.agents.tools.base_tools import get_base_tools
from src.agents.tools.metaphor_tool import get_metaphor_tool


@dataclass(frozen=True)
class AgentProfile:
    """Static configuration used to initialize a specialized research agent."""

    name: str
    expertise: str
    categories: List[str]
    role_focus: str
    communication_style: str


class SpecializedAgent(BaseAgent):
    """
    Base class for domain-specific agents in the ArXiv editor workflow.

    Subclasses provide an AgentProfile. This class turns that profile into a
    consistent system prompt and attaches the default phase-3 research tools.
    """

    profile: AgentProfile
    extra_tools: Iterable[AgentTool] = ()

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tools: Optional[Iterable[AgentTool]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        configured_tools = list(tools) if tools is not None else self._default_tools()
        super().__init__(
            name=self.profile.name,
            expertise=self.profile.expertise,
            categories=list(self.profile.categories),
            llm_client=llm_client,
            system_prompt=system_prompt or self.build_system_prompt(self.profile),
            tools=configured_tools,
        )

    @classmethod
    def build_system_prompt(cls, profile: AgentProfile) -> str:
        """Build the domain-specific system prompt for a specialist agent."""
        categories = ", ".join(profile.categories)
        return (
            f"You are {profile.name}, a specialized ArXiv research agent.\n"
            f"Expertise: {profile.expertise}.\n"
            f"Assigned ArXiv categories: {categories}.\n"
            f"Role focus: {profile.role_focus}\n"
            f"Communication style: {profile.communication_style}\n\n"
            "When Julius delegates a task, work within your domain unless the task "
            "explicitly asks for cross-domain comparison. Use tools to fetch papers, "
            "check whether enough papers were collected, analyze full paper text, "
            "and summarize representative papers. Report uncertainty, missing data, "
            "and date-range extension needs clearly."
        )

    @classmethod
    def _default_tools(cls) -> List[AgentTool]:
        """Return base research tools plus any subclass-specific tools."""
        return [*get_base_tools(), *cls.extra_tools]


class MichelAgent(SpecializedAgent):
    """Mathematics education agent focused on intuition and accessibility."""

    profile = AgentProfile(
        name="Michel",
        expertise="Mathematics education, mathematical intuition, and exposition",
        categories=["math.HO", "math.GM"],
        role_focus=(
            "Explain complex mathematical ideas to non-experts while preserving "
            "the core expert meaning."
        ),
        communication_style=(
            "Use concrete intuition, carefully bounded metaphors, and plain "
            "language before technical terminology."
        ),
    )
    extra_tools = (get_metaphor_tool(),)


class ChrisAgent(SpecializedAgent):
    """Probability specialist for stochastic processes and statistics theory."""

    profile = AgentProfile(
        name="Chris",
        expertise="Probability theory, stochastic processes, and statistics theory",
        categories=["math.PR", "stat.TH"],
        role_focus=(
            "Identify probabilistic models, limiting behavior, assumptions, and "
            "the significance of stochastic results."
        ),
        communication_style=(
            "Be precise about hypotheses and explain how random mechanisms drive "
            "the main theorem or application."
        ),
    )


class AlainAgent(SpecializedAgent):
    """Algebra specialist for structures, symmetry, and classification."""

    profile = AgentProfile(
        name="Alain",
        expertise="Algebraic geometry, rings and algebras, and group theory",
        categories=["math.AG", "math.RA", "math.GR"],
        role_focus=(
            "Surface the algebraic structures, invariants, symmetries, and "
            "classification problems behind new papers."
        ),
        communication_style=(
            "Connect abstract definitions to the structural question they answer, "
            "with careful attention to exact assumptions."
        ),
    )


class BrunoAgent(SpecializedAgent):
    """Geometry specialist for differential, spectral, and Riemannian geometry."""

    profile = AgentProfile(
        name="Bruno",
        expertise="Spectral geometry, Riemannian geometry, and differential geometry",
        categories=["math.DG", "math.SP"],
        role_focus=(
            "Explain how geometric shape, curvature, spectra, and analytic "
            "operators interact in the selected research."
        ),
        communication_style=(
            "Emphasize geometric intuition, then state the rigorous object or "
            "result that supports it."
        ),
    )


class ElisaAgent(SpecializedAgent):
    """Applied mathematics and cryptography specialist."""

    profile = AgentProfile(
        name="Elisa",
        expertise="Applied mathematics, cryptography, and optimization",
        categories=["cs.CR", "math.OC"],
        role_focus=(
            "Assess practical security, algorithmic tradeoffs, optimization "
            "methods, and real-world relevance."
        ),
        communication_style=(
            "Balance mathematical rigor with implementation and security impact, "
            "especially when claims have applied consequences."
        ),
    )


class FelixAgent(SpecializedAgent):
    """Dynamical systems and symplectic geometry specialist."""

    profile = AgentProfile(
        name="Felix",
        expertise="Dynamical systems, long-term behavior, and symplectic geometry",
        categories=["math.DS", "math.SG"],
        role_focus=(
            "Track how systems evolve, which invariants constrain them, and what "
            "long-term or symplectic phenomena the papers reveal."
        ),
        communication_style=(
            "Explain the motion or phase-space picture first, then connect it to "
            "the formal theorem."
        ),
    )


class AbdoulayeAgent(SpecializedAgent):
    """Machine learning specialist for algorithms, models, and applications."""

    profile = AgentProfile(
        name="Abdoulaye",
        expertise="Machine learning research, statistical learning, and applications",
        categories=["cs.LG", "stat.ML"],
        role_focus=(
            "Evaluate learning algorithms, experimental claims, benchmarks, and "
            "the methodological contribution of ML papers."
        ),
        communication_style=(
            "Explain the model or algorithm in operational terms and distinguish "
            "empirical evidence from theoretical guarantees."
        ),
    )


class JeanBaptisteAgent(SpecializedAgent):
    """Data science, NLP, LLM, and agentic AI specialist."""

    profile = AgentProfile(
        name="JeanBaptiste",
        expertise="Data science, NLP, large language models, and agentic AI systems",
        categories=["cs.CL", "cs.AI", "cs.MA", "cs.CE"],
        role_focus=(
            "Analyze language, agent, and data-centric AI papers with attention to "
            "evaluation design, system behavior, and deployment implications."
        ),
        communication_style=(
            "Be concrete about tasks, data, metrics, model behavior, and practical "
            "limits of the reported results."
        ),
    )


SPECIALIZED_AGENT_CLASSES: Dict[str, Type[SpecializedAgent]] = {
    "michel": MichelAgent,
    "chris": ChrisAgent,
    "alain": AlainAgent,
    "bruno": BrunoAgent,
    "elisa": ElisaAgent,
    "felix": FelixAgent,
    "abdoulaye": AbdoulayeAgent,
    "jeanbaptiste": JeanBaptisteAgent,
}


def create_specialized_agent(
    agent_name: str,
    llm_client: Optional[Any] = None,
) -> SpecializedAgent:
    """
    Instantiate a specialized agent by name.

    The lookup is case-insensitive and ignores spaces, hyphens, and underscores
    so callers can use values such as "Jean Baptiste" or "jean_baptiste".
    """
    normalized_name = _normalize_agent_name(agent_name)
    agent_class = SPECIALIZED_AGENT_CLASSES.get(normalized_name)
    if agent_class is None:
        available = ", ".join(sorted(SPECIALIZED_AGENT_CLASSES))
        raise ValueError(f"Unknown specialized agent '{agent_name}'. Available: {available}")
    return agent_class(llm_client=llm_client)


def create_all_specialized_agents(
    llm_client: Optional[Any] = None,
) -> List[SpecializedAgent]:
    """Instantiate all phase-3.2 specialized agents in a stable order."""
    return [
        agent_class(llm_client=llm_client)
        for agent_class in SPECIALIZED_AGENT_CLASSES.values()
    ]


def _normalize_agent_name(agent_name: str) -> str:
    """Normalize user-facing agent names for factory lookup."""
    return "".join(character for character in agent_name.lower() if character.isalnum())
