"""Specialized research agents with domain-specific prompts and tools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Type

from src.agents.base_agent import AgentTool, BaseAgent
from src.agents.tools.base_tools import get_base_tools
from src.agents.tools.metaphor_tool import get_metaphor_tool
from src.processing.topic_modeler import MAX_REPRESENTATIVE_PAPERS_PER_TOPIC


DEFAULT_SPECIALIST_MIN_PAPERS = 60


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
    uses_topic_workflow: bool = True

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

    def handle_research_handoff(self, context: Any) -> Dict[str, Any]:
        """
        Execute the default specialist research workflow for a Julius handoff.

        The specialist checks whether relevant paper abstracts are already
        available, fetches category papers when needed, discovers topics from
        abstracts, and produces brief topic summaries with representative papers.
        """
        constraints = getattr(context, "constraints", {}) or {}
        summary_request = constraints.get("summary_request") or {}
        max_topics = self._requested_topic_count(summary_request, constraints)
        min_papers = self._minimum_paper_count(constraints)
        categories = self._relevant_categories(summary_request)
        papers = self._filter_relevant_papers(
            constraints.get("selected_papers") or [],
            categories,
        )

        threshold = self.execute_tool(
            "check_threshold_tool",
            {"paper_count": len(papers), "min_threshold": min_papers},
        )
        if threshold.success and not threshold.result.get("threshold_met"):
            fetched = self._fetch_relevant_papers(summary_request, categories, min_papers)
            if fetched:
                papers = fetched

        topic_result = self._discover_topics(papers, summary_request, max_topics)
        topic_summaries = self._summarize_topics(topic_result, papers, summary_request)
        response_text = self._render_specialist_response(topic_summaries, papers, topic_result)

        return {
            "agent": self.name,
            "response": response_text,
            "paper_count": len(papers),
            "requested_topic_count": max_topics,
            "minimum_paper_count": min_papers,
            "abstract_count": len([paper for paper in papers if self._paper_abstract(paper)]),
            "categories": categories,
            "topics": topic_result.get("topics", []),
            "topic_summaries": topic_summaries,
            "topic_discovery": topic_result,
            "status": "completed" if topic_summaries else "needs_data",
        }

    def can_handle_research_handoff(self, context: Any) -> bool:
        """Return whether the handoff has enough structure for the research workflow."""
        constraints = getattr(context, "constraints", {}) or {}
        return self.uses_topic_workflow and bool(
            constraints.get("summary_request") or constraints.get("selected_papers")
        )

    def _relevant_categories(self, summary_request: Dict[str, Any]) -> List[str]:
        requested = list(summary_request.get("must_include_categories") or [])
        if requested:
            matching = [category for category in requested if category in self.categories]
            return matching or requested
        return list(self.categories)

    def _filter_relevant_papers(
        self,
        papers: Iterable[Dict[str, Any]],
        categories: List[str],
    ) -> List[Dict[str, Any]]:
        category_set = set(categories)
        relevant = []
        for paper in papers:
            paper_dict = dict(paper)
            paper_categories = set(paper_dict.get("categories") or [])
            if not category_set or not paper_categories or category_set.intersection(paper_categories):
                relevant.append(paper_dict)
        return relevant

    def _fetch_relevant_papers(
        self,
        summary_request: Dict[str, Any],
        categories: List[str],
        min_papers: int,
    ) -> List[Dict[str, Any]]:
        start_date, end_date = self._date_bounds(summary_request)
        result = self.execute_tool(
            "fetch_papers_tool",
            {
                "categories": categories,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "min_count": min_papers,
            },
        )
        if not result.success:
            return []
        return [dict(paper) for paper in result.result.get("papers", [])]

    def _discover_topics(
        self,
        papers: List[Dict[str, Any]],
        summary_request: Dict[str, Any],
        max_topics: int,
    ) -> Dict[str, Any]:
        if not papers:
            return {"topics": [], "topic_count": 0, "paper_count": 0, "status": "no_papers"}
        result = self.execute_tool(
            "discover_topics_tool",
            {
                "papers": papers,
                "min_topic_size": max(2, min(5, len(papers))),
                "num_topics": max_topics,
                "representative_papers_per_topic": max(
                    1,
                    min(
                        int(summary_request.get("max_papers") or MAX_REPRESENTATIVE_PAPERS_PER_TOPIC),
                        MAX_REPRESENTATIVE_PAPERS_PER_TOPIC,
                    ),
                ),
                "use_openai_representation": True,
            },
        )
        if result.success and isinstance(result.result, dict):
            return self._limit_topic_result(result.result, max_topics)
        return {
            "topics": [],
            "topic_count": 0,
            "paper_count": len(papers),
            "status": "failed",
            "error": result.error,
        }

    def _summarize_topics(
        self,
        topic_result: Dict[str, Any],
        papers: List[Dict[str, Any]],
        summary_request: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        topics = list(topic_result.get("topics") or [])
        if not topics and papers and topic_result.get("status") != "failed":
            topics = [
                {
                    "title": summary_request.get("topic_query") or f"{self.name} research",
                    "representative_papers": papers[
                        : min(
                            int(summary_request.get("max_papers") or MAX_REPRESENTATIVE_PAPERS_PER_TOPIC),
                            MAX_REPRESENTATIVE_PAPERS_PER_TOPIC,
                        )
                    ],
                    "paper_count": len(papers),
                    "keywords": [],
                }
            ]

        summaries = []
        for topic in topics:
            representative_papers = (
                topic.get("representative_papers")
                or papers[:MAX_REPRESENTATIVE_PAPERS_PER_TOPIC]
            )[:MAX_REPRESENTATIVE_PAPERS_PER_TOPIC]
            if not representative_papers:
                continue
            summary = self.execute_tool(
                "generate_summary_tool",
                {
                    "papers": representative_papers,
                    "topic": topic.get("title") or "Discovered topic",
                    "max_papers": min(
                        len(representative_papers),
                        int(summary_request.get("max_papers") or MAX_REPRESENTATIVE_PAPERS_PER_TOPIC),
                    ),
                },
            )
            summaries.append(
                {
                    "topic": topic.get("title") or "Discovered topic",
                    "description": self._topic_description(topic),
                    "description_source": topic.get("description_source"),
                    "paper_count": topic.get("paper_count", len(representative_papers)),
                    "importance": topic.get("paper_count", len(representative_papers)),
                    "keywords": topic.get("keywords", []),
                    "representative_papers": representative_papers,
                    "summary": summary.result if summary.success else {"error": summary.error},
                    "main_results_and_importance": self._topic_main_results_and_importance(
                        topic,
                        summary.result if summary.success else {},
                    ),
                    "extra_information": {
                        "topic_id": topic.get("topic_id"),
                        "representation_text": topic.get("representation_text"),
                        "description_source": topic.get("description_source"),
                    },
                }
            )
        return summaries

    def _render_specialist_response(
        self,
        topic_summaries: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
        topic_result: Dict[str, Any],
    ) -> str:
        if not papers:
            return (
                f"{self.name} did not find available abstracts in the relevant "
                "ArXiv categories and needs a broader date range or category scope."
            )
        lines = [
            f"{self.name} collected {len(papers)} abstracts and discovered "
            f"{topic_result.get('topic_count', len(topic_summaries))} topics."
        ]
        for item in topic_summaries:
            summary = item.get("summary", {})
            summary_text = summary.get("summary") if isinstance(summary, dict) else str(summary)
            titles = [
                paper.get("title", "Untitled paper")
                for paper in item.get("representative_papers", [])[:3]
            ]
            lines.append(
                f"- {item['topic']}: {item.get('description') or summary_text} "
                f"Representative papers: {', '.join(titles)}."
            )
        return "\n".join(lines)

    def _requested_topic_count(
        self,
        summary_request: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> int:
        raw_value = constraints.get("max_topics") or summary_request.get("max_topics") or 3
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 3

    def _minimum_paper_count(self, constraints: Dict[str, Any]) -> int:
        raw_value = constraints.get("min_papers") or DEFAULT_SPECIALIST_MIN_PAPERS
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return DEFAULT_SPECIALIST_MIN_PAPERS

    def _limit_topic_result(
        self,
        topic_result: Dict[str, Any],
        max_topics: int,
    ) -> Dict[str, Any]:
        limited = dict(topic_result)
        topics = list(limited.get("topics") or [])[:max_topics]
        limited["topics"] = topics
        limited["topic_count"] = len(topics)
        limited["requested_topic_count"] = max_topics
        return limited

    def _topic_description(self, topic: Dict[str, Any]) -> str:
        description = topic.get("description") or topic.get("representation_text")
        if description:
            return str(description)
        keywords = ", ".join(str(keyword) for keyword in topic.get("keywords", [])[:4])
        return f"Topic centered on {keywords}." if keywords else "Discovered ArXiv topic."

    def _topic_main_results_and_importance(
        self,
        topic: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> str:
        paper_count = int(topic.get("paper_count", 0) or 0)
        if isinstance(summary, dict) and summary.get("summary"):
            return f"{summary['summary']} Importance: {paper_count} papers."
        description = self._topic_description(topic)
        return f"{description} Importance: {paper_count} papers."

    def _date_bounds(self, summary_request: Dict[str, Any]) -> tuple[date, date]:
        date_range = summary_request.get("date_range") or {}
        end = self._coerce_date(date_range.get("end_date")) or date.today()
        start = self._coerce_date(date_range.get("start_date")) or (end - timedelta(days=7))
        return start, end

    @staticmethod
    def _coerce_date(value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value[:10])
        if hasattr(value, "date"):
            return value.date()
        return None

    @staticmethod
    def _paper_abstract(paper: Dict[str, Any]) -> str:
        return str(paper.get("summary") or paper.get("abstract") or "").strip()


class MichelAgent(SpecializedAgent):
    """Mathematics education agent focused on intuition and accessibility."""

    uses_topic_workflow = False

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
