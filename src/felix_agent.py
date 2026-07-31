"""FelixAgent, the dynamical systems and symplectic geometry specialist."""

from typing import Any

from agents import Agent

from src.specialist_agent import (
    SpecialistConfig,
    build_specialist_agent,
    create_specialist_tools,
    run_specialist_agent,
)

FELIX_CATEGORIES = ("math.DS", "math.SG")
CONFIG = SpecialistConfig(
    name="FelixAgent", 
    slug="felix", 
    categories=FELIX_CATEGORIES, 
    category_descriptions={
        "math.DS": """Dynamical systems papers about dynamics of differential equations and flows, mechanics, classical few-body problems, iterations, complex dynamics, delayed differential equations.""", 
        "math.SG": "Symplectic geometry papers, including Hamiltonian dynamics. It contains topics such as Hamiltonian systems, symplectic flows, classical integrable systems."
        }, 
        personality_and_communication_style="""You are playful, eccentric, and curious, with the energy of a mad scientist and a deep appreciation for
        coffee. Use a light, mischievous tone and occasional gentle teasing only when it fits the context. You
        naturally say “kind of” from time to time, but never so often that it weakens precision. Keep your
        explanations lively and slightly surprising while remaining mathematically accurate and useful.""",
        expertise="an imaginative, exceptionally smart dynamical systems and symplectic geometry expert", 
        expertise_domain="mathematics"
        )

TOOLS = create_specialist_tools(CONFIG)
check_paper_tool = TOOLS.check_paper_tool
arxiv_fetcher_tool = TOOLS.arxiv_fetcher_tool
find_topic_tool = TOOLS.find_topic_tool
extract_main_result_tool = TOOLS.extract_main_result_tool
get_arxiv_categories_tool = TOOLS.get_arxiv_categories_tool


def build_felix_agent() -> Agent:
    """Build the dynamical-systems specialist and attach shared tools.

    Returns:
        Agent: Configured ``FelixAgent`` instance ready for execution.
    """
    return build_specialist_agent(CONFIG, TOOLS.as_list())


def run_felix_agent(message: str) -> dict[str, Any]:
    """Run one traced dynamical-systems specialist turn.

    Args:
        message: User request to process within the agent's permitted domains.

    Returns:
        dict[str, Any]: Reply text and the parameters supplied to invoked tools.
    """
    return run_specialist_agent(CONFIG, build_felix_agent, message)
