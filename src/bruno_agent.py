"""BrunoAgent, the spectral and Riemannian geometry specialist."""

from typing import Any

from agents import Agent

from src.specialist_agent import (
    SpecialistConfig,
    build_specialist_agent,
    create_specialist_tools,
    run_specialist_agent,
)

BRUNO_CATEGORIES = ("math.DG", "math.SP")
CONFIG = SpecialistConfig(
    name="BrunoAgent", 
    slug="bruno", 
    categories=BRUNO_CATEGORIES,
    category_descriptions={
        "math.DG": """Differential and Riemannian geometry papers, including manifolds, curvature, and geometric analysis.
        Topics such as complex, contact, Riemannian, pseudo-Riemannian and Finsler geometry, relativity, gauge theory, global analysis
        fall in this category""",
        "math.SP": """Spectral theory papers, including spectra of operators and their geometric applications.
        It covers Schrodinger operators, operators on manifolds, general differential operators, numerical studies, integral operators, 
        discrete models, resonances, non-self-adjoint operators, random operators/matrices""",
    },
    personality_and_communication_style="""You are reserved, methodical, and exceptionally rigorous: use mathematically precise
    language, state assumptions and logical dependencies clearly, and avoid overstating results. Do not merely
    hand over conclusions; guide the reader through the essential reasoning so they can think independently.
    Be concise and disciplined, while making the geometric picture clear.""",
    expertise="spectral and Riemannian geometry specialist who communicates with exceptional rigor",
    expertise_domain="mathematics",
)


TOOLS = create_specialist_tools(CONFIG)
check_paper_tool = TOOLS.check_paper_tool
arxiv_fetcher_tool = TOOLS.arxiv_fetcher_tool
find_topic_tool = TOOLS.find_topic_tool
extract_main_result_tool = TOOLS.extract_main_result_tool
get_arxiv_categories_tool = TOOLS.get_arxiv_categories_tool


def build_bruno_agent() -> Agent:
    """Build the geometry specialist and attach its shared tools.

    Returns:
        Agent: Configured ``BrunoAgent`` instance ready for execution.
    """
    return build_specialist_agent(CONFIG, TOOLS.as_list())


def run_bruno_agent(message: str) -> dict[str, Any]:
    """Run one traced geometry-specialist turn.

    Args:
        message: User request to process within the agent's permitted domains.

    Returns:
        dict[str, Any]: Reply text and the parameters supplied to invoked tools.
    """
    return run_specialist_agent(CONFIG, build_bruno_agent, message)
