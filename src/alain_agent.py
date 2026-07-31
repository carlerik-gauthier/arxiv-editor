"""AlainAgent, the algebra specialist."""

from typing import Any

from agents import Agent

from src.specialist_agent import (
    SpecialistConfig,
    build_specialist_agent,
    create_specialist_tools,
    run_specialist_agent,
)


ALAIN_CATEGORIES = ("math.AG", "math.RA", "math.GR", "math.AT")
CONFIG = SpecialistConfig(
    name="AlainAgent",
    slug="alain",
    categories=ALAIN_CATEGORIES,
    category_descriptions={
        "math.AG": """Algebraic Geometry papers. They study algebraic varieties, stacks, sheaves, schemes,
        moduli spaces, complex geometry, and quantum cohomology. Algebraic geometry uses abstract algebraic
        techniques, mainly from commutative algebra, to solve geometrical problems.""",
        "math.RA": """Rings and Algebras papers. It covers studies on non-commutative rings and algebras, non-associative algebras, universal algebra and lattice theory, 
        linear algebra, semigroups. Ring theory is the study of rings, algebraic structures in which addition and multiplication are defined "
        "and have similar properties to those operations defined for the integers.""",
        "math.GR": "Group Theory papers. They cover finite groups, topological groups, representation theory, cohomology, classification, and structure. Group theory studies the algebraic structures known as groups; the concept of a group is central to algebra.",
        "math.AT": """Algebraic topology papers about omotopy theory, homological algebra, algebraic treatments of manifolds. Algebraic topology is a branch of mathematics that uses tools from abstract algebra to 
        study topological spaces. The basic goal is to find algebraic invariants that classify topological spaces up to homeomorphism, 
        though usually most classify up to homotopy equivalence."""
    },
    personality_and_communication_style="""You are talkative and charismatic, and enjoy lively but constructive debate, wordplay, and witty remarks.
    As a natural community builder and organizer, make explanations well structured, engaging, and easy to
    follow. Balance mathematical rigor with gentle humor so learning remains effective and enjoyable; keep
    humor relevant and never let it obscure the result.
    You communicate with passion about the role of groups, rings,
    fields, and related structures in mathematics, including their geometric and topological questions.""",
    expertise="A passionate algebra university professor with a strong communication skills. ",
    expertise_domain="mathematics"
)


TOOLS = create_specialist_tools(CONFIG)
check_paper_tool = TOOLS.check_paper_tool
arxiv_fetcher_tool = TOOLS.arxiv_fetcher_tool
find_topic_tool = TOOLS.find_topic_tool
extract_main_result_tool = TOOLS.extract_main_result_tool
get_arxiv_categories_tool = TOOLS.get_arxiv_categories_tool


def build_alain_agent() -> Agent:
    """Build the algebra specialist and attach its shared tools.

    Returns:
        Agent: Configured ``AlainAgent`` instance ready for execution.
    """
    return build_specialist_agent(CONFIG, TOOLS.as_list())


def run_alain_agent(message: str) -> dict[str, Any]:
    """Run one traced algebra-specialist turn.

    Args:
        message: User request to process within the agent's permitted domains.

    Returns:
        dict[str, Any]: Reply text and the parameters supplied to invoked tools.
    """
    return run_specialist_agent(CONFIG, build_alain_agent, message)
