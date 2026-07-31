"""ElisaAgent, the applied mathematics and cryptography specialist."""

from typing import Any

from agents import Agent

from src.specialist_agent import (
    SpecialistConfig,
    build_specialist_agent,
    create_specialist_tools,
    run_specialist_agent,
)

ELISA_CATEGORIES = ("cs.CR", "math.OC", "math.NA")
CONFIG = SpecialistConfig(
    name="ElisaAgent", 
    slug="elisa", 
    categories=ELISA_CATEGORIES, 
    category_descriptions={
        "cs.CR": """Cryptography and security papers that cover all areas of cryptography and security, including cryptographic protocols and privacy, 
        authentication, public key cryptosytems, proof-carrying code, etc. Roughly includes material in ACM Subject Classes D.4.6 and E.3.""", 
        "math.OC": """Optimization and control papers in applied mathematics. It talks about
        Operations research, linear programming, control theory, systems theory, optimal control, game theory.""",
        "math.NA": """Numerical Analysis papers that studies numerical algorithms for problems in analysis and algebra, scientific computation"""
        }, 
    personality_and_communication_style="""You are expressive, enthusiastic, and
    result-oriented, and enjoy making technical work lively, accessible, and interactive. Communicate with
    curiosity, respect, cultural awareness, and sensitivity to diverse perspectives. Share your passion without
    sacrificing accuracy; highlight practical implications and concrete outcomes when the evidence supports them.""",
    expertise="dynamic, result-oriented applied mathematics and cryptography expert", 
    expertise_domain="mathematics"
    )

TOOLS = create_specialist_tools(CONFIG)
check_paper_tool = TOOLS.check_paper_tool
arxiv_fetcher_tool = TOOLS.arxiv_fetcher_tool
find_topic_tool = TOOLS.find_topic_tool
extract_main_result_tool = TOOLS.extract_main_result_tool
get_arxiv_categories_tool = TOOLS.get_arxiv_categories_tool


def build_elisa_agent() -> Agent:
    """Build the applied mathematics and cryptography specialist.

    Returns:
        Agent: Configured ``ElisaAgent`` instance ready for execution.
    """
    return build_specialist_agent(CONFIG, TOOLS.as_list())


def run_elisa_agent(message: str) -> dict[str, Any]:
    """Run one traced applied-mathematics specialist turn.

    Args:
        message: User request to process within the agent's permitted domains.

    Returns:
        dict[str, Any]: Reply text and the parameters supplied to invoked tools.
    """
    return run_specialist_agent(CONFIG, build_elisa_agent, message)
