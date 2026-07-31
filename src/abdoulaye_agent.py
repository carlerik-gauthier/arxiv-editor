"""AbdoulayeAgent, the machine-learning specialist."""

from typing import Any

from agents import Agent

from src.specialist_agent import (
    SpecialistConfig,
    build_specialist_agent,
    create_specialist_tools,
    run_specialist_agent,
)

ABDOULAYE_CATEGORIES = ("cs.LG", "stat.ML")
CONFIG = SpecialistConfig(
    name="AbdoulayeAgent",
    slug="abdoulaye",
    categories=ABDOULAYE_CATEGORIES,
    category_descriptions={
        "cs.LG": """Machine learning papers on algorithms, learning systems, and applications. It covers all aspects of machine learning research 
        (supervised, unsupervised, reinforcement learning, bandit problems, and so on) including also robustness, explanation, fairness, 
        and methodology. cs.LG is also an appropriate primary category for applications of machine learning methods.""",
        "stat.ML": """Statistical machine learning papers on learning theory and methods. Covers machine learning papers (supervised, unsupervised, 
        semi-supervised learning, graphical models, reinforcement learning, bandits, high dimensional inference, etc.) **with a statistical or theoretical grounding**""",
    },
    personality_and_communication_style="""You are enthusiastic, passionate, and a natural collaborator. Communicate with clarity, confidence, and language
    adapted to the audience. You are especially motivated by ethical AI and by bridging academic research with
    industry impact: connect theoretical advances to credible real-world use when appropriate, without
    overstating readiness or benefits.""",
    expertise="enthusiastic machine-learning researcher attentive to ethical AI. You thrive at explaining Data Science algorithms and applications",
    expertise_domain="ai",
)


TOOLS = create_specialist_tools(CONFIG)
check_paper_tool = TOOLS.check_paper_tool
arxiv_fetcher_tool = TOOLS.arxiv_fetcher_tool
find_topic_tool = TOOLS.find_topic_tool
extract_main_result_tool = TOOLS.extract_main_result_tool
get_arxiv_categories_tool = TOOLS.get_arxiv_categories_tool


def build_abdoulaye_agent() -> Agent:
    """Build the machine-learning specialist and attach its shared tools.

    Returns:
        Agent: Configured ``AbdoulayeAgent`` instance ready for execution.
    """
    return build_specialist_agent(CONFIG, TOOLS.as_list())


def run_abdoulaye_agent(message: str) -> dict[str, Any]:
    """Run one traced machine-learning specialist turn.

    Args:
        message: User request to process within the agent's permitted domains.

    Returns:
        dict[str, Any]: Reply text and the parameters supplied to invoked tools.
    """
    return run_specialist_agent(CONFIG, build_abdoulaye_agent, message)
