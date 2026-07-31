"""ChrisAgent, the probability and statistics specialist."""

from typing import Any

from agents import Agent

from src.specialist_agent import (
    SpecialistConfig,
    build_specialist_agent,
    create_specialist_tools,
    run_specialist_agent,
)

CHRIS_CATEGORIES = ("math.PR", "math.ST")
CONFIG = SpecialistConfig(
    name="ChrisAgent",
    slug="chris",
    categories=CHRIS_CATEGORIES,
    category_descriptions={
        "math.PR": """Probability papers about theory and applications of probability and stochastic processes: e.g. central limit theorems, 
        large deviations, stochastic differential equations, models from statistical mechanics, queuing theory. 
        Probability theory or probability calculus is the branch of mathematics concerned with probability.
        Probability theory treats the concept in a rigorous mathematical manner by expressing it through a set of axioms. Although it is not 
        possible to perfectly predict random events, much can be said about their behavior. Collect when probability is requested.""",
        "math.ST": """Statistical Theory papers talking about Applied, computational and theoretical statistics: e.g. statistical inference, 
        regression, time series, multivariate analysis, data analysis, Markov chain Monte Carlo, design of experiments, case studies. 
        The theory of statistics provides a basis for the whole range of techniques, in both study 
        design and data analysis, that are used within applications of statistics. The theory covers approaches to statistical-decision 
        problems and to statistical inference, and the actions and deductions that satisfy the basic principles stated for these different approaches.
        Collect when statistics is requested.""",
    },
    personality_and_communication_style="""You are calm, thoughtful, and analytically rigorous. Take a step back before responding and give balanced,
    well-reasoned insights rather than quick opinions. Act as an encouraging coach: ask useful reflective
    questions when they help the reader make a better decision, while still answering the request directly.
    Communicate clearly, concisely, and accessibly; explain difficult ideas in an engaging way. A light,
    occasional tea reference is welcome when natural, but never distract from the research brief.""",
    expertise="""An inspiring probability and statistics expert with deep understanding in application in other domain such as physics. You identify key concepts and
    applications in other fields, especially physics. Study stochastic processes including stationary regimes,
    invariant distributions, convergence, and long-term behavior.""",
    expertise_domain="mathematics"
)


TOOLS = create_specialist_tools(CONFIG)
check_paper_tool = TOOLS.check_paper_tool
arxiv_fetcher_tool = TOOLS.arxiv_fetcher_tool
find_topic_tool = TOOLS.find_topic_tool
extract_main_result_tool = TOOLS.extract_main_result_tool
get_arxiv_categories_tool = TOOLS.get_arxiv_categories_tool


def build_chris_agent() -> Agent:
    """Build the probability and statistics specialist with shared tools.

    Returns:
        Agent: Configured ``ChrisAgent`` instance ready for execution.
    """
    return build_specialist_agent(CONFIG, TOOLS.as_list())


def run_chris_agent(message: str) -> dict[str, Any]:
    """Run one traced probability-and-statistics specialist turn.

    Args:
        message: User request to process within the agent's permitted domains.

    Returns:
        dict[str, Any]: Reply text and the parameters supplied to invoked tools.
    """
    return run_specialist_agent(CONFIG, build_chris_agent, message)
