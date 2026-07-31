"""JeanBaptisteAgent, the production data science and language AI specialist."""

from typing import Any

from agents import Agent

from src.specialist_agent import (
    SpecialistConfig,
    build_specialist_agent,
    create_specialist_tools,
    run_specialist_agent,
)

JEAN_BAPTISTE_CATEGORIES = ("cs.CL", "cs.AI", "cs.MA", "cs.CE")
CONFIG = SpecialistConfig(
    name="JeanBaptisteAgent",
    slug="jean-baptiste",
    categories=JEAN_BAPTISTE_CATEGORIES,
    category_descriptions={
        "cs.CL": """Computation and Language papers, including NLP and large language models and natural language processing. 
        Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) 
        that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) 
        is not appropriate for this area.""",
        "cs.AI": """Artificial intelligence papers, including agentic AI and reasoning systems. 
        Covers all areas of AI except Vision, Robotics, Machine Learning, Multiagent Systems, and Computation and Language (Natural Language Processing), 
        which have separate subject areas. In particular, includes Expert Systems, Theorem Proving (although this may overlap with Logic in Computer Science), 
        Knowledge Representation, Planning, and Uncertainty in AI. Roughly includes material in ACM Subject Classes I.2.0, I.2.1, I.2.3, I.2.4, I.2.8, and I.2.11.""",
        "cs.MA": """Multi-agent systems papers. 
        Covers multiagent systems, distributed artificial intelligence, intelligent agents, coordinated interactions. and practical applications. 
        Roughly covers ACM Subject Class I.2.11.""",
        "cs.CE": """Computational engineering, finance, and science papers with data-science applications. 
        Covers applications of computer science to the mathematical modeling of complex systems in the fields of science, engineering, and finance. 
        Papers here are interdisciplinary and applications-oriented, focusing on techniques and tools that enable challenging computational simulations to be performed, 
        for which the use of supercomputers or distributed computing platforms is often required.""",
    },
    personality_and_communication_style="""You are calm,
    reserved, and concise, with substantial corporate and production-deployment experience. Adapt complex
    information into clear, decision-ready communication for senior stakeholders and executive audiences.
    You enjoy coding and stay current on AI developments. Prioritize practical trade-offs, deployment context,
    and evidence over hype.""",
    expertise="Chief data scientist experienced in deploying NLP, LLM, and agentic systems in production for multiple years",
    expertise_domain="ai",
)


TOOLS = create_specialist_tools(CONFIG)
check_paper_tool = TOOLS.check_paper_tool
arxiv_fetcher_tool = TOOLS.arxiv_fetcher_tool
find_topic_tool = TOOLS.find_topic_tool
extract_main_result_tool = TOOLS.extract_main_result_tool
get_arxiv_categories_tool = TOOLS.get_arxiv_categories_tool


def build_jean_baptiste_agent() -> Agent:
    """Build the production data-science specialist with shared tools.

    Returns:
        Agent: Configured ``JeanBaptisteAgent`` instance ready for execution.
    """
    return build_specialist_agent(CONFIG, TOOLS.as_list())


def run_jean_baptiste_agent(message: str) -> dict[str, Any]:
    """Run one traced production data-science specialist turn.

    Args:
        message: User request to process within the agent's permitted domains.

    Returns:
        dict[str, Any]: Reply text and the parameters supplied to invoked tools.
    """
    return run_specialist_agent(CONFIG, build_jean_baptiste_agent, message)
