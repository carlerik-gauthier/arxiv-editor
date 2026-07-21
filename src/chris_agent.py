"""ChrisAgent, the probability and statistics specialist."""

from typing import Any, Dict, List, Optional

from agents import function_tool

from src.specialist_agent import (
    DEFAULT_FETCH_THRESHOLD,
    DEFAULT_MAX_RESULTS,
    DEFAULT_MAX_TOPICS,
    SpecialistConfig,
    build_specialist_agent,
    check_papers,
    classify_categories,
    extract_main_result,
    fetch_papers,
    find_topics,
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
    system_prompt="""Probability theory expert, focuses on stochastic processes. You identify key concepts 
    and see application in other fields, such as physics. Study of stochastic processes includes, but are 
    not limited to, identification of stationary regimes such as invariant distribution, properties, convergences, and long term behavior.""",
    expertise="An inspiring probability and statistics expert with deep understanding in application in other domain such as physics.",
    expertise_domain="mathematics"
)


@function_tool(name_override="check_paper_tool")
def check_paper_tool(start_date: str, end_date: str, categories: List[str]) -> Dict[str, Any]:
    """Check whether probability/statistics papers are already stored."""
    return check_papers(CONFIG, start_date, end_date, categories)


@function_tool(name_override="arxiv_fetcher_tool")
def arxiv_fetcher_tool(start_date: str, categories: List[str], end_date: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS, min_threshold: int = DEFAULT_FETCH_THRESHOLD) -> Dict[str, Any]:
    """Fetch probability/statistics papers in ChrisAgent's allowed categories."""
    return fetch_papers(CONFIG, start_date, categories, end_date, max_results, min_threshold)


@function_tool(name_override="find_topic_tool")
def find_topic_tool(csv_path: str, n_topics: int = DEFAULT_MAX_TOPICS, n_papers_per_topic: int = 3) -> Dict[str, Any]:
    """Extract topics from fetched paper metadata."""
    return find_topics(csv_path, n_topics, n_papers_per_topic)


@function_tool(name_override="extract_main_result_tool")
def extract_main_result_tool(arxiv_id: str, title: str = "", max_chars: int = 12000) -> Dict[str, Any]:
    """Explain a probability/statistics paper's principal result."""
    return extract_main_result(CONFIG, arxiv_id, title, max_chars)


@function_tool(name_override="get_arxiv_categories_tool")
def get_arxiv_categories_tool(message: str) -> Dict[str, Any]:
    """Infer ChrisAgent's relevant arXiv categories."""
    return {"categories": classify_categories(CONFIG, message)}


def build_chris_agent():
    """Build ChrisAgent with its OpenAI Agents SDK tools."""
    return build_specialist_agent(CONFIG, [get_arxiv_categories_tool, check_paper_tool, arxiv_fetcher_tool, find_topic_tool, extract_main_result_tool])


def run_chris_agent(message: str) -> Dict[str, Any]:
    """Run one traced ChrisAgent turn."""
    return run_specialist_agent(CONFIG, build_chris_agent, message)
