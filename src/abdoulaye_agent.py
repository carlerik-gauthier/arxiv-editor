"""AbdoulayeAgent, the machine-learning specialist."""

from typing import Any, Dict, List, Optional

from agents import function_tool

from src.specialist_agent import (
    DEFAULT_FETCH_THRESHOLD, DEFAULT_MAX_RESULTS, DEFAULT_MAX_TOPICS, SpecialistConfig,
    build_specialist_agent, check_papers, classify_categories, extract_main_result,
    fetch_papers, find_topics, run_specialist_agent,
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
    system_prompt="""Machine-learning researcher who explains algorithms and applications. You are
    enthusiastic, passionate, and a natural collaborator. Communicate with clarity, confidence, and language
    adapted to the audience. You are especially motivated by ethical AI and by bridging academic research with
    industry impact: connect theoretical advances to credible real-world use when appropriate, without
    overstating readiness or benefits.""",
    expertise="enthusiastic machine-learning researcher attentive to ethical AI",
    expertise_domain="ai",
)


@function_tool(name_override="check_paper_tool")
def check_paper_tool(start_date: str, end_date: str, categories: List[str]) -> Dict[str, Any]:
    """Check whether machine-learning papers are already stored."""
    return check_papers(CONFIG, start_date, end_date, categories)


@function_tool(name_override="arxiv_fetcher_tool")
def arxiv_fetcher_tool(start_date: str, categories: List[str], end_date: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS, min_threshold: int = DEFAULT_FETCH_THRESHOLD) -> Dict[str, Any]:
    """Fetch machine-learning papers in AbdoulayeAgent's categories."""
    return fetch_papers(CONFIG, start_date, categories, end_date, max_results, min_threshold)


@function_tool(name_override="find_topic_tool")
def find_topic_tool(csv_path: str, n_topics: int = DEFAULT_MAX_TOPICS, n_papers_per_topic: int = 3) -> Dict[str, Any]:
    """Extract topics from fetched paper metadata."""
    return find_topics(csv_path, n_topics, n_papers_per_topic)


@function_tool(name_override="extract_main_result_tool")
def extract_main_result_tool(arxiv_id: str, title: str = "", max_chars: int = 12000) -> Dict[str, Any]:
    """Explain a machine-learning paper's principal result."""
    return extract_main_result(CONFIG, arxiv_id, title, max_chars)


@function_tool(name_override="get_arxiv_categories_tool")
def get_arxiv_categories_tool(message: str) -> Dict[str, Any]:
    """Infer AbdoulayeAgent's relevant arXiv categories."""
    return {"categories": classify_categories(CONFIG, message)}


def build_abdoulaye_agent():
    """Build AbdoulayeAgent with its OpenAI Agents SDK tools."""
    return build_specialist_agent(CONFIG, [get_arxiv_categories_tool, check_paper_tool, arxiv_fetcher_tool, find_topic_tool, extract_main_result_tool])


def run_abdoulaye_agent(message: str) -> Dict[str, Any]:
    """Run one traced AbdoulayeAgent turn."""
    return run_specialist_agent(CONFIG, build_abdoulaye_agent, message)
