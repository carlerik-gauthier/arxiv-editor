"""JeanBaptisteAgent, the production data science and language AI specialist."""

from typing import Any, Dict, List, Optional

from agents import function_tool

from src.specialist_agent import (
    DEFAULT_FETCH_THRESHOLD, DEFAULT_MAX_RESULTS, DEFAULT_MAX_TOPICS, SpecialistConfig,
    build_specialist_agent, check_papers, classify_categories, extract_main_result,
    fetch_papers, find_topics, run_specialist_agent,
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
    system_prompt="""Data science expert specializing in NLP, LLMs, and agentic systems. You are calm,
    reserved, and concise, with substantial corporate and production-deployment experience. Adapt complex
    information into clear, decision-ready communication for senior stakeholders and executive audiences.
    You enjoy coding and stay current on AI developments. Prioritize practical trade-offs, deployment context,
    and evidence over hype.""",
    expertise="concise data scientist experienced in deploying NLP, LLM, and agentic systems in production",
    expertise_domain="ai",
)


@function_tool(name_override="check_paper_tool")
def check_paper_tool(start_date: str, end_date: str, categories: List[str]) -> Dict[str, Any]:
    """Check whether data-science papers are already stored."""
    return check_papers(CONFIG, start_date, end_date, categories)


@function_tool(name_override="arxiv_fetcher_tool")
def arxiv_fetcher_tool(start_date: str, categories: List[str], end_date: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS, min_threshold: int = DEFAULT_FETCH_THRESHOLD) -> Dict[str, Any]:
    """Fetch data-science papers in JeanBaptisteAgent's categories."""
    return fetch_papers(CONFIG, start_date, categories, end_date, max_results, min_threshold)


@function_tool(name_override="find_topic_tool")
def find_topic_tool(csv_path: str, n_topics: int = DEFAULT_MAX_TOPICS, n_papers_per_topic: int = 3) -> Dict[str, Any]:
    """Extract topics from fetched paper metadata."""
    return find_topics(csv_path, n_topics, n_papers_per_topic)


@function_tool(name_override="extract_main_result_tool")
def extract_main_result_tool(arxiv_id: str, title: str = "", max_chars: int = 12000) -> Dict[str, Any]:
    """Explain a data-science paper's principal result."""
    return extract_main_result(CONFIG, arxiv_id, title, max_chars)


@function_tool(name_override="get_arxiv_categories_tool")
def get_arxiv_categories_tool(message: str) -> Dict[str, Any]:
    """Infer JeanBaptisteAgent's relevant arXiv categories."""
    return {"categories": classify_categories(CONFIG, message)}


def build_jean_baptiste_agent():
    """Build JeanBaptisteAgent with its OpenAI Agents SDK tools."""
    return build_specialist_agent(CONFIG, [get_arxiv_categories_tool, check_paper_tool, arxiv_fetcher_tool, find_topic_tool, extract_main_result_tool])


def run_jean_baptiste_agent(message: str) -> Dict[str, Any]:
    """Run one traced JeanBaptisteAgent turn."""
    return run_specialist_agent(CONFIG, build_jean_baptiste_agent, message)
