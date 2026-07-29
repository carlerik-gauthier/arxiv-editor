"""BrunoAgent, the spectral and Riemannian geometry specialist."""

from typing import Any, Dict, List, Optional

from agents import function_tool

from src.specialist_agent import (
    DEFAULT_FETCH_THRESHOLD, DEFAULT_MAX_RESULTS, DEFAULT_MAX_TOPICS, SpecialistConfig,
    build_specialist_agent, check_papers, classify_categories, extract_main_result,
    fetch_papers, find_topics, run_specialist_agent,
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
    system_prompt="""Geometry expert who emphasizes geometric intuition, with expertise in spectral and
    Riemannian geometry. You are reserved, methodical, and exceptionally rigorous: use mathematically precise
    language, state assumptions and logical dependencies clearly, and avoid overstating results. Do not merely
    hand over conclusions; guide the reader through the essential reasoning so they can think independently.
    Be concise and disciplined, while making the geometric picture clear.""",
    expertise="spectral and Riemannian geometry specialist who communicates with exceptional rigor",
    expertise_domain="mathematics",
)


@function_tool(name_override="check_paper_tool")
def check_paper_tool(start_date: str, end_date: str, categories: List[str]) -> Dict[str, Any]:
    """Check whether geometry papers are already stored."""
    return check_papers(CONFIG, start_date, end_date, categories)


@function_tool(name_override="arxiv_fetcher_tool")
def arxiv_fetcher_tool(start_date: str, categories: List[str], end_date: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS, min_threshold: int = DEFAULT_FETCH_THRESHOLD) -> Dict[str, Any]:
    """Fetch geometry papers in BrunoAgent's allowed categories."""
    return fetch_papers(CONFIG, start_date, categories, end_date, max_results, min_threshold)


@function_tool(name_override="find_topic_tool")
def find_topic_tool(csv_path: str, n_topics: int = DEFAULT_MAX_TOPICS, n_papers_per_topic: int = 3) -> Dict[str, Any]:
    """Extract topics from fetched paper metadata."""
    return find_topics(csv_path, n_topics, n_papers_per_topic)


@function_tool(name_override="extract_main_result_tool")
def extract_main_result_tool(arxiv_id: str, title: str = "", max_chars: int = 12000) -> Dict[str, Any]:
    """Explain a geometry paper's principal result."""
    return extract_main_result(CONFIG, arxiv_id, title, max_chars)


@function_tool(name_override="get_arxiv_categories_tool")
def get_arxiv_categories_tool(message: str) -> Dict[str, Any]:
    """Infer BrunoAgent's relevant arXiv categories."""
    return {"categories": classify_categories(CONFIG, message)}


def build_bruno_agent():
    """Build BrunoAgent with its OpenAI Agents SDK tools."""
    return build_specialist_agent(CONFIG, [get_arxiv_categories_tool, check_paper_tool, arxiv_fetcher_tool, find_topic_tool, extract_main_result_tool])


def run_bruno_agent(message: str) -> Dict[str, Any]:
    """Run one traced BrunoAgent turn."""
    return run_specialist_agent(CONFIG, build_bruno_agent, message)
