"""AlainAgent, the algebra specialist."""

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


ALAIN_CATEGORIES = ("math.AG", "math.RA", "math.GR", "math.AT")
CONFIG = SpecialistConfig(
    name="AlainAgent",
    slug="alain",
    categories=ALAIN_CATEGORIES,
    category_descriptions={
        "math.AG": """Algebraic Geometry papers. Algebraic geometry is a branch of mathematics which uses abstract algebraic techniques, 
        mainly from commutative algebra, to solve geometrical problems.""",
        "math.RA": """Rings and Algebras papers. Ring theory is the study of rings, algebraic structures in which addition and multiplication are defined "
        "and have similar properties to those operations defined for the integers. Never select this category alone""",
        "math.GR": "Group Theory papers. Group theory studies the algebraic structures known as groups. The concept of a group is central to algebra.",
        "math.AT": """Algebraic topology papers. Algebraic topology is a branch of mathematics that uses tools from abstract algebra to 
        study topological spaces. The basic goal is to find algebraic invariants that classify topological spaces up to homeomorphism, 
        though usually most classify up to homotopy equivalence."""
    },
    system_prompt="""Algebraic structures specialist, you communicate with passion about how those structures play a crucial role in mathematics. 
    Algebraic structures include, but are not limited to, groups, rings and fields. The study of such structures addresses questions about their 
    geometry and topology.""",
    expertise="algebra specialist",
    expertise_domain="mathematics"
)


@function_tool(name_override="check_paper_tool")
def check_paper_tool(start_date: str, end_date: str, categories: List[str]) -> Dict[str, Any]:
    """Check whether algebra papers are already stored."""
    return check_papers(CONFIG, start_date, end_date, categories)


@function_tool(name_override="arxiv_fetcher_tool")
def arxiv_fetcher_tool(start_date: str, categories: List[str], end_date: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS, min_threshold: int = DEFAULT_FETCH_THRESHOLD) -> Dict[str, Any]:
    """Fetch algebra papers in AlainAgent's allowed categories."""
    return fetch_papers(CONFIG, start_date, categories, end_date, max_results, min_threshold)


@function_tool(name_override="find_topic_tool")
def find_topic_tool(csv_path: str, n_topics: int = DEFAULT_MAX_TOPICS, n_papers_per_topic: int = 3) -> Dict[str, Any]:
    """Extract topics from fetched paper metadata."""
    return find_topics(csv_path, n_topics, n_papers_per_topic)


@function_tool(name_override="extract_main_result_tool")
def extract_main_result_tool(arxiv_id: str, title: str = "", max_chars: int = 12000) -> Dict[str, Any]:
    """Explain an algebra paper's principal result."""
    return extract_main_result(CONFIG, arxiv_id, title, max_chars)


@function_tool(name_override="get_arxiv_categories_tool")
def get_arxiv_categories_tool(message: str) -> Dict[str, Any]:
    """Infer AlainAgent's relevant arXiv categories."""
    return {"categories": classify_categories(CONFIG, message)}


def build_alain_agent():
    """Build AlainAgent with its OpenAI Agents SDK tools."""
    return build_specialist_agent(CONFIG, [get_arxiv_categories_tool, check_paper_tool, arxiv_fetcher_tool, find_topic_tool, extract_main_result_tool])


def run_alain_agent(message: str) -> Dict[str, Any]:
    """Run one traced AlainAgent turn."""
    return run_specialist_agent(CONFIG, build_alain_agent, message)
