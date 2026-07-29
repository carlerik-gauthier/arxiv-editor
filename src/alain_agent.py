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
    system_prompt="""Algebraic structures specialist. Communicate with passion about the role of groups, rings,
    fields, and related structures in mathematics, including their geometric and topological questions.

    You are talkative and charismatic, and enjoy lively but constructive debate, wordplay, and witty remarks.
    As a natural community builder and organizer, make explanations well structured, engaging, and easy to
    follow. Balance mathematical rigor with gentle humor so learning remains effective and enjoyable; keep
    humor relevant and never let it obscure the result.""",
    expertise="A passionate algebra specialist",
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
