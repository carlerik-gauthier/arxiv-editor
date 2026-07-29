"""FelixAgent, the dynamical systems and symplectic geometry specialist."""

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
    run_specialist_agent
    )

FELIX_CATEGORIES = ("math.DS", "math.SG")
CONFIG = SpecialistConfig(
    name="FelixAgent", 
    slug="felix", 
    categories=FELIX_CATEGORIES, 
    category_descriptions={
        "math.DS": """Dynamical systems papers about dynamics of differential equations and flows, mechanics, classical few-body problems, iterations, complex dynamics, delayed differential equations.""", 
        "math.SG": "Symplectic geometry papers, including Hamiltonian dynamics. It contains topics such as Hamiltonian systems, symplectic flows, classical integrable systems."
        }, 
        system_prompt="""Dynamical systems and symplectic geometry expert focused on long-term behavior results.
        You are playful, eccentric, and curious, with the energy of a mad scientist and a deep appreciation for
        coffee. Use a light, mischievous tone and occasional gentle teasing only when it fits the context. You
        naturally say “kind of” from time to time, but never so often that it weakens precision. Keep your
        explanations lively and slightly surprising while remaining mathematically accurate and useful.""",
        expertise="an imaginative, exceptionally smart dynamical systems and symplectic geometry specialist", 
        expertise_domain="mathematics"
        )

@function_tool(name_override="check_paper_tool")
def check_paper_tool(start_date: str, end_date: str, categories: List[str]) -> Dict[str, Any]: 
    return check_papers(CONFIG, start_date, end_date, categories)

@function_tool(name_override="arxiv_fetcher_tool")
def arxiv_fetcher_tool(start_date: str, categories: List[str], end_date: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS, min_threshold: int = DEFAULT_FETCH_THRESHOLD) -> Dict[str, Any]: 
    return fetch_papers(CONFIG, start_date, categories, end_date, max_results, min_threshold)

@function_tool(name_override="find_topic_tool")
def find_topic_tool(csv_path: str, n_topics: int = DEFAULT_MAX_TOPICS, n_papers_per_topic: int = 3) -> Dict[str, Any]: 
    return find_topics(csv_path, n_topics, n_papers_per_topic)

@function_tool(name_override="extract_main_result_tool")
def extract_main_result_tool(arxiv_id: str, title: str = "", max_chars: int = 12000) -> Dict[str, Any]: 
    return extract_main_result(CONFIG, arxiv_id, title, max_chars)

@function_tool(name_override="get_arxiv_categories_tool")
def get_arxiv_categories_tool(message: str) -> Dict[str, Any]: 
    return {"categories": classify_categories(CONFIG, message)}

def build_felix_agent(): 
    return build_specialist_agent(CONFIG, [get_arxiv_categories_tool, check_paper_tool, arxiv_fetcher_tool, find_topic_tool, extract_main_result_tool])

def run_felix_agent(message: str) -> Dict[str, Any]: 
    return run_specialist_agent(CONFIG, build_felix_agent, message)
