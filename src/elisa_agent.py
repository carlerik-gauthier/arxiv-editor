"""ElisaAgent, the applied mathematics and cryptography specialist."""

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

ELISA_CATEGORIES = ("cs.CR", "math.OC", "math.NA")
CONFIG = SpecialistConfig(
    name="ElisaAgent", 
    slug="elisa", 
    categories=ELISA_CATEGORIES, 
    category_descriptions={
        "cs.CR": """Cryptography and security papers that cover all areas of cryptography and security, including cryptographic protocols and privacy, 
        authentication, public key cryptosytems, proof-carrying code, etc. Roughly includes material in ACM Subject Classes D.4.6 and E.3.""", 
        "math.OC": """Optimization and control papers in applied mathematics. It talks about
        Operations research, linear programming, control theory, systems theory, optimal control, game theory.""",
        "math.NA": """Numerical Analysis papers that studies numerical algorithms for problems in analysis and algebra, scientific computation"""
        }, 
    system_prompt="""Applied mathematics and cryptography specialist. You are expressive, enthusiastic, and
    result-oriented, and enjoy making technical work lively, accessible, and interactive. Communicate with
    curiosity, respect, cultural awareness, and sensitivity to diverse perspectives. Share your passion without
    sacrificing accuracy; highlight practical implications and concrete outcomes when the evidence supports them.""",
    expertise="dynamic, result-oriented applied mathematics and cryptography specialist", 
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
def get_arxiv_categories_tool(message: str) -> Dict[str, Any]: return {"categories": classify_categories(CONFIG, message)}

def build_elisa_agent(): 
    return build_specialist_agent(CONFIG, [get_arxiv_categories_tool, check_paper_tool, arxiv_fetcher_tool, find_topic_tool, extract_main_result_tool])

def run_elisa_agent(message: str) -> Dict[str, Any]: 
    return run_specialist_agent(CONFIG, build_elisa_agent, message)
