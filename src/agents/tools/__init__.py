"""Tool definitions for research agents."""

from src.agents.tools.base_tools import (
    analyze_paper_tool,
    check_threshold_tool,
    fetch_papers_tool,
    generate_summary_tool,
    get_base_tools,
)
from src.agents.tools.embedding_tool import embed_text_tool, get_embedding_tool
from src.agents.tools.metaphor_tool import create_metaphor_tool, get_metaphor_tool
from src.agents.tools.paper_selection_tool import (
    get_paper_relevance_tool,
    get_paper_selection_tool,
    rank_papers_by_relevance_tool,
    select_representative_papers_tool,
)
from src.agents.tools.problem_extraction_tool import (
    extract_problem_statement_tool,
    get_problem_extraction_tool,
)
from src.agents.tools.results_extraction_tool import (
    extract_key_results_tool,
    get_results_extraction_tool,
)
from src.agents.tools.topic_discovery_tool import (
    discover_topics_tool,
    generate_topic_title_tool,
    get_topic_discovery_tool,
    get_topic_title_tool,
)

__all__ = [
    "analyze_paper_tool",
    "check_threshold_tool",
    "create_metaphor_tool",
    "discover_topics_tool",
    "embed_text_tool",
    "extract_key_results_tool",
    "extract_problem_statement_tool",
    "fetch_papers_tool",
    "generate_topic_title_tool",
    "generate_summary_tool",
    "get_base_tools",
    "get_embedding_tool",
    "get_metaphor_tool",
    "get_paper_relevance_tool",
    "get_paper_selection_tool",
    "get_problem_extraction_tool",
    "get_results_extraction_tool",
    "get_topic_discovery_tool",
    "get_topic_title_tool",
    "rank_papers_by_relevance_tool",
    "select_representative_papers_tool",
]
