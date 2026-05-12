"""Tool definitions for research agents."""

from src.agents.tools.base_tools import (
    analyze_paper_tool,
    check_threshold_tool,
    fetch_papers_tool,
    generate_summary_tool,
    get_base_tools,
)
from src.agents.tools.embedding_tool import embed_text_tool, get_embedding_tool
from src.agents.tools.formatting_tool import format_document_tool, get_formatting_tool
from src.agents.tools.impact_assessment_tool import (
    assess_impact_tool,
    get_impact_assessment_tool,
)
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
from src.agents.tools.quality_check_tool import (
    get_quality_check_tool,
    validate_quality_tool,
)
from src.agents.tools.results_extraction_tool import (
    extract_key_results_tool,
    get_results_extraction_tool,
)
from src.agents.tools.synthesis_tools import (
    create_paper_summary_tool,
    create_topic_overview_tool,
    generate_expert_explanation_tool,
    generate_layperson_explanation_tool,
    get_synthesis_tools,
    rank_summary_items_tool,
    review_and_refine_tool,
)
from src.agents.tools.topic_discovery_tool import (
    discover_topics_tool,
    generate_topic_title_tool,
    get_topic_discovery_tool,
    get_topic_title_tool,
)
from src.generation.user_request import clarify_request_tool, parse_user_request_tool

__all__ = [
    "analyze_paper_tool",
    "assess_impact_tool",
    "check_threshold_tool",
    "create_metaphor_tool",
    "create_paper_summary_tool",
    "create_topic_overview_tool",
    "discover_topics_tool",
    "embed_text_tool",
    "extract_key_results_tool",
    "extract_problem_statement_tool",
    "fetch_papers_tool",
    "format_document_tool",
    "generate_expert_explanation_tool",
    "generate_layperson_explanation_tool",
    "generate_topic_title_tool",
    "generate_summary_tool",
    "get_base_tools",
    "get_embedding_tool",
    "get_formatting_tool",
    "get_impact_assessment_tool",
    "get_metaphor_tool",
    "get_paper_relevance_tool",
    "get_paper_selection_tool",
    "get_problem_extraction_tool",
    "get_quality_check_tool",
    "get_results_extraction_tool",
    "get_synthesis_tools",
    "get_topic_discovery_tool",
    "get_topic_title_tool",
    "clarify_request_tool",
    "parse_user_request_tool",
    "rank_papers_by_relevance_tool",
    "rank_summary_items_tool",
    "review_and_refine_tool",
    "select_representative_papers_tool",
    "validate_quality_tool",
]
