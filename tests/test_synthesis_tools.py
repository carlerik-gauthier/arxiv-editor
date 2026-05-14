"""Tests for phase-6.3 synthesis tools and first-draft workflow."""

from src.agents import AgentTool, JeanBaptisteAgent, JuliusAgent, MichelAgent
from src.agents.tools import (
    create_paper_summary_tool,
    create_topic_overview_tool,
    generate_layperson_explanation_tool,
    generate_summary_tool,
    get_synthesis_tools,
    rank_summary_items_tool,
    review_and_refine_tool,
)
from src.generation.synthesizer import ContentSynthesizer
from src.generation.user_request import SummaryRequest, parse_user_request_tool


def _paper(title="Agent Planning Benchmarks", score=0.9):
    """Build a paper-like dict for deterministic synthesis tests."""
    return {
        "arxiv_id": "2605.00001",
        "title": title,
        "summary": "We introduce a benchmark for LLM agents and show stronger planning evaluation.",
        "categories": ["cs.AI"],
        "score": score,
    }


def _analysis():
    """Build a compact analysis record."""
    return {
        "problem": "LLM agents are difficult to evaluate across long-horizon planning tasks.",
        "main_results": ["The benchmark separates planning skill from tool-use noise."],
        "impact_summary": "The result helps compare agent systems more reliably.",
        "confidence": "medium",
    }


def _fast_specialist_tools():
    """Return deterministic tools that avoid live ArXiv and BERTopic calls."""
    def threshold_met(paper_count, min_threshold=60):
        return {
            "paper_count": paper_count,
            "min_threshold": min_threshold,
            "threshold_met": True,
            "missing_count": 0,
        }

    def discover_topics(
        papers,
        min_topic_size=2,
        num_topics=None,
        representative_papers_per_topic=5,
        use_openai_representation=True,
    ):
        paper_list = list(papers)
        return {
            "topics": [
                {
                    "title": "LLM Agent Planning",
                    "description": "LLM-generated topic description.",
                    "description_source": "llm",
                    "keywords": ["agents", "planning"],
                    "representative_papers": paper_list[:representative_papers_per_topic],
                    "paper_count": len(paper_list),
                }
            ][: num_topics or 1],
            "topic_count": 1,
            "paper_count": len(paper_list),
            "status": "completed",
        }

    return [
        AgentTool(
            name="check_threshold_tool",
            description="Check paper threshold.",
            function=threshold_met,
            required_parameters=["paper_count", "min_threshold"],
        ),
        AgentTool(
            name="discover_topics_tool",
            description="Discover topics.",
            function=discover_topics,
            required_parameters=["papers"],
        ),
        AgentTool(
            name="generate_summary_tool",
            description="Summarize topic papers.",
            function=generate_summary_tool,
            required_parameters=["papers", "topic"],
        ),
        *get_synthesis_tools(),
    ]


def test_synthesis_tools_create_overviews_summaries_and_rankings():
    """Step-6.3 tools return structured synthesis pieces."""
    request = SummaryRequest(topic_query="LLM agents")

    overview = create_topic_overview_tool(
        "LLM agents",
        papers=[_paper()],
        analyses=[_analysis()],
        summary_request=request,
    )
    paper_summary = create_paper_summary_tool(_paper(), _analysis(), request)
    ranking = rank_summary_items_tool([{"title": "B", "score": 0.2}, {"title": "A", "score": 0.9}])
    lay = generate_layperson_explanation_tool("agents plan with tools")
    review = review_and_refine_tool("A short ArXiv-linked draft.", criteria=["citation"])

    assert overview["topic"] == "LLM agents"
    assert overview["paper_titles"] == ["Agent Planning Benchmarks"]
    assert paper_summary["main_result"].startswith("The benchmark")
    assert ranking["items"][0]["title"] == "A"
    assert lay["style"] == "layperson"
    assert review["passed"] is True
    assert {tool.name for tool in get_synthesis_tools()} >= {
        "create_topic_overview_tool",
        "create_paper_summary_tool",
        "review_and_refine_tool",
    }


def test_content_synthesizer_respects_request_format_and_provenance():
    """ContentSynthesizer renders according to SummaryRequest preferences."""
    parsed = parse_user_request_tool(
        "Give me a paper ranking on LLM agents with at most 1 papers",
        reference_date="2026-05-12",
    )

    draft = ContentSynthesizer().synthesize_draft(
        summary_request=parsed["summary_request"],
        selected_papers=[_paper()],
        analyses=[_analysis()],
        agent_results=[{"to_agent": "JeanBaptiste", "status": "COMPLETED"}],
    )

    assert draft["status"] == "drafted"
    assert "Draft v1" in draft["content"]
    assert "Representative Paper Rankings" in draft["title"]
    assert len(draft["provenance"]["selected_papers"]) == 1
    assert draft["provenance"]["agent_callbacks"][0]["agent"] == "JeanBaptiste"


def test_content_synthesizer_caps_specialist_topics_to_seven():
    """Specialist topic callbacks are rendered with an absolute seven-topic cap."""
    callbacks = [
        {
            "to_agent": "JeanBaptiste",
            "status": "COMPLETED",
            "result": {
                "response": {
                    "topic_summaries": [
                        {
                            "topic": f"Topic {index}",
                            "description": f"Description {index}",
                            "main_results_and_importance": f"Main results {index}",
                            "representative_papers": [
                                {
                                    "title": f"Paper {index}",
                                    "arxiv_id": f"2605.{index:05d}",
                                }
                            ],
                        }
                        for index in range(1, 10)
                    ]
                }
            },
        }
    ]

    draft = ContentSynthesizer().synthesize_draft(
        summary_request=SummaryRequest(topic_query="LLM agents", max_topics=20),
        selected_papers=[_paper()],
        analyses=[_analysis()],
        agent_results=callbacks,
    )

    assert len(draft["sections"]) == 21
    assert "Topic 7" in draft["content"]
    assert "Topic 8" not in draft["content"]


def test_julius_generates_first_draft_with_specialist_handoff_provenance():
    """Julius delegates to matching specialists and compiles a first draft."""
    julius = JuliusAgent(
        specialist_agents=[
            MichelAgent(),
            JeanBaptisteAgent(tools=_fast_specialist_tools()),
        ]
    )
    request = SummaryRequest(
        topic_query="LLM agents",
        must_include_categories=["cs.AI"],
        max_papers=1,
    )

    result = julius.generate_first_draft_tool(
        summary_request=request,
        selected_papers=[_paper()],
        analyses=[_analysis()],
    )

    assert result["draft"]["status"] == "drafted"
    assert result["selected_agents"] == ["JeanBaptiste", "Michel"]
    assert result["agent_results"]["completed_count"] == 2
    assert "generate_first_draft_tool" in julius.list_tools()
    assert "create_topic_overview_tool" in julius.specialist_agents["JeanBaptiste"].list_tools()
    assert result["draft"]["provenance"]["selected_papers"][0]["title"] == "Agent Planning Benchmarks"

