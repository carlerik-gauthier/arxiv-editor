"""Tests for phase-3.2 specialized research agents."""

import pytest

from config.settings import Settings
from src.agents import (
    AbdoulayeAgent,
    AgentHandoff,
    AgentTool,
    AlainAgent,
    BrunoAgent,
    ChrisAgent,
    ElisaAgent,
    FelixAgent,
    HandoffContext,
    JeanBaptisteAgent,
    JuliusAgent,
    MichelAgent,
    create_all_specialized_agents,
    create_specialized_agent,
)
from agents import Agent as OpenAIAgent
from agents import FunctionTool
from src.agents.tools import check_threshold_tool, create_metaphor_tool, generate_summary_tool
from src.agents.specialized_agents import DEFAULT_SPECIALIST_MIN_PAPERS


EXPECTED_AGENTS = [
    (MichelAgent, "Michel", ["math.HO", "math.GM"], "create_metaphor_tool"),
    (ChrisAgent, "Chris", ["math.PR", "stat.TH"], None),
    (AlainAgent, "Alain", ["math.AG", "math.RA", "math.GR"], None),
    (BrunoAgent, "Bruno", ["math.DG", "math.SP"], None),
    (ElisaAgent, "Elisa", ["cs.CR", "math.OC"], None),
    (FelixAgent, "Felix", ["math.DS", "math.SG"], None),
    (AbdoulayeAgent, "Abdoulaye", ["cs.LG", "stat.ML"], None),
    (
        JeanBaptisteAgent,
        "JeanBaptiste",
        ["cs.CL", "cs.AI", "cs.MA", "cs.CE"],
        None,
    ),
]


@pytest.mark.parametrize("agent_class,name,categories,custom_tool", EXPECTED_AGENTS)
def test_specialized_agent_profiles_and_tools(agent_class, name, categories, custom_tool):
    """Each specialist exposes its profile, prompt, categories, and research tools."""
    agent = agent_class()

    assert agent.name == name
    assert agent.categories == categories
    assert "specialized ArXiv research agent" in agent.system_prompt
    assert agent.expertise in agent.system_prompt
    assert "fetch_papers_tool" in agent.list_tools()
    assert "check_threshold_tool" in agent.list_tools()
    assert "analyze_paper_tool" in agent.list_tools()
    assert "generate_summary_tool" in agent.list_tools()
    if custom_tool:
        assert custom_tool in agent.list_tools()
    assert isinstance(agent.sdk_agent, OpenAIAgent)
    assert all(isinstance(tool, FunctionTool) for tool in agent.get_sdk_tools())


def test_michel_metaphor_tool_execution():
    """Michel can call the custom metaphor tool required by step 3.2."""
    agent = MichelAgent()

    result = agent.execute_tool(
        "create_metaphor_tool",
        {"concept": "Riemannian curvature", "audience": "curious readers"},
    )

    assert result.success is True
    assert result.result["concept"] == "Riemannian curvature"
    assert result.result["audience"] == "curious readers"
    assert "landscape" in result.result["metaphor"]
    assert result.result["caveat"]


def test_create_metaphor_tool_rejects_empty_concept():
    """The metaphor tool validates required semantic input, not only parameters."""
    with pytest.raises(ValueError, match="concept cannot be empty"):
        create_metaphor_tool("   ")


def test_specialized_agent_factory_normalizes_names():
    """Factory lookup tolerates common user-facing JeanBaptiste spellings."""
    agent = create_specialized_agent("Jean Baptiste")

    assert isinstance(agent, JeanBaptisteAgent)
    assert agent.categories == ["cs.CL", "cs.AI", "cs.MA", "cs.CE"]


def test_create_all_specialized_agents():
    """All step-3.2 specialists are instantiated in a stable workflow order."""
    agents = create_all_specialized_agents()

    assert [agent.name for agent in agents] == [
        "Michel",
        "Chris",
        "Alain",
        "Bruno",
        "Elisa",
        "Felix",
        "Abdoulaye",
        "JeanBaptiste",
    ]


def test_julius_propagates_llm_client_to_default_specialists():
    """Default specialist construction should inherit Julius's OpenAI client."""
    class FakeOpenAIClient:
        def __init__(self):
            self.responses = self

        def create(self, model, input):
            return {"model": model, "input": input}

    client = FakeOpenAIClient()
    julius = JuliusAgent(llm_client=client)

    assert julius.llm_client is client
    assert julius.specialist_agents["JeanBaptiste"].llm_client is client
    assert julius.specialist_agents["Chris"].llm_client is client


def test_settings_include_jean_baptiste_categories():
    """Configuration exposes the step-3.2 JeanBaptiste category assignment."""
    settings = Settings()

    assert settings.get_agent_categories("JeanBaptiste") == [
        "cs.CL",
        "cs.AI",
        "cs.MA",
        "cs.CE",
    ]


def test_specialized_handoff_fetches_topics_and_summarizes_representatives():
    """A called specialist fetches abstracts, discovers topics, and summarizes representatives."""
    events = []

    def fake_fetch_papers_tool(categories, start_date, end_date=None, min_count=None):
        events.append(("fetch", categories, start_date, end_date, min_count))
        return {
            "papers": [
                {
                    "title": "Agent Planning Benchmarks",
                    "summary": "We study language agents and planning benchmarks.",
                    "arxiv_id": "2605.00001",
                    "categories": ["cs.AI"],
                },
                {
                    "title": "Tool Use in LLM Agents",
                    "summary": "We evaluate tool use in large language model agents.",
                    "arxiv_id": "2605.00002",
                    "categories": ["cs.AI"],
                },
            ],
            "paper_count": 2,
            "threshold_met": True,
        }

    def fake_discover_topics_tool(
        papers,
        min_topic_size=2,
        num_topics=None,
        representative_papers_per_topic=5,
        use_openai_representation=True,
    ):
        paper_list = list(papers)
        events.append(
            (
                "discover",
                len(paper_list),
                min_topic_size,
                num_topics,
                representative_papers_per_topic,
                use_openai_representation,
            )
        )
        return {
            "topics": [
                {
                    "title": "Language Agent Planning",
                    "description": "LLM-generated description of language agent planning.",
                    "description_source": "llm",
                    "keywords": ["agents", "planning"],
                    "paper_count": len(paper_list),
                    "representative_papers": paper_list[:2],
                }
            ],
            "topic_count": 1,
            "paper_count": len(paper_list),
            "status": "completed",
        }

    specialist = JeanBaptisteAgent(
        tools=[
            AgentTool(
                name="check_threshold_tool",
                description="Check paper availability.",
                function=check_threshold_tool,
                required_parameters=["paper_count", "min_threshold"],
            ),
            AgentTool(
                name="fetch_papers_tool",
                description="Fetch papers.",
                function=fake_fetch_papers_tool,
                required_parameters=["categories", "start_date"],
            ),
            AgentTool(
                name="discover_topics_tool",
                description="Discover topics.",
                function=fake_discover_topics_tool,
                required_parameters=["papers"],
            ),
            AgentTool(
                name="generate_summary_tool",
                description="Summarize topic papers.",
                function=generate_summary_tool,
                required_parameters=["papers", "topic"],
            ),
        ]
    )
    julius = JuliusAgent(specialist_agents=[specialist])
    context = HandoffContext(
        task_description="Review LLM agent work.",
        constraints={
            "summary_request": {
                "topic_query": "LLM agents",
                "date_range": {
                    "start_date": "2026-05-01",
                    "end_date": "2026-05-07",
                },
                    "must_include_categories": ["cs.AI"],
                    "max_papers": 2,
                    "max_topics": 1,
                },
                "selected_papers": [],
                "max_topics": 1,
            },
        )

    handoff = AgentHandoff.execute_handoff(julius, specialist, context)

    response = handoff.result["response"]
    tool_names = [call["tool_name"] for call in specialist.state["tool_calls"]]
    assert handoff.status.value == "COMPLETED"
    assert events[0] == (
        "fetch",
        ["cs.AI"],
        "2026-05-01",
        "2026-05-07",
        DEFAULT_SPECIALIST_MIN_PAPERS,
    )
    assert events[1] == ("discover", 2, 2, 1, 2, True)
    assert tool_names == [
        "check_threshold_tool",
        "fetch_papers_tool",
        "discover_topics_tool",
        "generate_summary_tool",
    ]
    assert response["paper_count"] == 2
    assert response["requested_topic_count"] == 1
    assert response["minimum_paper_count"] == DEFAULT_SPECIALIST_MIN_PAPERS
    assert response["topic_summaries"][0]["topic"] == "Language Agent Planning"
    assert response["topic_summaries"][0]["description_source"] == "llm"
    assert response["topic_summaries"][0]["importance"] == 2
    assert "Importance: 2 papers." in response["topic_summaries"][0]["main_results_and_importance"]
    assert "Representative papers" in response["response"]


def test_specialized_handoff_does_not_hide_failed_topic_discovery():
    """A failed discover_topics_tool call should not be masked by synthetic topic summaries."""
    specialist = JeanBaptisteAgent(
        tools=[
            AgentTool(
                name="check_threshold_tool",
                description="Check paper availability.",
                function=check_threshold_tool,
                required_parameters=["paper_count", "min_threshold"],
            ),
            AgentTool(
                name="fetch_papers_tool",
                description="Fetch papers.",
                function=lambda categories, start_date, end_date=None, min_count=None: {
                    "papers": [
                        {
                            "title": "Agent Planning Benchmarks",
                            "summary": "We study language agents and planning benchmarks.",
                            "arxiv_id": "2605.00001",
                            "categories": ["cs.AI"],
                        }
                    ],
                    "paper_count": 1,
                    "threshold_met": False,
                },
                required_parameters=["categories", "start_date"],
            ),
            AgentTool(
                name="discover_topics_tool",
                description="Discover topics.",
                function=lambda papers, **_kwargs: {
                    "topics": [],
                    "topic_count": 0,
                    "paper_count": len(list(papers)),
                    "status": "failed",
                    "error": "missing OpenAI client",
                },
                required_parameters=["papers"],
            ),
            AgentTool(
                name="generate_summary_tool",
                description="Summarize topic papers.",
                function=generate_summary_tool,
                required_parameters=["papers", "topic"],
            ),
        ]
    )
    julius = JuliusAgent(specialist_agents=[specialist])
    context = HandoffContext(
        task_description="Review LLM agent work.",
        constraints={
            "summary_request": {
                "topic_query": "LLM agents",
                "date_range": {
                    "start_date": "2026-05-01",
                    "end_date": "2026-05-07",
                },
                "must_include_categories": ["cs.AI"],
                "max_papers": 2,
                "max_topics": 1,
            },
            "selected_papers": [
                {
                    "title": "Agent Planning Benchmarks",
                    "summary": "We study language agents and planning benchmarks.",
                    "arxiv_id": "2605.00001",
                    "categories": ["cs.AI"],
                }
            ],
            "max_topics": 1,
        },
    )

    handoff = AgentHandoff.execute_handoff(julius, specialist, context)

    response = handoff.result["response"]
    tool_names = [call["tool_name"] for call in specialist.state["tool_calls"]]
    assert handoff.status.value == "COMPLETED"
    assert response["topic_summaries"] == []
    assert response["topic_discovery"]["status"] == "failed"
    assert response["status"] == "needs_data"
    assert tool_names == ["check_threshold_tool", "fetch_papers_tool", "discover_topics_tool"]
