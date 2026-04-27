"""Tests for phase-3.2 specialized research agents."""

import pytest

from config.settings import Settings
from src.agents import (
    AbdoulayeAgent,
    AlainAgent,
    BrunoAgent,
    ChrisAgent,
    ElisaAgent,
    FelixAgent,
    JeanBaptisteAgent,
    MichelAgent,
    create_all_specialized_agents,
    create_specialized_agent,
)
from src.agents.tools import create_metaphor_tool


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


def test_settings_include_jean_baptiste_categories():
    """Configuration exposes the step-3.2 JeanBaptiste category assignment."""
    settings = Settings()

    assert settings.get_agent_categories("JeanBaptiste") == [
        "cs.CL",
        "cs.AI",
        "cs.MA",
        "cs.CE",
    ]
