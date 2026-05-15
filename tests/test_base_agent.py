"""Tests for the base agent abstraction and tool system."""

from datetime import datetime
from unittest.mock import Mock

from src.agents.base_agent import AgentTool, BaseAgent, ToolCall
from src.agents.tools import (
    analyze_paper_tool,
    check_threshold_tool,
    fetch_papers_tool,
    generate_summary_tool,
    get_base_tools,
)
from src.fetchers.arxiv_fetcher import Paper


def _make_paper(arxiv_id: str = "2301.12345") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title="A Test Paper",
        authors=["Author One"],
        summary="We study a problem and prove a useful theorem.",
        published=datetime(2023, 1, 15),
        updated=datetime(2023, 1, 16),
        categories=["math.PR"],
        primary_category="math.PR",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        entry_id=f"http://arxiv.org/abs/{arxiv_id}",
    )


class TestBaseAgent:
    """Tests for BaseAgent behavior."""

    def test_base_agent_initialization_with_tools(self):
        agent = BaseAgent(
            name="Chris",
            expertise="Probability theory",
            categories=["math.PR"],
            tools=get_base_tools(),
        )

        assert agent.name == "Chris"
        assert agent.categories == ["math.PR"]
        assert "fetch_papers_tool" in agent.list_tools()
        assert agent.conversation_history[0]["role"] == "system"
        assert "Probability theory" in agent.system_prompt

    def test_execute_tool_success(self):
        agent = BaseAgent(
            name="Tester",
            expertise="Testing",
            tools=[
                AgentTool(
                    name="add",
                    description="Add two numbers",
                    function=lambda a, b: a + b,
                    required_parameters=["a", "b"],
                )
            ],
        )

        result = agent.execute_tool("add", {"a": 2, "b": 3})

        assert result.success is True
        assert result.result == 5
        assert agent.state["tool_calls"][0]["tool_name"] == "add"

    def test_execute_tool_missing_parameter_returns_error(self):
        agent = BaseAgent(
            name="Tester",
            expertise="Testing",
            tools=[
                AgentTool(
                    name="add",
                    description="Add two numbers",
                    function=lambda a, b: a + b,
                    required_parameters=["a", "b"],
                )
            ],
        )

        result = agent.execute_tool("add", {"a": 2})

        assert result.success is False
        assert "Missing required parameter" in result.error

    def test_parse_tool_calls_from_dict(self):
        calls = BaseAgent.parse_tool_calls(
            {
                "tool_calls": [
                    {
                        "name": "check_threshold_tool",
                        "parameters": {"paper_count": 120, "min_threshold": 100},
                    }
                ]
            }
        )

        assert calls == [
            ToolCall(
                name="check_threshold_tool",
                parameters={"paper_count": 120, "min_threshold": 100},
            )
        ]

    def test_parse_tool_calls_from_fenced_json(self):
        response = """
        I will check the threshold.

        ```json
        {"tool": "check_threshold_tool", "parameters": {"paper_count": 80, "min_threshold": 100}}
        ```
        """

        calls = BaseAgent.parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].name == "check_threshold_tool"
        assert calls[0].parameters["paper_count"] == 80

    def test_respond_executes_llm_tool_calls(self):
        llm_client = Mock(
            return_value={
                "tool_calls": [
                    {
                        "name": "check_threshold_tool",
                        "parameters": {"paper_count": 101, "min_threshold": 100},
                    }
                ]
            }
        )
        agent = BaseAgent(
            name="Julius",
            expertise="Editorial coordination",
            llm_client=llm_client,
            tools=get_base_tools(),
        )

        response = agent.respond("Check whether we have enough papers")

        assert response["tool_calls"][0].name == "check_threshold_tool"
        assert response["tool_results"][0].success is True
        assert response["tool_results"][0].result["threshold_met"] is True

    def test_execute_tool_injects_openai_client_for_tools_that_accept_llm_client(self):
        class FakeOpenAIClient:
            def __init__(self):
                self.responses = self

            def create(self, model, input):
                return {"model": model, "input": input}

        captured = {}
        client = FakeOpenAIClient()
        agent = BaseAgent(
            name="Tester",
            expertise="Testing",
            llm_client=client,
            tools=[
                AgentTool(
                    name="needs_llm",
                    description="Tool that expects an llm client",
                    function=lambda prompt, llm_client=None: captured.setdefault("llm_client", llm_client),
                    required_parameters=["prompt"],
                )
            ],
        )

        result = agent.execute_tool("needs_llm", {"prompt": "summarize this"})

        assert result.success is True
        assert captured["llm_client"] is client

    def test_execute_tool_injects_openai_client_for_tools_that_accept_openai_client(self):
        class FakeOpenAIClient:
            def __init__(self):
                self.responses = self

            def create(self, model, input):
                return {"model": model, "input": input}

        captured = {}
        client = FakeOpenAIClient()
        agent = BaseAgent(
            name="Tester",
            expertise="Testing",
            llm_client=client,
            tools=[
                AgentTool(
                    name="needs_openai",
                    description="Tool that expects an OpenAI client",
                    function=lambda prompt, openai_client=None: captured.setdefault("openai_client", openai_client),
                    required_parameters=["prompt"],
                )
            ],
        )

        result = agent.execute_tool("needs_openai", {"prompt": "discover topics"})

        assert result.success is True
        assert captured["openai_client"] is client


class TestBaseTools:
    """Tests for the base agent tools."""

    def test_check_threshold_tool(self):
        result = check_threshold_tool(paper_count=75, min_threshold=100)

        assert result["threshold_met"] is False
        assert result["missing_count"] == 25

    def test_analyze_paper_tool_extracts_problem_and_results(self):
        paper_text = (
            "This paper studies a problem in stochastic processes. "
            "We prove a convergence theorem for the model. "
            "We show that the limiting behavior is stable."
        )

        result = analyze_paper_tool(paper_text)

        assert "problem" in result["problem"].lower()
        assert len(result["main_results"]) >= 2
        assert result["confidence"] == "heuristic"

    def test_generate_summary_tool(self):
        result = generate_summary_tool([_make_paper()], topic="Probability advances")

        assert result["topic"] == "Probability advances"
        assert result["paper_count"] == 1
        assert result["representative_papers"] == ["A Test Paper"]
        assert result["summary"].startswith("Probability advances:")

    def test_fetch_papers_tool_uses_fetcher(self):
        mock_fetcher = Mock()
        mock_fetcher.fetch_multiple_categories.return_value = [_make_paper()]

        result = fetch_papers_tool(
            categories=["math.PR"],
            start_date="2023-01-01",
            end_date="2023-01-07",
            max_results=10,
            fetcher=mock_fetcher,
        )

        assert result["paper_count"] == 1
        assert result["papers"][0]["arxiv_id"] == "2301.12345"
        mock_fetcher.fetch_multiple_categories.assert_called_once()

    def test_fetch_papers_tool_uses_threshold_fetch_when_min_count_given(self):
        mock_fetcher = Mock()
        mock_fetcher.fetch_with_threshold.return_value = (
            [_make_paper()],
            datetime(2022, 12, 25),
            datetime(2023, 1, 7),
        )

        result = fetch_papers_tool(
            categories=["math.PR"],
            start_date="2023-01-01",
            end_date="2023-01-07",
            min_count=1,
            fetcher=mock_fetcher,
        )

        assert result["threshold_met"] is True
        assert result["start_date"] == "2022-12-25T00:00:00"
        mock_fetcher.fetch_with_threshold.assert_called_once()
