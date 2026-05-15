"""Tests for the phase-5.1 problem statement extraction tool."""

import json

import pytest

from src.agents import ChrisAgent
from src.agents.tools import extract_problem_statement_tool
from src.analysis import PaperAnalyzer


PAPER_TEXT = """
Abstract
Markov chain Monte Carlo methods are central in computational probability.
However, quantitative mixing bounds remain limited for adaptive chains.
This paper studies the problem of proving non-asymptotic convergence guarantees.

1 Introduction
Adaptive Markov chains are important because they are used in Bayesian computation.
We address the challenge of controlling convergence when transition kernels change.
The lack of sharp finite-time guarantees makes these algorithms hard to certify.

2 Methods
We construct a coupling argument.
""".strip()


class _FakeOpenAIClient:
    """Minimal OpenAI-compatible client for paper-analysis tests."""

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []
        self.responses = self

    def create(self, model, input):
        self.prompts.append({"model": model, "input": input})

        class Response:
            pass

        response = Response()
        response.output_text = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return response


def test_paper_analyzer_extracts_sections_and_problem_fields():
    """PaperAnalyzer extracts documented problem fields from paper text."""
    analyzer = PaperAnalyzer()

    result = analyzer.extract_problem_statement(
        paper_text=PAPER_TEXT,
        paper_metadata={
            "title": "Adaptive Markov Chain Guarantees",
            "categories": ["math.PR"],
        },
    )

    assert result["source"] == "heuristic"
    assert result["confidence"] == "heuristic"
    assert "problem of proving" in result["problem"]
    assert any(
        keyword in result["motivation"].lower()
        for keyword in ("important", "central")
    )
    assert "limited" in result["research_gap"].lower()
    assert "abstract" in result["sections_used"]
    assert "introduction" in result["sections_used"]
    assert result["chunks_analyzed"] == 1


def test_extract_sections_returns_canonical_section_names():
    """Section extraction normalizes headings into stable canonical keys."""
    analyzer = PaperAnalyzer()

    sections = analyzer.extract_sections(PAPER_TEXT)

    assert set(sections) >= {"abstract", "introduction", "methods"}
    assert "Markov chain Monte Carlo" in sections["abstract"]
    assert "Adaptive Markov chains" in sections["introduction"]


def test_chunk_text_uses_configurable_token_budget():
    """chunk_text splits long text by approximate token count."""
    analyzer = PaperAnalyzer(max_chunk_tokens=100)
    text = " ".join(f"word{index}" for index in range(250))

    chunks = analyzer.chunk_text(text, max_tokens=100)

    assert len(chunks) == 3
    assert chunks[0].startswith("word0 word1")
    assert chunks[-1].endswith("word249")


def test_extract_problem_statement_uses_injected_llm_client():
    """An injected LLM client can provide structured problem extraction."""
    client = _FakeOpenAIClient(
        {
            "problem": "How can adaptive chains be certified?",
            "motivation": "Certification matters for Bayesian computation.",
            "research_gap": "Finite-time guarantees are missing.",
            "context": "Probability theory",
            "evidence": ["The introduction states the certification gap."],
            "confidence": "high",
        }
    )

    analyzer = PaperAnalyzer(llm_client=client)

    result = analyzer.extract_problem_statement(PAPER_TEXT, {"title": "Adaptive chains"})

    assert result["source"] == "llm"
    assert result["confidence"] == "high"
    assert result["problem"] == "How can adaptive chains be certified?"
    assert "Return strict JSON" in client.prompts[0]["input"]


def test_extract_problem_statement_parses_chat_completion_shape():
    """LLM parsing accepts common chat-completion response dictionaries."""
    analyzer = PaperAnalyzer(
        llm_client=_FakeOpenAIClient(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "problem": "Which finite-time guarantee is possible?",
                                    "motivation": "Adaptive sampling needs certificates.",
                                    "research_gap": "Existing bounds are limited.",
                                    "context": "Markov chain theory",
                                    "evidence": [],
                                    "confidence": "medium",
                                }
                            )
                        }
                    }
                ]
            }
        )
    )

    result = analyzer.extract_problem_statement(PAPER_TEXT)

    assert result["source"] == "llm"
    assert result["confidence"] == "medium"
    assert result["problem"] == "Which finite-time guarantee is possible?"


def test_problem_extraction_tool_returns_failure_payload_for_bad_input():
    """Tool wrapper captures extraction failures as structured payloads."""
    result = extract_problem_statement_tool("")

    assert result["status"] == "failed"
    assert result["source"] == "tool_error"
    assert "paper_text cannot be empty" in result["error"]


def test_problem_extraction_tool_returns_completed_payload():
    """Tool wrapper returns a completed status around analyzer output."""
    result = extract_problem_statement_tool(
        paper_text=PAPER_TEXT,
        paper_metadata={"title": "Adaptive Markov Chain Guarantees"},
    )

    assert result["status"] == "completed"
    assert "problem" in result
    assert result["sections_used"]


def test_specialized_agents_have_problem_extraction_tool():
    """Problem extraction is registered through the default specialist tool set."""
    agent = ChrisAgent()

    assert "extract_problem_statement_tool" in agent.list_tools()


def test_paper_analyzer_rejects_too_small_chunk_budget():
    """Public constructor validation keeps chunk sizing meaningful."""
    with pytest.raises(ValueError, match="max_chunk_tokens"):
        PaperAnalyzer(max_chunk_tokens=50)
