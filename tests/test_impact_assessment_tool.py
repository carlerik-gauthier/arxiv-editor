"""Tests for the phase-5.3 impact assessment tool."""

import json

from src.agents import ChrisAgent
from src.agents.tools import assess_impact_tool
from src.analysis import PaperAnalyzer


IMPACT_PAPER = {
    "title": "Finite-Time Guarantees for Adaptive Markov Chains",
    "categories": ["math.PR"],
    "summary": (
        "We introduce a new coupling technique for adaptive Markov chains and "
        "resolve a long-standing open problem about finite-time convergence."
    ),
    "full_text": """
Abstract
We introduce a new coupling technique for adaptive Markov chains and resolve a
long-standing open problem about finite-time convergence.

1 Introduction
Adaptive samplers are used in Bayesian computation and simulation. However,
their finite-time behavior has remained open.

2 Results
Theorem 1. For every uniformly ergodic adaptive chain, the total variation
distance is bounded by a computable finite-time rate.

3 Discussion
The method has applications in Bayesian computation and certified simulation.
""".strip(),
}


KEY_RESULTS = [
    {
        "result_type": "theorem",
        "statement": (
            "Theorem 1 gives a computable finite-time rate for uniformly "
            "ergodic adaptive chains."
        ),
        "significance": "Formal convergence guarantee.",
        "location": "results",
        "importance_score": 0.93,
    }
]


class _FakeOpenAIClient:
    """Minimal OpenAI-compatible client for impact-assessment tests."""

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


def test_paper_analyzer_assesses_impact_with_heuristics():
    """Impact assessment detects novelty, open problems, techniques, and applications."""
    analyzer = PaperAnalyzer()

    result = analyzer.assess_impact(
        paper=IMPACT_PAPER,
        results=KEY_RESULTS,
        field_context="Adaptive MCMC convergence is a known certification bottleneck.",
    )

    assert result["source"] == "heuristic"
    assert result["confidence"] == "heuristic"
    assert result["domain"] == "math"
    assert result["novelty_score"] > 0.7
    assert result["solves_open_problem"] is True
    assert result["introduces_new_techniques"] is True
    assert result["community_impact"] == "high"
    assert result["potential_applications"]
    assert "Bayesian" in " ".join(result["potential_applications"])
    assert "open" in result["impact_summary"]


def test_generate_impact_narrative_documents_assessment():
    """Public narrative helper converts structured scores into readable text."""
    analyzer = PaperAnalyzer()

    narrative = analyzer.generate_impact_narrative(
        {
            "community_impact": "moderate",
            "novelty_score": 0.61,
            "solves_open_problem": False,
            "introduces_new_techniques": True,
            "potential_applications": ["benchmark design"],
        }
    )

    assert "moderate" in narrative
    assert "0.61" in narrative
    assert "technique" in narrative
    assert "benchmark design" in narrative


def test_assess_impact_uses_injected_llm_client():
    """An injected LLM client can provide structured impact assessment."""
    client = _FakeOpenAIClient(
        {
            "novelty_score": 0.88,
            "solves_open_problem": True,
            "introduces_new_techniques": True,
            "potential_applications": ["certified adaptive sampling"],
            "community_impact": "high",
            "community_impact_score": 0.91,
            "impact_summary": "The paper likely matters because it closes a certification gap.",
            "evidence": ["long-standing open problem"],
            "confidence": "high",
        }
    )

    analyzer = PaperAnalyzer(llm_client=client)

    result = analyzer.assess_impact(IMPACT_PAPER, KEY_RESULTS)

    assert result["source"] == "llm"
    assert result["confidence"] == "high"
    assert result["community_impact_score"] == 0.91
    assert result["potential_applications"] == ["certified adaptive sampling"]
    assert "Assess this paper" in client.prompts[0]["input"]


def test_impact_assessment_tool_returns_completed_and_failure_payloads():
    """Tool wrapper returns stable completed and failed payloads."""
    completed = assess_impact_tool(
        paper=IMPACT_PAPER,
        results=KEY_RESULTS,
        field_context="Adaptive MCMC certification",
    )
    failed = assess_impact_tool(paper={})

    assert completed["status"] == "completed"
    assert completed["community_impact_score"] > 0
    assert failed["status"] == "failed"
    assert failed["source"] == "tool_error"
    assert "paper cannot be empty" in failed["error"]


def test_specialized_agents_have_impact_assessment_tool():
    """Impact assessment is registered through specialist tool defaults."""
    agent = ChrisAgent()

    assert "assess_impact_tool" in agent.list_tools()
