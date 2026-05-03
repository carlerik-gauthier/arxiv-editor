"""Tests for the phase-5.2 key results extraction tool."""

import json

from src.agents import ChrisAgent
from src.agents.tools import extract_key_results_tool
from src.analysis import PaperAnalyzer


MATH_PAPER_TEXT = """
Abstract
We study adaptive Markov chains and prove finite-time convergence guarantees.

1 Introduction
The main result gives a quantitative bound for non-stationary kernels.

2 Results
Theorem 1. For every uniformly ergodic adaptive Markov chain, the total
variation distance is bounded by a computable finite-time rate.
We prove a coupling bound that applies to time-inhomogeneous chains.

3 Conclusion
Our results establish certifiable convergence for adaptive samplers.
""".strip()


ML_PAPER_TEXT = """
Abstract
We introduce a retrieval-augmented agent architecture for theorem proving.

1 Introduction
Benchmarks for agentic theorem proving remain difficult.

2 Experiments
The model outperforms prior state-of-the-art systems by 8 accuracy points on
the ProofBench benchmark.
An ablation shows that memory retrieval improves proof success.

3 Conclusion
We present a practical architecture for agentic proof search.
""".strip()


CRYPTO_PAPER_TEXT = """
Abstract
We propose a lattice-based protocol for private aggregation.

1 Results
We prove the protocol is secure against any polynomial-time adversary under
the learning with errors assumption.
The attack analysis shows that malformed shares can be detected.
""".strip()


def test_paper_analyzer_extracts_math_key_results():
    """Math extraction prioritizes theorem-like formal statements."""
    analyzer = PaperAnalyzer()

    result = analyzer.extract_key_results(
        paper_text=MATH_PAPER_TEXT,
        paper_metadata={"title": "Adaptive Markov Chains", "categories": ["math.PR"]},
    )

    assert result["source"] == "heuristic"
    assert result["confidence"] == "heuristic"
    assert result["domain"] == "math"
    assert result["result_count"] >= 2
    assert result["results"][0]["rank"] == 1
    assert any(item["result_type"] == "theorem" for item in result["results"])
    assert any("finite-time" in item["statement"] for item in result["results"])
    assert "results" in result["sections_used"]


def test_paper_analyzer_extracts_ml_empirical_results():
    """ML extraction surfaces benchmark and ablation findings."""
    analyzer = PaperAnalyzer()

    result = analyzer.extract_key_results(
        paper_text=ML_PAPER_TEXT,
        paper_metadata={"title": "Agentic Proof Search", "categories": ["cs.LG"]},
    )

    assert result["domain"] == "ml"
    assert result["results"][0]["result_type"] == "empirical"
    assert "outperforms" in result["results"][0]["statement"]
    assert result["results"][0]["importance_score"] > 0.7


def test_paper_analyzer_extracts_crypto_security_results():
    """Crypto extraction classifies security and attack results."""
    analyzer = PaperAnalyzer()

    result = analyzer.extract_key_results(
        paper_text=CRYPTO_PAPER_TEXT,
        paper_metadata={"title": "Private Aggregation", "categories": ["cs.CR"]},
    )

    assert result["domain"] == "crypto"
    assert result["results"][0]["result_type"] == "security_guarantee"
    assert "adversary" in result["results"][0]["statement"].lower()
    assert "Security-relevant" in result["results"][0]["significance"]


def test_rank_results_by_importance_assigns_stable_ranks():
    """Public ranking helper normalizes scores and rank fields."""
    analyzer = PaperAnalyzer()

    ranked = analyzer.rank_results_by_importance(
        [
            {"result_type": "summary", "statement": "A contextual observation."},
            {"result_type": "theorem", "statement": "We prove the main theorem."},
        ],
        domain="math",
    )

    assert [item["rank"] for item in ranked] == [1, 2]
    assert ranked[0]["result_type"] == "theorem"
    assert ranked[0]["importance_score"] > ranked[1]["importance_score"]


def test_extract_key_results_uses_injected_llm_client():
    """An injected LLM client can provide structured key results."""
    captured = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return json.dumps(
            {
                "results": [
                    {
                        "result_type": "guarantee",
                        "statement": "The algorithm has a finite-sample guarantee.",
                        "significance": "This gives certified behavior.",
                        "location": "results",
                        "evidence": ["Theorem 2"],
                        "importance_score": 0.91,
                    }
                ],
                "confidence": "high",
            }
        )

    analyzer = PaperAnalyzer(llm_client=fake_llm)

    result = analyzer.extract_key_results(MATH_PAPER_TEXT, domain="math")

    assert result["source"] == "llm"
    assert result["confidence"] == "high"
    assert result["results"][0]["importance_score"] == 0.91
    assert "Extract the key results" in captured["prompt"]


def test_results_extraction_tool_returns_completed_and_failure_payloads():
    """Tool wrapper returns stable completed and failed payloads."""
    completed = extract_key_results_tool(
        paper_text=MATH_PAPER_TEXT,
        paper_metadata={"categories": ["math.PR"]},
    )
    failed = extract_key_results_tool("")

    assert completed["status"] == "completed"
    assert completed["result_count"] >= 1
    assert failed["status"] == "failed"
    assert failed["source"] == "tool_error"
    assert "paper_text cannot be empty" in failed["error"]


def test_specialized_agents_have_results_extraction_tool():
    """Key result extraction is registered through specialist tool defaults."""
    agent = ChrisAgent()

    assert "extract_key_results_tool" in agent.list_tools()
