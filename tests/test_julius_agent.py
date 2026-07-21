"""Unit tests for JuliusAgent's phase-six routing helpers."""

from src import julius_agent
from src.field_family import FieldFamily, family_for_agent


def test_supported_request_detection_handles_specialist_domains(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert julius_agent._is_supported_specialist_request("Two topics on stochastic processes")
    assert julius_agent._is_supported_specialist_request("One topic in group theory")
    assert julius_agent._is_supported_specialist_request("Recent Riemannian geometry papers")
    assert julius_agent._is_supported_specialist_request("Recent machine learning papers")


def test_family_helper_separates_mathematics_and_ai_agents():
    assert family_for_agent("BrunoAgent") is FieldFamily.MATHEMATICS
    assert family_for_agent("AbdoulayeAgent") is FieldFamily.AI


def test_normalize_allocation_discards_invalid_agents():
    allocations = julius_agent._normalize_allocation_payload(
        {"allocations": [{"agent_name": "ChrisAgent", "topic_count": 2}, {"agent_name": "Unknown", "topic_count": 3}]},
        ["ChrisAgent", "AlainAgent"],
    )

    assert allocations == [{"agent_name": "ChrisAgent", "tool_name": "chris_agent_tool", "topic_count": 2, "reason": "Interest in probability or statistics detected."}]


def test_parse_json_object_response_accepts_fenced_json():
    assert julius_agent._parse_json_object_response("```json\n{\"status\": \"compiled\"}\n```", "test") == {"status": "compiled"}


def test_run_julius_agent_declines_out_of_scope_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = julius_agent.run_julius_agent("Cover recent astrophysics papers.")

    assert "coordinate mathematics and AI research briefs" in result["reply"]
    assert result["tool_parameters"] == []
