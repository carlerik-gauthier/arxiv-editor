"""Routing helper for the mathematics and AI specialist families."""

from __future__ import annotations

from enum import StrEnum


class FieldFamily(StrEnum):
    """Top-level research families supported by the specialist team."""

    MATHEMATICS = "mathematics"
    AI = "ai"


MATHEMATICS_AGENTS = frozenset({"ChrisAgent", "AlainAgent", "BrunoAgent", "ElisaAgent", "FelixAgent"})
AI_AGENTS = frozenset({"AbdoulayeAgent", "JeanBaptisteAgent"})


def family_for_agent(agent_name: str) -> FieldFamily:
    """Return the research family represented by a specialist agent."""
    if agent_name in MATHEMATICS_AGENTS:
        return FieldFamily.MATHEMATICS
    if agent_name in AI_AGENTS:
        return FieldFamily.AI
    raise ValueError(f"Unknown specialist agent: {agent_name}")
