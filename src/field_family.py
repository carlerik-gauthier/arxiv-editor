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
    """Map a specialist agent name to its top-level research family.

    Args:
        agent_name: Canonical name of a configured specialist agent.

    Returns:
        FieldFamily: Mathematics or AI family represented by ``agent_name``.

    Raises:
        ValueError: If ``agent_name`` is not a configured specialist.
    """
    if agent_name in MATHEMATICS_AGENTS:
        return FieldFamily.MATHEMATICS
    if agent_name in AI_AGENTS:
        return FieldFamily.AI
    raise ValueError(f"Unknown specialist agent: {agent_name}")
