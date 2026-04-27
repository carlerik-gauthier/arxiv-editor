"""Agent abstractions and tools for the ArXiv editor system."""

from src.agents.base_agent import AgentTool, BaseAgent, ToolCall, ToolResult
from src.agents.julius_agent import (
    AgentHandoff,
    AgentTaskStatus,
    HandoffContext,
    JuliusAgent,
    WorkflowState,
)
from src.agents.specialized_agents import (
    AbdoulayeAgent,
    AlainAgent,
    BrunoAgent,
    ChrisAgent,
    ElisaAgent,
    FelixAgent,
    JeanBaptisteAgent,
    MichelAgent,
    SpecializedAgent,
    create_all_specialized_agents,
    create_specialized_agent,
)

__all__ = [
    "AbdoulayeAgent",
    "AgentHandoff",
    "AgentTaskStatus",
    "AgentTool",
    "AlainAgent",
    "BaseAgent",
    "BrunoAgent",
    "ChrisAgent",
    "ElisaAgent",
    "FelixAgent",
    "HandoffContext",
    "JeanBaptisteAgent",
    "JuliusAgent",
    "MichelAgent",
    "SpecializedAgent",
    "ToolCall",
    "ToolResult",
    "WorkflowState",
    "create_all_specialized_agents",
    "create_specialized_agent",
]
