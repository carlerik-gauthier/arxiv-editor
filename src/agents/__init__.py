"""Agent abstractions and tools for the ArXiv editor system."""

from src.agents.base_agent import AgentTool, BaseAgent, ToolCall, ToolResult
from src.agents.julius_agent import (
    AgentHandoff,
    AgentTaskStatus,
    HandoffContext,
    JuliusAgent,
    WorkflowState,
)
from src.agents.julius_session import (
    JuliusIntent,
    JuliusSession,
    JuliusSessionResponse,
    JuliusSessionState,
    classify_user_intent_tool,
    explain_draft_choice_tool,
    update_summary_request_tool,
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
    "JuliusIntent",
    "JuliusSession",
    "JuliusSessionResponse",
    "JuliusSessionState",
    "MichelAgent",
    "SpecializedAgent",
    "ToolCall",
    "ToolResult",
    "WorkflowState",
    "classify_user_intent_tool",
    "create_all_specialized_agents",
    "create_specialized_agent",
    "explain_draft_choice_tool",
    "update_summary_request_tool",
]
