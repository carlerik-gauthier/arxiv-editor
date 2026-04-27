"""Julius coordinator agent and hand-off workflow primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional
from uuid import uuid4

from src.agents.base_agent import AgentTool, BaseAgent
from src.agents.specialized_agents import (
    SpecializedAgent,
    create_all_specialized_agents,
)


class WorkflowState(str, Enum):
    """High-level states for Julius's editorial workflow."""

    PLANNING = "PLANNING"
    DELEGATING = "DELEGATING"
    COLLECTING = "COLLECTING"
    COMPILING = "COMPILING"
    REVIEWING = "REVIEWING"
    COMPLETE = "COMPLETE"


class AgentTaskStatus(str, Enum):
    """Task lifecycle states for delegated specialist work."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class HandoffContext:
    """
    Serializable context transferred from Julius to a specialist agent.

    Args:
        task_description: Concrete work requested from the specialist.
        constraints: Operational limits such as date range, audience, or topics.
        previous_results: Results or notes that should inform the delegated task.
    """

    task_description: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    previous_results: List[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the hand-off context for logs, prompts, and tool output."""
        return {
            "task_description": self.task_description,
            "constraints": self.constraints,
            "previous_results": self.previous_results,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AgentHandoff:
    """
    Executable hand-off record between Julius and a specialist agent.

    The current implementation executes synchronously for determinism in phase
    3.3. The record shape preserves enough metadata to replace execution with a
    queue or parallel worker pool later without changing Julius's public tools.
    """

    handoff_context: HandoffContext
    from_agent: str
    to_agent: str
    handoff_id: str = field(default_factory=lambda: str(uuid4()))
    status: AgentTaskStatus = AgentTaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @classmethod
    def execute_handoff(
        cls,
        from_agent: BaseAgent,
        to_agent: BaseAgent,
        context: HandoffContext,
    ) -> "AgentHandoff":
        """
        Transfer task context to a specialist and capture the callback result.

        The delegated agent receives a structured prompt and uses its normal
        `respond` method, so LLM-backed and deterministic agents share the same
        workflow path.
        """
        handoff = cls(
            handoff_context=context,
            from_agent=from_agent.name,
            to_agent=to_agent.name,
        )
        handoff.started_at = datetime.now(timezone.utc)
        handoff.status = AgentTaskStatus.IN_PROGRESS

        try:
            response = to_agent.respond(
                _format_handoff_prompt(from_agent.name, context),
                auto_execute_tools=False,
            )
            handoff.result = {
                "agent": to_agent.name,
                "categories": list(getattr(to_agent, "categories", [])),
                "response": response,
                "handoff_context": context.to_dict(),
            }
            handoff.status = AgentTaskStatus.COMPLETED
        except Exception as exc:
            handoff.error = str(exc)
            handoff.status = AgentTaskStatus.FAILED
        finally:
            handoff.completed_at = datetime.now(timezone.utc)

        return handoff

    def callback_on_completion(self) -> Dict[str, Any]:
        """Return a structured callback payload for Julius to collect."""
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the hand-off record."""
        return {
            "handoff_id": self.handoff_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "status": self.status.value,
            "handoff_context": self.handoff_context.to_dict(),
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class JuliusAgent(BaseAgent):
    """
    Editor and coordinator agent responsible for delegation and synthesis.

    Julius owns the workflow state machine, a registry of specialist agents, and
    coordination tools for delegation, collection, one-pager compilation, date
    extension requests, and optional email delivery.
    """

    def __init__(
        self,
        specialist_agents: Optional[Iterable[SpecializedAgent]] = None,
        llm_client: Optional[Any] = None,
        email_sender: Optional[Callable[..., Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.workflow_state = WorkflowState.PLANNING
        self.state_history: List[Dict[str, str]] = []
        self.handoffs: List[AgentHandoff] = []
        self.agent_task_status: Dict[str, AgentTaskStatus] = {}
        self.extension_requests: List[Dict[str, Any]] = []
        self.email_sender = email_sender
        self.specialist_agents = self._build_agent_registry(
            specialist_agents or create_all_specialized_agents()
        )

        for agent_name in self.specialist_agents:
            self.agent_task_status[agent_name] = AgentTaskStatus.PENDING

        super().__init__(
            name="Julius",
            expertise="Editorial coordination, planning, delegation, and synthesis",
            categories=[],
            llm_client=llm_client,
            system_prompt=system_prompt or self._build_julius_system_prompt(),
            tools=self._build_coordination_tools(),
        )
        self._record_state(self.workflow_state)

    def _build_julius_system_prompt(self) -> str:
        """Create Julius's editor-and-coordinator system prompt."""
        specialist_summary = ", ".join(
            f"{agent.name} ({', '.join(agent.categories)})"
            for agent in self.specialist_agents.values()
        )
        return (
            "You are Julius, the editor and coordinator of the ArXiv research "
            "publishing team. You parse user requests, plan the editorial workflow, "
            "delegate domain-specific analysis to specialist agents, collect their "
            "results, handle partial failures transparently, and compile a clear "
            "one-pager for expert and non-expert readers.\n\n"
            f"Available specialists: {specialist_summary}.\n"
            "Use coordination tools when you need to delegate, request a date-range "
            "extension, collect specialist callbacks, compile the one-pager, or "
            "send the final result."
        )

    def _build_coordination_tools(self) -> List[AgentTool]:
        """Register Julius's specialized coordination tools."""
        return [
            AgentTool(
                name="delegate_to_agent_tool",
                description="Hand off a concrete task to a specialized research agent.",
                function=self.delegate_to_agent_tool,
                required_parameters=["agent_name", "task_description"],
            ),
            AgentTool(
                name="request_agent_extension_tool",
                description="Record that a specialist needs a wider date range.",
                function=self.request_agent_extension_tool,
                required_parameters=["agent_name", "reason"],
            ),
            AgentTool(
                name="collect_agent_results_tool",
                description="Collect completed, failed, or pending specialist callbacks.",
                function=self.collect_agent_results_tool,
                required_parameters=["agent_names"],
            ),
            AgentTool(
                name="compile_one_pager_tool",
                description="Synthesize specialist callbacks into a one-pager draft.",
                function=self.compile_one_pager_tool,
                required_parameters=["agent_results"],
            ),
            AgentTool(
                name="send_email_tool",
                description="Deliver or queue the compiled one-pager by email.",
                function=self.send_email_tool,
                required_parameters=["recipient", "content"],
            ),
        ]

    def parse_user_request(
        self,
        user_request: str,
        date_range: Optional[Dict[str, str]] = None,
        topics: Optional[List[str]] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parse the user request into workflow inputs Julius can plan from.

        This phase-3 parser is intentionally conservative and deterministic. It
        keeps the raw request, extracts simple ISO dates when present, and leaves
        richer language understanding to the injected LLM layer in later phases.
        """
        extracted_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", user_request)
        parsed_date_range = dict(date_range or {})
        if extracted_dates and "start_date" not in parsed_date_range:
            parsed_date_range["start_date"] = extracted_dates[0]
        if len(extracted_dates) > 1 and "end_date" not in parsed_date_range:
            parsed_date_range["end_date"] = extracted_dates[1]

        return {
            "raw_request": user_request,
            "date_range": parsed_date_range,
            "topics": topics or [],
            "preferences": preferences or {},
        }

    def create_execution_plan(
        self,
        parsed_request: Dict[str, Any],
        agent_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a delegation plan from a parsed user request.

        Agents are marked as one parallel group because their domain analysis is
        independent at this stage, even though this phase executes them
        synchronously for simple local testing.
        """
        selected_agent_names = [
            self._resolve_agent_name(agent_name)
            for agent_name in (agent_names or self.specialist_agents.keys())
        ]
        assignments = [
            {
                "agent_name": agent_name,
                "task_description": self._build_task_description(
                    self.specialist_agents[agent_name],
                    parsed_request,
                ),
                "constraints": {
                    "date_range": parsed_request.get("date_range", {}),
                    "topics": parsed_request.get("topics", []),
                    "preferences": parsed_request.get("preferences", {}),
                },
            }
            for agent_name in selected_agent_names
        ]

        return {
            "request": parsed_request,
            "assignments": assignments,
            "parallel_groups": [selected_agent_names],
        }

    def run_delegated_workflow(
        self,
        user_request: str,
        agent_names: Optional[Iterable[str]] = None,
        date_range: Optional[Dict[str, str]] = None,
        topics: Optional[List[str]] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Julius's phase-3.3 workflow from planning through compilation.

        This method is a compact integration path for tests and early CLI
        experiments. Future phases can swap the internals for asynchronous task
        execution while preserving the state and output contracts.
        """
        self._transition_to(WorkflowState.PLANNING)
        parsed_request = self.parse_user_request(
            user_request=user_request,
            date_range=date_range,
            topics=topics,
            preferences=preferences,
        )
        plan = self.create_execution_plan(parsed_request, agent_names=agent_names)

        self._transition_to(WorkflowState.DELEGATING)
        for assignment in plan["assignments"]:
            self.delegate_to_agent_tool(**assignment)

        self._transition_to(WorkflowState.COLLECTING)
        agent_results = self.collect_agent_results_tool(
            [assignment["agent_name"] for assignment in plan["assignments"]]
        )

        self._transition_to(WorkflowState.COMPILING)
        one_pager = self.compile_one_pager_tool(agent_results)

        self._transition_to(WorkflowState.REVIEWING)
        review = {
            "status": "ready_for_delivery",
            "completed_agents": agent_results["completed_count"],
            "failed_agents": agent_results["failed_count"],
        }

        self._transition_to(WorkflowState.COMPLETE)
        return {
            "request": parsed_request,
            "plan": plan,
            "agent_results": agent_results,
            "one_pager": one_pager,
            "review": review,
            "workflow_state": self.workflow_state.value,
            "state_history": list(self.state_history),
        }

    def delegate_to_agent_tool(
        self,
        agent_name: str,
        task_description: str,
        constraints: Optional[Dict[str, Any]] = None,
        previous_results: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Hand off a task to a specialist agent and track its lifecycle."""
        resolved_name = self._resolve_agent_name(agent_name)
        target_agent = self.specialist_agents[resolved_name]
        context = HandoffContext(
            task_description=task_description,
            constraints=constraints or {},
            previous_results=previous_results or [],
        )

        self.agent_task_status[resolved_name] = AgentTaskStatus.IN_PROGRESS
        handoff = AgentHandoff.execute_handoff(self, target_agent, context)
        self.handoffs.append(handoff)
        self.agent_task_status[resolved_name] = handoff.status
        self._add_message("handoff", handoff.to_dict())

        return handoff.callback_on_completion()

    def request_agent_extension_tool(self, agent_name: str, reason: str) -> Dict[str, Any]:
        """Record that a specialist needs a broader date range or more papers."""
        resolved_name = self._resolve_agent_name(agent_name)
        request = {
            "agent_name": resolved_name,
            "reason": reason,
            "status": "requested",
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        self.extension_requests.append(request)
        self._add_message("extension_request", request)
        return request

    def collect_agent_results_tool(self, agent_names: Iterable[str]) -> Dict[str, Any]:
        """
        Gather callback payloads for selected specialists.

        Missing or not-yet-completed work is reported explicitly so Julius can
        compile partial results without silently dropping failures.
        """
        requested_names = [self._resolve_agent_name(agent_name) for agent_name in agent_names]
        latest_by_agent = self._latest_handoffs_by_agent()

        callbacks: List[Dict[str, Any]] = []
        missing_agents: List[str] = []
        for agent_name in requested_names:
            handoff = latest_by_agent.get(agent_name)
            if handoff is None:
                missing_agents.append(agent_name)
                callbacks.append(
                    {
                        "to_agent": agent_name,
                        "status": AgentTaskStatus.PENDING.value,
                        "result": None,
                        "error": "No handoff has been executed for this agent.",
                    }
                )
                continue
            callbacks.append(handoff.callback_on_completion())

        completed = [
            callback
            for callback in callbacks
            if callback["status"] == AgentTaskStatus.COMPLETED.value
        ]
        failed = [
            callback
            for callback in callbacks
            if callback["status"] == AgentTaskStatus.FAILED.value
        ]

        return {
            "requested_agents": requested_names,
            "completed_count": len(completed),
            "failed_count": len(failed),
            "pending_count": len(callbacks) - len(completed) - len(failed),
            "missing_agents": missing_agents,
            "results": callbacks,
        }

    def compile_one_pager_tool(
        self,
        agent_results: Any,
        title: str = "ArXiv Research Brief",
    ) -> Dict[str, Any]:
        """
        Synthesize specialist callbacks into a deterministic one-pager draft.

        Later phases can replace this with richer LLM generation while keeping
        the same structured output fields for Julius's workflow.
        """
        normalized_results = self._normalize_agent_results(agent_results)
        completed_sections: List[str] = []
        failed_sections: List[str] = []

        for callback in normalized_results:
            agent_name = callback.get("to_agent") or callback.get("agent", "Unknown")
            if callback.get("status") != AgentTaskStatus.COMPLETED.value:
                failed_sections.append(
                    f"- {agent_name}: unavailable ({callback.get('error', 'not completed')})"
                )
                continue

            context = callback.get("handoff_context", {})
            task = context.get("task_description", "No task description provided.")
            response = callback.get("result", {}).get("response", {})
            response_text = response.get("response") if isinstance(response, dict) else response
            completed_sections.append(
                f"## {agent_name}\n"
                f"Task: {task}\n"
                f"Result: {response_text}"
            )

        content_parts = [
            f"# {title}",
            "Julius coordinated the specialist review and collected the following callbacks.",
        ]
        if completed_sections:
            content_parts.append("\n\n".join(completed_sections))
        if failed_sections:
            content_parts.append("## Partial or Failed Work\n" + "\n".join(failed_sections))

        return {
            "title": title,
            "content": "\n\n".join(content_parts),
            "completed_sections": len(completed_sections),
            "failed_sections": len(failed_sections),
            "status": "compiled",
        }

    def send_email_tool(
        self,
        recipient: str,
        content: str,
        subject: str = "ArXiv Research Brief",
    ) -> Dict[str, Any]:
        """
        Deliver the one-pager with an injectable sender or return queued status.

        No SMTP connection is opened unless an `email_sender` callable was
        provided to Julius. This keeps phase-3 tests side-effect free while
        preserving the delivery contract for later integration.
        """
        if not recipient or "@" not in recipient:
            raise ValueError("recipient must be a valid email-like address")
        if not content:
            raise ValueError("content cannot be empty")

        if self.email_sender is None:
            return {
                "recipient": recipient,
                "subject": subject,
                "status": "queued",
                "sent": False,
                "reason": "No email_sender configured.",
            }

        try:
            delivery_result = self.email_sender(
                recipient=recipient,
                subject=subject,
                content=content,
            )
        except TypeError:
            delivery_result = self.email_sender(recipient, subject, content)

        return {
            "recipient": recipient,
            "subject": subject,
            "status": "sent",
            "sent": True,
            "provider_result": delivery_result,
        }

    def _transition_to(self, next_state: WorkflowState) -> None:
        """Move Julius to the next workflow state and record the transition."""
        self.workflow_state = next_state
        self._record_state(next_state)

    def _record_state(self, state: WorkflowState) -> None:
        """Append a workflow state transition to Julius's state history."""
        self.state_history.append(
            {
                "state": state.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _build_task_description(
        self,
        agent: SpecializedAgent,
        parsed_request: Dict[str, Any],
    ) -> str:
        """Build a concrete specialist task for an execution plan."""
        topics = parsed_request.get("topics") or ["your assigned categories"]
        return (
            f"Review recent ArXiv work in {', '.join(agent.categories)} for "
            f"{', '.join(topics)}. User request: {parsed_request['raw_request']}"
        )

    def _latest_handoffs_by_agent(self) -> Dict[str, AgentHandoff]:
        """Return the latest hand-off for each specialist by resolved name."""
        latest: Dict[str, AgentHandoff] = {}
        for handoff in self.handoffs:
            latest[handoff.to_agent] = handoff
        return latest

    def _normalize_agent_results(self, agent_results: Any) -> List[Dict[str, Any]]:
        """Accept collect-tool output or a direct list of callbacks."""
        if isinstance(agent_results, dict) and "results" in agent_results:
            return list(agent_results["results"])
        if isinstance(agent_results, list):
            return list(agent_results)
        raise TypeError("agent_results must be collect_agent_results_tool output or a list")

    def _resolve_agent_name(self, agent_name: str) -> str:
        """Resolve user-facing specialist names to registered canonical names."""
        normalized_name = _normalize_agent_name(agent_name)
        for registered_name in self.specialist_agents:
            if _normalize_agent_name(registered_name) == normalized_name:
                return registered_name
        available = ", ".join(sorted(self.specialist_agents))
        raise ValueError(f"Unknown specialist agent '{agent_name}'. Available: {available}")

    @staticmethod
    def _build_agent_registry(
        agents: Iterable[SpecializedAgent],
    ) -> Dict[str, SpecializedAgent]:
        """Create a canonical-name registry for specialist agents."""
        registry: Dict[str, SpecializedAgent] = {}
        for agent in agents:
            registry[agent.name] = agent
        return registry


def _format_handoff_prompt(from_agent_name: str, context: HandoffContext) -> str:
    """Format hand-off context as a clear delegated task prompt."""
    return (
        f"{from_agent_name} is delegating a research task.\n"
        f"Task: {context.task_description}\n"
        f"Constraints: {context.constraints}\n"
        f"Previous results: {context.previous_results}\n"
        "Return a concise specialist callback with useful findings, blockers, "
        "and any need for additional data."
    )


def _normalize_agent_name(agent_name: str) -> str:
    """Normalize agent names for case-insensitive and punctuation-insensitive lookup."""
    return "".join(character for character in agent_name.lower() if character.isalnum())
