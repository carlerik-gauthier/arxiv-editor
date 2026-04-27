"""Tests for Julius's phase-3.3 coordination and hand-off workflow."""

from src.agents import (
    AgentHandoff,
    AgentTaskStatus,
    ChrisAgent,
    HandoffContext,
    JuliusAgent,
    MichelAgent,
    WorkflowState,
)


def _make_julius():
    """Create Julius with a small deterministic specialist registry."""
    return JuliusAgent(specialist_agents=[MichelAgent(), ChrisAgent()])


def test_julius_initializes_coordination_tools_and_prompt():
    """Julius exposes the step-3.3 coordination tools and editor prompt."""
    julius = _make_julius()

    assert julius.name == "Julius"
    assert julius.workflow_state == WorkflowState.PLANNING
    assert "editor and coordinator" in julius.system_prompt
    assert "delegate_to_agent_tool" in julius.list_tools()
    assert "collect_agent_results_tool" in julius.list_tools()
    assert "compile_one_pager_tool" in julius.list_tools()
    assert julius.agent_task_status["Michel"] == AgentTaskStatus.PENDING


def test_agent_handoff_executes_and_returns_callback():
    """A hand-off transfers context to a specialist and captures completion."""
    julius = _make_julius()
    context = HandoffContext(
        task_description="Explain a geometry result for non-experts.",
        constraints={"audience": "non-experts"},
    )

    handoff = AgentHandoff.execute_handoff(
        from_agent=julius,
        to_agent=julius.specialist_agents["Michel"],
        context=context,
    )

    callback = handoff.callback_on_completion()
    assert callback["from_agent"] == "Julius"
    assert callback["to_agent"] == "Michel"
    assert callback["status"] == AgentTaskStatus.COMPLETED.value
    assert callback["handoff_context"]["constraints"]["audience"] == "non-experts"


def test_delegate_tool_tracks_status_and_handoff_history():
    """Julius delegates through an AgentTool and records status."""
    julius = _make_julius()

    result = julius.execute_tool(
        "delegate_to_agent_tool",
        {
            "agent_name": "Michel",
            "task_description": "Create intuition for curvature.",
            "constraints": {"topic": "Riemannian geometry"},
        },
    )

    assert result.success is True
    assert result.result["status"] == AgentTaskStatus.COMPLETED.value
    assert julius.agent_task_status["Michel"] == AgentTaskStatus.COMPLETED
    assert len(julius.handoffs) == 1
    assert julius.handoffs[0].to_agent == "Michel"


def test_collect_results_aggregates_completed_and_pending_agents():
    """Collection returns completed callbacks and explicit pending records."""
    julius = _make_julius()
    julius.delegate_to_agent_tool("Michel", "Explain mathematical intuition.")

    collected = julius.collect_agent_results_tool(["Michel", "Chris"])

    assert collected["completed_count"] == 1
    assert collected["pending_count"] == 1
    assert collected["missing_agents"] == ["Chris"]
    assert collected["results"][0]["to_agent"] == "Michel"
    assert collected["results"][1]["status"] == AgentTaskStatus.PENDING.value


def test_run_delegated_workflow_delegates_to_multiple_agents_and_compiles():
    """Julius can plan, delegate to two agents, collect, and compile results."""
    julius = _make_julius()

    workflow = julius.run_delegated_workflow(
        user_request="Summarize research from 2026-04-01 to 2026-04-07.",
        agent_names=["Michel", "Chris"],
        topics=["accessible mathematics", "probability"],
        preferences={"audience": "mixed"},
    )

    assert workflow["workflow_state"] == WorkflowState.COMPLETE.value
    assert workflow["agent_results"]["completed_count"] == 2
    assert workflow["one_pager"]["status"] == "compiled"
    assert workflow["one_pager"]["completed_sections"] == 2
    assert "## Michel" in workflow["one_pager"]["content"]
    assert "## Chris" in workflow["one_pager"]["content"]
    assert [entry["state"] for entry in workflow["state_history"]][-1] == "COMPLETE"


def test_extension_request_is_recorded():
    """Julius records requests for more data from specialists."""
    julius = _make_julius()

    request = julius.request_agent_extension_tool(
        agent_name="Chris",
        reason="Only 42 probability papers were found.",
    )

    assert request["agent_name"] == "Chris"
    assert request["status"] == "requested"
    assert julius.extension_requests == [request]


def test_send_email_tool_uses_injected_sender():
    """Email delivery is injectable and side-effect free by default."""
    sent_messages = []

    def fake_sender(recipient, subject, content):
        sent_messages.append((recipient, subject, content))
        return {"message_id": "test-1"}

    julius = JuliusAgent(
        specialist_agents=[MichelAgent()],
        email_sender=fake_sender,
    )

    result = julius.send_email_tool(
        recipient="reader@example.com",
        subject="Brief",
        content="One-pager content",
    )

    assert result["sent"] is True
    assert result["provider_result"]["message_id"] == "test-1"
    assert sent_messages == [("reader@example.com", "Brief", "One-pager content")]
