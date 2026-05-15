"""Streamlit UI adapter for Julius.

The module is import-safe without Streamlit installed. `run_app` imports
Streamlit only when called, while helper functions operate on plain session
state dictionaries for unit tests.
"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

from src.agents import AgentTool, JeanBaptisteAgent, JuliusAgent, JuliusSession, MichelAgent
from src.agents import openai_tracing
from src.agents.tools import format_document_tool, generate_summary_tool
from src.generation.interactive_workflow import InteractiveSummaryWorkflow


def ensure_workflow_state(
    session_state: MutableMapping[str, Any],
    workflow: Optional[InteractiveSummaryWorkflow] = None,
) -> InteractiveSummaryWorkflow:
    """Initialize persistent Streamlit state without overwriting rerun data."""
    if "workflow" not in session_state:
        session_state["workflow"] = workflow or InteractiveSummaryWorkflow(
            julius_session=JuliusSession(run_specialists_in_preview=True)
        )
    session_state.setdefault("messages", [])
    session_state.setdefault("draft_versions", {})
    session_state.setdefault("validation_reports", [])
    session_state.setdefault("final_output_path", None)
    session_state.setdefault("last_response", None)
    return session_state["workflow"]


def sync_session_state(session_state: MutableMapping[str, Any], workflow: InteractiveSummaryWorkflow) -> None:
    """Copy backend state into Streamlit session state after each action."""
    session = workflow.session
    session_state["draft_versions"] = dict(session.draft_versions)
    session_state["validation_reports"] = list(session.validation_reports)
    session_state["final_output_path"] = session.final_output_path


def collect_agent_activity(workflow: InteractiveSummaryWorkflow) -> Dict[str, Any]:
    """Return concise Julius coordination and specialist activity for the UI."""
    julius = workflow.session.julius
    tool_calls = [
        {
            "agent": "Julius",
            "tool": call.get("tool_name"),
            "success": call.get("success"),
            "error": call.get("error"),
            "completed_at": call.get("completed_at"),
        }
        for call in julius.state.get("tool_calls", [])
    ]
    handoffs = []
    for handoff in julius.handoffs:
        agent_name = handoff.to_agent
        agent = julius.specialist_agents[agent_name]
        response = (handoff.result or {}).get("response", {})
        called_tools = _unique_names(
            [
                *_tool_names_from_agent_state(agent),
                *_tool_names_from_response(response),
            ]
        )
        available_tools = agent.list_tools()
        displayed_tools = called_tools or available_tools
        handoffs.append(
            {
                "agent": agent_name,
                "status": handoff.status.value,
                "task": handoff.handoff_context.task_description,
                "tools": displayed_tools,
                "tool_source": "called" if called_tools else "available",
            }
        )
    return {"julius_tool_calls": tool_calls, "specialist_handoffs": handoffs}


def collect_topic_discovery_debug(workflow: InteractiveSummaryWorkflow) -> Dict[str, Any]:
    """Return raw discover_topics_tool outputs for the debug panel."""
    debug_entries = []
    for agent_name, agent in workflow.session.julius.specialist_agents.items():
        for call in agent.state.get("tool_calls", []):
            if call.get("tool_name") != "discover_topics_tool":
                continue
            debug_entries.append(
                {
                    "agent": agent_name,
                    "success": call.get("success"),
                    "completed_at": call.get("completed_at"),
                    "error": call.get("error"),
                    "content": call.get("result"),
                }
            )
    return {
        "tool": "discover_topics_tool",
        "call_count": len(debug_entries),
        "calls": debug_entries,
    }


def append_chat_result(
    session_state: MutableMapping[str, Any],
    user_message: Optional[str],
    result: Dict[str, Any],
) -> None:
    """Append a user turn and Julius response to chat history."""
    if user_message:
        session_state["messages"].append({"role": "user", "content": user_message})
    session_state["messages"].append(
        {"role": "assistant", "content": _format_assistant_message(result)}
    )
    session_state["last_response"] = result


def _format_assistant_message(result: Dict[str, Any]) -> str:
    """Include Julius clarification questions directly in chat."""
    message = str(result.get("message", ""))
    questions = [question for question in result.get("next_questions", []) if question]
    if not questions:
        return message
    return "\n\n".join(
        [
            message,
            "Julius needs one clarification:"
            if len(questions) == 1
            else "Julius needs a few clarifications:",
            "\n".join(f"- {question}" for question in questions),
        ]
    )


def sidebar_output_preferences(st: Any) -> Dict[str, Any]:
    """Render output-only preferences that complement the chat request."""
    with st.sidebar:
        st.header("Output")
        audience = st.selectbox("Audience", ["mixed audience", "expert", "non-expert"])
        tone = st.selectbox("Tone", ["editorial", "technical", "pedagogical", "executive"])
        output_format = st.selectbox(
            "Format",
            ["one-pager", "bullet digest", "paper rankings", "custom format"],
        )
        custom_structure = ""
        if output_format == "custom format":
            custom_structure = st.text_input("Custom structure", value="")
        delivery = st.selectbox("Delivery", ["preview", "file", "email"])
        email_recipient = ""
        if delivery == "email":
            email_recipient = st.text_input("Email recipient", value="")
    return {
        "audience": audience,
        "tone": tone,
        "format": output_format,
        "custom_structure": custom_structure,
        "delivery": delivery,
        "email_recipient": email_recipient,
    }


def render_chat(
    st: Any,
    session_state: MutableMapping[str, Any],
    workflow: InteractiveSummaryWorkflow,
    output_preferences: Optional[Dict[str, Any]] = None,
) -> None:
    """Render chat history and route new chat input."""
    for message in session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_message = st.chat_input(
        "Tell Julius the topic, date range, categories, and scope"
    )
    if user_message:
        process_chat_input(st, session_state, workflow, user_message, output_preferences)
        st.rerun()


def process_chat_input(
    st: Any,
    session_state: MutableMapping[str, Any],
    workflow: InteractiveSummaryWorkflow,
    user_message: str,
    output_preferences: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Handle one chat turn and auto-start draft generation when ready."""
    trace_metadata = {
        "trace_agent": "Streamlit",
        "workflow_state": workflow.session.state.value,
        "has_output_preferences": bool(output_preferences),
        "user_message": user_message,
    }
    with openai_tracing.trace_workflow("Streamlit Julius chat turn", metadata=trace_metadata):
        julius_message = apply_output_preferences_to_message(
            user_message,
            output_preferences,
            workflow,
        )
        result = workflow.handle_message(julius_message)
        append_chat_result(session_state, user_message, result)
        if should_auto_generate_draft(result, workflow):
            with st.status("Julius is generating the draft", expanded=False):
                result = workflow.generate_draft()
            append_chat_result(session_state, None, result)
        sync_session_state(session_state, workflow)
        return result


def apply_output_preferences_to_message(
    user_message: str,
    output_preferences: Optional[Dict[str, Any]],
    workflow: InteractiveSummaryWorkflow,
) -> str:
    """Fold sidebar output preferences into intake messages for Julius."""
    if not output_preferences or not should_apply_output_preferences(user_message, workflow):
        return user_message
    preference_text = output_preferences_message(output_preferences)
    if not preference_text:
        return user_message
    return f"{user_message}. Output preferences: {preference_text}."


def should_apply_output_preferences(user_message: str, workflow: InteractiveSummaryWorkflow) -> bool:
    """Apply sidebar preferences to request intake, not draft commands or questions."""
    lowered = user_message.lower().strip()
    if lowered.endswith("?"):
        return False
    if any(keyword in lowered for keyword in ("why did", "why choose", "finalize", "finalise", "approve")):
        return False
    return not workflow.session.drafts


def output_preferences_message(output_preferences: Dict[str, Any]) -> str:
    """Build a compact natural-language preference clause for Julius."""
    parts = [
        str(output_preferences.get("audience") or "mixed audience"),
        f"{output_preferences.get('tone') or 'editorial'} tone",
        str(output_preferences.get("format") or "one-pager"),
    ]
    custom_structure = str(output_preferences.get("custom_structure") or "").strip()
    if output_preferences.get("format") == "custom format" and custom_structure:
        parts.append(custom_structure)

    delivery = output_preferences.get("delivery") or "preview"
    email_recipient = str(output_preferences.get("email_recipient") or "").strip()
    if delivery == "email":
        parts.append(f"email to {email_recipient}" if email_recipient else "email")
    elif delivery == "file":
        parts.append("save to file")
    else:
        parts.append("preview")
    return ", ".join(parts)


def should_auto_generate_draft(result: Dict[str, Any], workflow: InteractiveSummaryWorkflow) -> bool:
    """Return true after complete intake, before any draft exists."""
    return (
        result.get("state") == "PLANNING"
        and "updated_summary_request" in result.get("actions_taken", [])
        and not result.get("next_questions")
        and not result.get("recoverable")
        and not workflow.session.drafts
    )


def render_action_buttons(
    st: Any,
    session_state: MutableMapping[str, Any],
    workflow: InteractiveSummaryWorkflow,
) -> None:
    """Render explicit action buttons for expensive workflow steps."""
    col1, col2, col3 = st.columns(3)
    if col1.button("Generate draft"):
        with openai_tracing.trace_workflow(
            "Streamlit Julius generate draft",
            metadata={"trace_agent": "Streamlit", "action": "generate_draft"},
        ):
            with st.status("Generating draft", expanded=False):
                result = workflow.generate_draft()
        append_chat_result(session_state, "Generate draft", result)
        sync_session_state(session_state, workflow)
        st.rerun()
    if col2.button("Validate"):
        with openai_tracing.trace_workflow(
            "Streamlit Julius validate draft",
            metadata={"trace_agent": "Streamlit", "action": "validate_draft"},
        ):
            result = workflow.validate_current_draft()
        append_chat_result(session_state, "Validate draft", result)
        sync_session_state(session_state, workflow)
        st.rerun()
    if col3.button("Finalize"):
        with openai_tracing.trace_workflow(
            "Streamlit Julius finalize draft",
            metadata={"trace_agent": "Streamlit", "action": "finalize"},
        ):
            result = workflow.finalize()
        append_chat_result(session_state, "Finalize", result)
        sync_session_state(session_state, workflow)
        st.rerun()


def render_draft_review(st: Any, workflow: InteractiveSummaryWorkflow) -> None:
    """Render draft preview, metadata, quality, and revision history tabs."""
    session = workflow.session
    draft = session.drafts[-1] if session.drafts else None
    preview, metadata, agents, topic_debug, quality, history = st.tabs(
        ["Preview", "Metadata", "Agents", "Topic debug", "Quality", "Revision history"]
    )
    with preview:
        st.markdown(draft.get("content", "No draft yet.") if draft else "No draft yet.")
        if draft and hasattr(st, "download_button"):
            markdown = format_document_tool(draft, output_format="markdown")["document"]
            html = format_document_tool(draft, output_format="html")["document"]
            st.download_button("Download Markdown", markdown, file_name="julius_summary.md")
            st.download_button("Download HTML", html, file_name="julius_summary.html")
    with metadata:
        st.json(
            {
                "summary_request": session.current_request.model_dump(mode="json") if session.current_request else None,
                "selected_papers": session.selected_papers,
                "final_output_path": session.final_output_path,
            }
        )
    with agents:
        render_agent_activity(st, workflow)
    with topic_debug:
        render_topic_discovery_debug(st, workflow)
    with quality:
        st.json(session.validation_reports[-1] if session.validation_reports else {"status": "not validated"})
    with history:
        st.json(session.draft_versions or {"status": "no draft versions"})


def render_agent_activity(st: Any, workflow: InteractiveSummaryWorkflow) -> None:
    """Render specialists called by Julius and the tools involved."""
    activity = collect_agent_activity(workflow)
    if activity["specialist_handoffs"]:
        st.subheader("Agents Called")
        st.dataframe(
            [
                {
                    "Agent": handoff["agent"],
                    "Status": handoff["status"],
                    "Tools": ", ".join(handoff["tools"]),
                    "Tool source": handoff["tool_source"],
                    "Task": handoff["task"],
                }
                for handoff in activity["specialist_handoffs"]
            ],
            hide_index=True,
        )
    else:
        st.write("No specialist agents have been called yet.")

    if activity["julius_tool_calls"]:
        st.subheader("Julius Tools")
        st.dataframe(
            [
                {
                    "Agent": call["agent"],
                    "Tool": call["tool"],
                    "Success": call["success"],
                    "Error": call["error"],
                    "Completed": call["completed_at"],
                }
                for call in activity["julius_tool_calls"]
            ],
            hide_index=True,
        )
    else:
        st.write("No Julius coordination tools have run yet.")


def render_topic_discovery_debug(st: Any, workflow: InteractiveSummaryWorkflow) -> None:
    """Render raw topic discovery tool payloads for debugging."""
    debug_payload = collect_topic_discovery_debug(workflow)
    if debug_payload["call_count"] == 0:
        st.write("No topic discovery tool output has been recorded yet.")
        return
    st.json(debug_payload)


def _tool_names_from_response(response: Any) -> list[str]:
    """Normalize tool call names from BaseAgent response payloads."""
    if not isinstance(response, dict):
        return []
    tool_calls = response.get("tool_calls") or []
    names = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            name = tool_call.get("name") or tool_call.get("tool") or tool_call.get("tool_name")
        else:
            name = getattr(tool_call, "name", None) or getattr(tool_call, "tool_name", None)
        if name:
            names.append(str(name))
    return names


def _tool_names_from_agent_state(agent: Any) -> list[str]:
    """Return tool names already executed by a specialist agent."""
    return [
        str(call.get("tool_name"))
        for call in agent.state.get("tool_calls", [])
        if call.get("tool_name")
    ]


def _unique_names(names: list[str]) -> list[str]:
    """Deduplicate names while preserving their first-seen order."""
    unique = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _has_streamlit_context() -> bool:
    """Return whether Streamlit attached a script context to this thread."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def run_app(st_module: Optional[Any] = None) -> None:
    """Run the Streamlit app."""
    using_injected_module = st_module is not None
    if st_module is None:
        try:
            import streamlit as st_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install streamlit with `pip install -r requirements.txt`.") from exc

    st = st_module
    if not using_injected_module and not _has_streamlit_context():
        raise RuntimeError("Run the UI with `python app.py` or `streamlit run app.py`.")

    st.set_page_config(page_title="Julius ArXiv Editor", layout="wide")
    st.title("Julius ArXiv Editor")
    workflow = ensure_workflow_state(st.session_state)
    output_preferences = sidebar_output_preferences(st)
    render_action_buttons(st, st.session_state, workflow)
    render_chat(st, st.session_state, workflow, output_preferences)
    render_draft_review(st, workflow)


def build_smoke_workflow() -> InteractiveSummaryWorkflow:
    """Create a workflow with deterministic sample data for smoke tests."""
    return InteractiveSummaryWorkflow(
        julius_session=JuliusSession(
            julius=JuliusAgent(
                specialist_agents=[
                    MichelAgent(),
                    JeanBaptisteAgent(tools=_smoke_specialist_tools()),
                ]
            ),
            selected_papers=[
                {
                    "title": "Agent Planning Benchmarks",
                    "summary": "We study LLM agents and planning.",
                    "arxiv_id": "2605.00001",
                    "score": 1.0,
                }
            ],
            run_specialists_in_preview=True,
        )
    )


def _smoke_specialist_tools() -> list[AgentTool]:
    """Return deterministic specialist tools for UI smoke workflows."""
    def threshold_met(paper_count: int, min_threshold: int = 60) -> Dict[str, Any]:
        return {
            "paper_count": paper_count,
            "min_threshold": min_threshold,
            "threshold_met": True,
            "missing_count": 0,
        }

    def discover_topics(
        papers: Any,
        min_topic_size: int = 2,
        num_topics: Optional[int] = None,
        representative_papers_per_topic: int = 5,
        use_openai_representation: bool = True,
    ) -> Dict[str, Any]:
        paper_list = list(papers)
        return {
            "topics": [
                {
                    "title": "LLM Agent Planning",
                    "description": "LLM-generated smoke topic description.",
                    "description_source": "llm",
                    "keywords": ["agents", "planning"],
                    "paper_count": len(paper_list),
                    "representative_papers": paper_list[:representative_papers_per_topic],
                }
            ][: num_topics or 1],
            "topic_count": 1,
            "paper_count": len(paper_list),
            "status": "completed",
        }

    return [
        AgentTool(
            name="check_threshold_tool",
            description="Check whether enough papers are available.",
            function=threshold_met,
            required_parameters=["paper_count", "min_threshold"],
        ),
        AgentTool(
            name="discover_topics_tool",
            description="Discover topics.",
            function=discover_topics,
            required_parameters=["papers"],
        ),
        AgentTool(
            name="generate_summary_tool",
            description="Summarize representative papers.",
            function=generate_summary_tool,
            required_parameters=["papers", "topic"],
        ),
    ]
