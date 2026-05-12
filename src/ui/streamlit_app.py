"""Streamlit UI adapter for Julius.

The module is import-safe without Streamlit installed. `run_app` imports
Streamlit only when called, while helper functions operate on plain session
state dictionaries for unit tests.
"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping

from src.agents import JuliusSession
from src.agents.tools import format_document_tool
from src.generation.interactive_workflow import InteractiveSummaryWorkflow


DEFAULT_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "cs.CR",
    "math.PR",
    "math.AG",
    "math.DG",
    "stat.ML",
]


def ensure_workflow_state(
    session_state: MutableMapping[str, Any],
    workflow: InteractiveSummaryWorkflow | None = None,
) -> InteractiveSummaryWorkflow:
    """Initialize persistent Streamlit state without overwriting rerun data."""
    if "workflow" not in session_state:
        session_state["workflow"] = workflow or InteractiveSummaryWorkflow()
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


def append_chat_result(
    session_state: MutableMapping[str, Any],
    user_message: str | None,
    result: Dict[str, Any],
) -> None:
    """Append a user turn and Julius response to chat history."""
    if user_message:
        session_state["messages"].append({"role": "user", "content": user_message})
    session_state["messages"].append({"role": "assistant", "content": result["message"]})
    session_state["last_response"] = result


def sidebar_preferences(st: Any) -> Dict[str, Any]:
    """Render sidebar preference controls and return a preference dict."""
    with st.sidebar:
        st.header("Preferences")
        topic_query = st.text_input("Topic", value="")
        quick_range = st.selectbox("Date range", ["last week", "last month", "last 14 days", "custom"])
        audience = st.selectbox("Audience", ["mixed", "expert", "non_expert"])
        depth = st.selectbox("Depth", ["standard", "brief", "deep"])
        tone = st.selectbox("Tone", ["editorial", "technical", "pedagogical", "executive"])
        output_format = st.selectbox("Format", ["one_pager", "bullet_digest", "paper_rankings", "custom"])
        include = st.multiselect("Include categories", DEFAULT_CATEGORIES)
        exclude = st.multiselect("Exclude categories", DEFAULT_CATEGORIES)
        max_topics = st.slider("Max topics", min_value=1, max_value=20, value=5)
        max_papers = st.slider("Max papers", min_value=1, max_value=50, value=10)
    return {
        "topic_query": topic_query or None,
        "date_range": quick_range,
        "audience": audience,
        "depth": depth,
        "tone": tone,
        "format": output_format,
        "must_include_categories": include,
        "exclude_categories": exclude,
        "max_topics": max_topics,
        "max_papers": max_papers,
    }


def render_chat(st: Any, session_state: MutableMapping[str, Any], workflow: InteractiveSummaryWorkflow) -> None:
    """Render chat history and route new chat input."""
    for message in session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_message = st.chat_input("Ask Julius for a research summary")
    if user_message:
        result = workflow.handle_message(user_message)
        append_chat_result(session_state, user_message, result)
        sync_session_state(session_state, workflow)
        st.rerun()


def render_action_buttons(
    st: Any,
    session_state: MutableMapping[str, Any],
    workflow: InteractiveSummaryWorkflow,
    preferences: Dict[str, Any],
) -> None:
    """Render explicit action buttons for expensive workflow steps."""
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Apply preferences"):
        result = workflow.apply_preferences(preferences)
        append_chat_result(session_state, "Apply sidebar preferences", result)
        sync_session_state(session_state, workflow)
        st.rerun()
    if col2.button("Generate draft"):
        with st.status("Generating draft", expanded=False):
            result = workflow.generate_draft()
        append_chat_result(session_state, "Generate draft", result)
        sync_session_state(session_state, workflow)
        st.rerun()
    if col3.button("Validate"):
        result = workflow.validate_current_draft()
        append_chat_result(session_state, "Validate draft", result)
        sync_session_state(session_state, workflow)
        st.rerun()
    if col4.button("Finalize"):
        result = workflow.finalize()
        append_chat_result(session_state, "Finalize", result)
        sync_session_state(session_state, workflow)
        st.rerun()


def render_draft_review(st: Any, workflow: InteractiveSummaryWorkflow) -> None:
    """Render draft preview, metadata, quality, and revision history tabs."""
    session = workflow.session
    draft = session.drafts[-1] if session.drafts else None
    preview, metadata, quality, history = st.tabs(["Preview", "Metadata", "Quality", "Revision history"])
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
    with quality:
        st.json(session.validation_reports[-1] if session.validation_reports else {"status": "not validated"})
    with history:
        st.json(session.draft_versions or {"status": "no draft versions"})


def run_app(st_module: Any | None = None) -> None:
    """Run the Streamlit app."""
    if st_module is None:
        try:
            import streamlit as st_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install streamlit with `pip install -r requirements.txt`.") from exc

    st = st_module
    st.set_page_config(page_title="Julius ArXiv Editor", layout="wide")
    st.title("Julius ArXiv Editor")
    workflow = ensure_workflow_state(st.session_state)
    preferences = sidebar_preferences(st)
    render_action_buttons(st, st.session_state, workflow, preferences)
    render_chat(st, st.session_state, workflow)
    render_draft_review(st, workflow)


def build_smoke_workflow() -> InteractiveSummaryWorkflow:
    """Create a workflow with deterministic sample data for smoke tests."""
    return InteractiveSummaryWorkflow(
        julius_session=JuliusSession(
            selected_papers=[
                {
                    "title": "Agent Planning Benchmarks",
                    "summary": "We study LLM agents and planning.",
                    "arxiv_id": "2605.00001",
                    "score": 1.0,
                }
            ]
        )
    )
