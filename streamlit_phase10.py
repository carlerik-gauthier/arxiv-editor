"""Phase 10 Streamlit interface for the documented ArXiv editorial workflow."""

from __future__ import annotations

from typing import TypedDict

import streamlit as st

from src.julius_agent import run_julius_agent


class ChatMessage(TypedDict):
    """One persisted chat message in the Streamlit session."""

    role: str
    content: str


def _messages() -> list[ChatMessage]:
    """Return the session conversation, creating the storage when necessary.

    Returns:
        list[ChatMessage]: Mutable list of persisted chat messages for this
        Streamlit session.
    """
    return st.session_state.setdefault("messages", [])


def run_app() -> None:
    """Render and run the Phase 10 conversation-memory chat application.

    Returns:
        None: Renders the interface and updates session-state messages.
    """
    st.set_page_config(page_title="ArXiv Editor", page_icon="📝", layout="wide")
    st.title("ArXiv Editor")
    st.caption("A documented multi-agent workflow for mathematics and AI research one-pagers.")

    with st.sidebar:
        st.subheader("Workflow")
        st.markdown("Julius plans → specialists retrieve and cluster papers → Michel improves accessibility → Julius edits the brief.")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    messages = _messages()
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask for a mathematics or AI ArXiv research brief")
    if not prompt:
        return

    history = list(messages)
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("JuliusAgent is coordinating the research brief..."):
            try:
                result = run_julius_agent(prompt, conversation_history=history)
                reply = str(result.get("reply", "I could not generate a response."))
            except Exception as exc:
                reply = f"The workflow could not complete: {exc}"
        st.markdown(reply)
    messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_app()
