"""Phase 7 Streamlit app for testing the refactored agent workflow."""

from __future__ import annotations

import streamlit as st

from src.julius_agent import run_julius_agent


def run_app() -> None:
    """Render and run the Phase 7 session-memory JuliusAgent chat.

    Returns:
        None: Renders the interface and updates session-state messages.
    """
    st.set_page_config(page_title="Phase 7 - Refactored Workflow", page_icon="R", layout="wide")
    st.title("Phase 7: Refactored ArXiv Workflow")
    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask for a probability, statistics, or algebra research brief")
    if not prompt:
        return
    history = list(messages)
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("JuliusAgent is coordinating the specialists..."):
            result = run_julius_agent(prompt, conversation_history=history)
        reply = str(result.get("reply", ""))
        st.markdown(reply)
    messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_app()
