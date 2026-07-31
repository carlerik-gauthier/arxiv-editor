"""Phase 8 Streamlit app for the full mathematics and AI specialist team."""

from __future__ import annotations

import streamlit as st

from src.julius_agent import run_julius_agent


def run_app() -> None:
    """Render and run the Phase 8 full-specialist chat interface.

    Returns:
        None: Renders the interface and updates session-state messages.
    """
    st.set_page_config(page_title="Phase 8 - Mathematics & AI", page_icon="🧭", layout="wide")
    st.title("Phase 8: Mathematics & AI ArXiv Workflow")
    st.caption("Ask for a research brief in mathematics, machine learning, data science, NLP, LLMs, or agentic AI.")

    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask for an ArXiv research brief")
    if not prompt:
        return

    history = list(messages)
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("JuliusAgent is coordinating the specialist team..."):
            result = run_julius_agent(prompt, conversation_history=history)
        reply = str(result.get("reply", ""))
        st.markdown(reply)
    messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_app()
