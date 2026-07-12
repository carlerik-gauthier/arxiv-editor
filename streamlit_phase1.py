"""Phase 1 Streamlit app to test ChrisAgent in a chat session."""

from __future__ import annotations

import streamlit as st

from src.chris_agent_phase1 import run_chris_agent


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def run_app() -> None:
    st.set_page_config(page_title="Phase 1 - ChrisAgent", page_icon="📘", layout="wide")
    st.title("Phase 1: ChrisAgent (Probability/Statistics)")
    st.caption("Chat memory is stored in this session.")
    _init_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask ChrisAgent")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("ChrisAgent is working..."):
            result = run_chris_agent(prompt)
        reply = str(result.get("reply", ""))
        st.markdown(reply)
        tool_parameters = result.get("tool_parameters", [])
        if tool_parameters:
            st.markdown("**Tool Parameters**")
            st.json(tool_parameters)

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_app()
