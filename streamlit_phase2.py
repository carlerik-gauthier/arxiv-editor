"""Phase 2 Streamlit app to test ChrisAgent with topic/result tools."""

from __future__ import annotations

import streamlit as st

from src_new.chris_agent_phase2 import run_chris_agent_phase2


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def run_app() -> None:
    st.set_page_config(page_title="Phase 2 - ChrisAgent", page_icon="📗", layout="wide")
    st.title("Phase 2: ChrisAgent (Topics + Main Results)")
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
            result = run_chris_agent_phase2(prompt)
        reply = str(result.get("reply", ""))
        st.markdown(reply)
        tool_parameters = result.get("tool_parameters", [])
        if tool_parameters:
            st.markdown("**Tool Parameters**")
            st.json(tool_parameters)

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_app()
