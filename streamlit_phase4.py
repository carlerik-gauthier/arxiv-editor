"""Phase 4 Streamlit app to test JuliusAgent in a chat session."""

from __future__ import annotations

import logging
import warnings

import streamlit as st

from src.julius_agent import run_julius_agent


class _TransformersPathAliasFilter(logging.Filter):
    """Drop known noisy Transformers alias warnings about `__path__`."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a log record should remain visible.

        Args:
            record: Log record emitted by the Transformers logger.

        Returns:
            bool: ``False`` for the known noisy alias warning; otherwise ``True``.
        """
        msg = record.getMessage()
        return not (
            "Accessing `__path__`" in msg
            and "Returning `__path__` instead" in msg
            and "alias will be removed in future versions" in msg
        )


warnings.filterwarnings(
    "ignore",
    message=r".*Returning `__path__` instead\..*alias will be removed in future versions\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"'cgi' is deprecated and slated for removal in Python 3\.13",
    category=DeprecationWarning,
)
logging.getLogger("transformers").addFilter(_TransformersPathAliasFilter())


def _init_state() -> None:
    """Initialize the current session's chat transcript when absent.

    Returns:
        None: Creates an empty ``messages`` list in Streamlit session state.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []


def run_app() -> None:
    """Render and run the Phase 4 JuliusAgent chat interface.

    Returns:
        None: Renders the interface and updates session-state messages.
    """
    st.set_page_config(page_title="Phase 4 - JuliusAgent", page_icon="📰", layout="wide")
    st.title("Phase 4: JuliusAgent (Editorial Coordinator)")
    st.caption("Chat memory is stored in this session.")
    _init_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask JuliusAgent")
    if not prompt:
        return

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("JuliusAgent is coordinating..."):
            result = run_julius_agent(prompt, conversation_history=history)
        reply = str(result.get("reply", ""))
        st.markdown(reply)
        tool_parameters = result.get("tool_parameters", [])
        if tool_parameters:
            st.markdown("**Tool Parameters**")
            st.json(tool_parameters)

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_app()
