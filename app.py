"""Local Streamlit entry point.

Supports both:
    streamlit run app.py
    python app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

from src.ui.streamlit_app import run_app


def _is_running_under_streamlit() -> bool:
    """Return whether the current process is executing inside Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


def _launch_with_streamlit() -> None:
    """Re-exec the current file via the Streamlit CLI."""
    script_path = str(Path(__file__).resolve())
    os.execvp(
        sys.executable,
        [sys.executable, "-m", "streamlit", "run", script_path],
    )


def main(
    *,
    run_app_fn: Callable[[], None] = run_app,
    is_streamlit_context_fn: Callable[[], bool] = _is_running_under_streamlit,
    launch_streamlit_fn: Callable[[], None] = _launch_with_streamlit,
) -> None:
    """Launch the Julius Streamlit app through the correct runtime."""
    if is_streamlit_context_fn():
        run_app_fn()
        return
    launch_streamlit_fn()


if __name__ == "__main__":
    main()
