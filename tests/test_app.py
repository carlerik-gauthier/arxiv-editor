"""Tests for the local Streamlit entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import app
from src.ui import streamlit_app


def test_main_runs_ui_when_already_inside_streamlit():
    """The entry point should render directly inside Streamlit."""
    events: list[str] = []

    app.main(
        run_app_fn=lambda: events.append("run_app"),
        is_streamlit_context_fn=lambda: True,
        launch_streamlit_fn=lambda: events.append("launch_streamlit"),
    )

    assert events == ["run_app"]


def test_main_reexecs_through_streamlit_when_run_directly():
    """Direct python execution should delegate to the Streamlit CLI."""
    events: list[str] = []

    app.main(
        run_app_fn=lambda: events.append("run_app"),
        is_streamlit_context_fn=lambda: False,
        launch_streamlit_fn=lambda: events.append("launch_streamlit"),
    )

    assert events == ["launch_streamlit"]


def test_launch_with_streamlit_uses_current_script(monkeypatch):
    """The delegated CLI launch should target this repository's app.py file."""
    captured: dict[str, object] = {}

    def fake_execvp(executable, args):
        captured["executable"] = executable
        captured["args"] = args
        raise SystemExit(0)

    monkeypatch.setattr(app.os, "execvp", fake_execvp)

    try:
        app._launch_with_streamlit()
    except SystemExit:
        pass

    assert captured["executable"] == app.sys.executable
    assert captured["args"] == [
        app.sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(app.__file__).resolve()),
    ]


def test_run_app_requires_streamlit_context(monkeypatch):
    """The UI adapter should fail fast instead of using Streamlit bare mode."""
    monkeypatch.setattr(streamlit_app, "_has_streamlit_context", lambda: False)

    class _FakeStreamlitModule:
        session_state = {}

        def set_page_config(self, **_kwargs):
            raise AssertionError("set_page_config should not run without Streamlit context")

    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlitModule())

    try:
        streamlit_app.run_app()
    except RuntimeError as exc:
        assert "python app.py" in str(exc)
    else:
        raise AssertionError("run_app should require a Streamlit context")
