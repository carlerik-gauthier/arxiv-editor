"""Compatibility checks for the documented minimum supported Python version."""

from __future__ import annotations

import ast
from pathlib import Path


RUNTIME_FILES = [
    "app.py",
    "src/ui/streamlit_app.py",
    "src/agents/julius_session.py",
    "src/agents/tools/formatting_tool.py",
    "src/generation/interactive_workflow.py",
    "src/processing/embedder.py",
]


def test_runtime_files_parse_with_python_38_grammar():
    """Modules imported by the Streamlit app should match the documented Python floor."""
    repo_root = Path(__file__).resolve().parent.parent

    for relative_path in RUNTIME_FILES:
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        ast.parse(source, filename=relative_path, feature_version=(3, 8))
