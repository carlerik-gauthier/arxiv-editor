"""Shared pytest configuration."""

import os


def pytest_configure():
    """Keep unit tests from exporting OpenAI traces over the network."""
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
