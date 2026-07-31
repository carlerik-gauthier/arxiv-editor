"""Shared pytest configuration."""

import os

# Set this during conftest import, before test-module collection imports agents.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


def pytest_configure():
    """Disable OpenAI tracing before the pytest test session starts.

    Returns:
        None: Sets the tracing environment default for the current process.
    """
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
