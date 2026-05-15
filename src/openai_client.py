"""Shared OpenAI client helpers for tool-backed LLM calls."""

from __future__ import annotations

import os
from typing import Any, Optional


def is_openai_client_like(client: Any) -> bool:
    """Return whether a client exposes common OpenAI SDK request surfaces."""
    responses_api = getattr(client, "responses", None)
    if callable(getattr(responses_api, "create", None)):
        return True

    chat_api = getattr(client, "chat", None)
    completions_api = getattr(chat_api, "completions", None)
    return callable(getattr(completions_api, "create", None))


def resolve_openai_client(
    llm_client: Optional[Any] = None,
    *,
    api_key: Optional[str] = None,
    required: bool = False,
    purpose: str = "llm_client",
) -> Optional[Any]:
    """Return an OpenAI client from an explicit client or configured API key."""
    if llm_client is not None and is_openai_client_like(llm_client):
        return llm_client

    client = create_openai_client(api_key=api_key)
    if client is not None:
        return client

    if llm_client is not None and required:
        raise TypeError(
            f"{purpose} must be an OpenAI client exposing responses.create or "
            "chat.completions.create"
        )
    return None


def create_openai_client(api_key: Optional[str] = None) -> Optional[Any]:
    """Create an OpenAI client when an API key is available."""
    resolved_api_key = (
        api_key
        or os.getenv("OPENAI_API_KEY")
        or _settings_openai_api_key()
        or os.getenv("LLM_API_KEY")
    )
    if not resolved_api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        return OpenAI(api_key=resolved_api_key)
    except Exception:
        return None


def default_openai_model() -> str:
    """Resolve the default OpenAI model for tool-backed prompts."""
    model = os.getenv("OPENAI_MODEL")
    if model:
        return model
    try:
        from config.settings import Settings
        settings = Settings()
        if settings.llm_model:
            return settings.llm_model
    except Exception:
        pass
    return "gpt-4o-mini"


def _settings_openai_api_key() -> str:
    """Read OpenAI API key from settings when available."""
    try:
        from config.settings import Settings
    except Exception:
        return ""

    try:
        settings = Settings()
    except Exception:
        return ""
    return settings.openai_api_key or settings.llm_api_key
