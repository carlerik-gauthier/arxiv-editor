"""Agent-facing text embedding tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.agents.base_agent import AgentTool
from src.processing.embedder import DEFAULT_EMBEDDING_MODEL, TextEmbedder


def embed_text_tool(
    texts: Iterable[str],
    batch_size: int = 32,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_path: Optional[str] = None,
    embedder: Optional[TextEmbedder] = None,
) -> Dict[str, Any]:
    """
    Convert text inputs into dense vector embeddings for downstream analysis.

    Args:
        texts: Abstracts, summaries, or paper text snippets to embed.
        batch_size: Number of uncached texts to encode per model call.
        model_name: sentence-transformers model identifier.
        cache_path: Optional pickle cache path. Defaults to `data/cache/embeddings.pkl`.
        embedder: Optional injected TextEmbedder for tests or alternate providers.

    Returns:
        JSON-serializable embeddings plus model and cache metadata.
    """
    active_embedder = embedder or TextEmbedder(
        model_name=model_name,
        cache_path=Path(cache_path) if cache_path else "data/cache/embeddings.pkl",
    )
    return active_embedder.embed_texts(texts, batch_size=batch_size)


def get_embedding_tool() -> AgentTool:
    """Return the default text embedding tool registered with research agents."""
    return AgentTool(
        name="embed_text_tool",
        description=(
            "Embed paper abstracts, summaries, or text snippets with "
            "sentence-transformers all-MiniLM-L6-v2 and a file-backed cache. "
            "Use this before topic modeling, clustering, or representative paper "
            "selection."
        ),
        function=embed_text_tool,
        required_parameters=["texts"],
    )
