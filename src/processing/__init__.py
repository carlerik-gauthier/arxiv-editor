"""Processing utilities for embeddings, topic modeling, and analysis."""

from src.processing.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingCache,
    TextEmbedder,
)
from src.processing.topic_modeler import (
    PaperTopicRecord,
    TopicModeler,
    generate_topic_title,
    normalize_papers,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingCache",
    "PaperTopicRecord",
    "TextEmbedder",
    "TopicModeler",
    "generate_topic_title",
    "normalize_papers",
]
