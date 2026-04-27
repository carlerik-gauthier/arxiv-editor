"""Tests for the phase-4.1 embedding tool and cache."""

import pickle

import pytest

from src.agents import ChrisAgent
from src.agents.tools import embed_text_tool
from src.processing import DEFAULT_EMBEDDING_MODEL, EmbeddingCache, TextEmbedder


class FakeSentenceTransformer:
    """Small deterministic encoder with a sentence-transformers compatible API."""

    def __init__(self):
        self.calls = []

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=False,
        show_progress_bar=False,
    ):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [
            [float(len(text)), float(sum(ord(character) for character in text) % 997)]
            for text in texts
        ]


def test_embedding_cache_get_or_create_embedding_reuses_cached_vector(tmp_path):
    """EmbeddingCache stores vectors by text hash and model version."""
    cache = EmbeddingCache(tmp_path / "embeddings.pkl")
    calls = {"count": 0}

    def create_embedding(text):
        calls["count"] += 1
        return [1.0, float(len(text))]

    first = cache.get_or_create_embedding(
        "A paper summary",
        DEFAULT_EMBEDDING_MODEL,
        create_embedding,
    )
    second = cache.get_or_create_embedding(
        "A paper summary",
        DEFAULT_EMBEDDING_MODEL,
        create_embedding,
    )

    assert first == [1.0, 15.0]
    assert second == first
    assert calls["count"] == 1
    assert cache.hits == 1
    assert cache.misses == 1


def test_text_embedder_batches_uncached_texts_and_reuses_cache(tmp_path):
    """TextEmbedder batches model calls and avoids re-encoding cached texts."""
    fake_model = FakeSentenceTransformer()
    embedder = TextEmbedder(
        cache_path=tmp_path / "embeddings.pkl",
        model=fake_model,
    )
    texts = [f"abstract {index}" for index in range(50)]

    first = embedder.embed_texts(texts, batch_size=16)
    second = embedder.embed_texts(texts, batch_size=16)

    assert first["embedding_count"] == 50
    assert first["dimension"] == 2
    assert first["cache_hits"] == 0
    assert first["cache_misses"] == 50
    assert second["cache_hits"] == 50
    assert second["cache_misses"] == 0
    assert len(fake_model.calls) == 4
    assert [len(call["texts"]) for call in fake_model.calls] == [16, 16, 16, 2]


def test_embed_text_tool_returns_json_serializable_metadata(tmp_path):
    """Agent tool wrapper returns embeddings plus model/cache metadata."""
    fake_model = FakeSentenceTransformer()
    embedder = TextEmbedder(
        model_name="fake-model",
        cache_path=tmp_path / "tool-cache.pkl",
        model=fake_model,
    )

    result = embed_text_tool(
        texts=["first abstract", "second abstract"],
        batch_size=2,
        embedder=embedder,
    )

    assert result["model_name"] == "fake-model"
    assert result["embedding_count"] == 2
    assert result["dimension"] == 2
    assert result["cache_misses"] == 2
    assert isinstance(result["embeddings"][0][0], float)


def test_specialized_agents_have_embedding_tool():
    """The embedding tool is registered through the default specialist tools."""
    agent = ChrisAgent()

    assert "embed_text_tool" in agent.list_tools()


def test_embed_text_tool_rejects_invalid_text_inputs(tmp_path):
    """Input validation catches empty and incorrectly shaped text arguments."""
    embedder = TextEmbedder(
        cache_path=tmp_path / "invalid-cache.pkl",
        model=FakeSentenceTransformer(),
    )

    with pytest.raises(TypeError, match="iterable of strings"):
        embed_text_tool("not a list", embedder=embedder)

    with pytest.raises(ValueError, match="empty strings"):
        embed_text_tool(["valid", "   "], embedder=embedder)


def test_embedding_cache_persists_to_disk(tmp_path):
    """Cache files can be loaded by a new EmbeddingCache instance."""
    cache_path = tmp_path / "persistent-cache.pkl"
    cache = EmbeddingCache(cache_path)
    cache.set("summary", "model-a", [0.25, 0.75])
    cache.save()

    reloaded_cache = EmbeddingCache(cache_path)

    assert reloaded_cache.get("summary", "model-a") == [0.25, 0.75]
    with cache_path.open("rb") as cache_file:
        assert isinstance(pickle.load(cache_file), dict)
