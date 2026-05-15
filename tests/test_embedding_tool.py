"""Tests for the phase-4.1 embedding tool and cache."""

import logging
import pickle
import sys
import types

import pytest

from src.agents import ChrisAgent
from src.agents.tools import embed_text_tool
from src.processing import DEFAULT_EMBEDDING_MODEL, EmbeddingCache, TextEmbedder
from src.processing.embedder import ALLOW_MODEL_DOWNLOAD_ENV
from src.processing.hf_logging import configure_third_party_logging


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


def test_text_embedder_uses_local_files_only_by_default(monkeypatch, tmp_path):
    """Runtime model loading should avoid background Hugging Face downloads by default."""
    captured = {}

    class FakeSentenceTransformerLoader:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformerLoader),
    )
    monkeypatch.delenv(ALLOW_MODEL_DOWNLOAD_ENV, raising=False)

    embedder = TextEmbedder(model_name="cached-model", cache_path=tmp_path / "cache.pkl")

    _ = embedder.model

    assert captured == {
        "model_name": "cached-model",
        "kwargs": {"local_files_only": True},
    }


def test_text_embedder_raises_clear_error_when_local_model_is_missing(monkeypatch, tmp_path):
    """Missing cached models should fail fast without retrying noisy runtime downloads."""
    class FakeSentenceTransformerLoader:
        def __init__(self, model_name, **kwargs):
            raise OSError(f"{model_name} not cached")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformerLoader),
    )
    monkeypatch.delenv(ALLOW_MODEL_DOWNLOAD_ENV, raising=False)

    embedder = TextEmbedder(model_name="missing-model", cache_path=tmp_path / "cache.pkl")

    with pytest.raises(RuntimeError, match="local Hugging Face cache"):
        _ = embedder.model


def test_text_embedder_can_opt_in_to_runtime_model_downloads(monkeypatch, tmp_path):
    """An explicit env var should preserve opt-in download behavior for first-time setup."""
    captured = {}

    class FakeSentenceTransformerLoader:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformerLoader),
    )
    monkeypatch.setenv(ALLOW_MODEL_DOWNLOAD_ENV, "1")

    embedder = TextEmbedder(model_name="downloadable-model", cache_path=tmp_path / "cache.pkl")

    _ = embedder.model

    assert captured == {
        "model_name": "downloadable-model",
        "kwargs": {},
    }


def test_configure_third_party_logging_filters_known_transformers_alias_warning():
    """Only the noisy transformers alias warning should be filtered."""
    logger = logging.getLogger("transformers")
    alias_logger = logging.getLogger("transformers.__init__")
    existing_filters = list(logger.filters)
    existing_alias_filters = list(alias_logger.filters)
    logger.filters[:] = []
    alias_logger.filters[:] = []
    try:
        configure_third_party_logging()
        allowed = logging.LogRecord(
            name="transformers",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="Different warning",
            args=(),
            exc_info=None,
        )
        blocked = logging.LogRecord(
            name="transformers",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=(
                "Accessing `__path__` from `.models.zoedepth.image_processing_pil_zoedepth`. "
                "Returning `__path__` instead. Behavior may be different and this alias "
                "will be removed in future versions."
            ),
            args=(),
            exc_info=None,
        )

        assert all(active_filter.filter(allowed) for active_filter in logger.filters)
        assert any(active_filter.filter(blocked) is False for active_filter in logger.filters)
    finally:
        logger.filters[:] = existing_filters
        alias_logger.filters[:] = existing_alias_filters
