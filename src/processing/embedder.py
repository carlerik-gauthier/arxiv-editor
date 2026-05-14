"""Text embedding implementation with file-backed caching."""

from __future__ import annotations

import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingCache:
    """
    File-backed embedding cache keyed by text content and model version.

    The cache stores a dictionary in a pickle file. It is intentionally simple
    for phase 4.1, but the public methods isolate cache behavior so a database
    or HDF5 backend can replace it later without changing the embedder API.
    """

    def __init__(self, cache_path: Path | str = "data/cache/embeddings.pkl") -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[float]] = self._load()
        self.hits = 0
        self.misses = 0

    def make_key(self, text: str, model_name: str) -> str:
        """Create a stable cache key from model version and text hash."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
        return f"{model_hash}:{text_hash}"

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Return a cached embedding, if present, and update hit/miss counters."""
        key = self.make_key(text, model_name)
        if key in self._cache:
            self.hits += 1
            return list(self._cache[key])
        self.misses += 1
        return None

    def set(self, text: str, model_name: str, embedding: Iterable[float]) -> None:
        """Store one embedding in memory under its text/model cache key."""
        self._cache[self.make_key(text, model_name)] = [float(value) for value in embedding]

    def save(self) -> None:
        """Persist the cache to disk."""
        with self.cache_path.open("wb") as cache_file:
            pickle.dump(self._cache, cache_file)

    def get_or_create_embedding(
        self,
        text: str,
        model_name: str,
        create_embedding: Any,
    ) -> List[float]:
        """
        Return a cached embedding or create, store, and persist a new one.

        Args:
            text: Input text to embed.
            model_name: Embedding model identifier used in the cache key.
            create_embedding: Callable that accepts text and returns one vector.
        """
        cached = self.get(text, model_name)
        if cached is not None:
            return cached

        embedding = [float(value) for value in create_embedding(text)]
        self.set(text, model_name, embedding)
        self.save()
        return embedding

    def _load(self) -> Dict[str, List[float]]:
        """Load cache data from disk, returning an empty cache if no file exists."""
        if not self.cache_path.exists():
            return {}
        with self.cache_path.open("rb") as cache_file:
            data = pickle.load(cache_file)
        if not isinstance(data, dict):
            raise ValueError(f"Embedding cache at {self.cache_path} is not a dictionary")
        return {str(key): [float(value) for value in values] for key, values in data.items()}


class TextEmbedder:
    """
    Batch text embedder backed by sentence-transformers and EmbeddingCache.

    The `model` argument is injectable for tests or alternate providers. When no
    model is injected, `sentence_transformers.SentenceTransformer` is imported
    lazily and initialized with `all-MiniLM-L6-v2` by default.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache: Optional[EmbeddingCache] = None,
        cache_path: Path | str = "data/cache/embeddings.pkl",
        model: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.cache = cache or EmbeddingCache(cache_path)
        self._model = model

    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> Dict[str, Any]:
        """
        Embed texts with transparent cache reuse and batch model calls.

        Returns a structured dictionary containing JSON-serializable embeddings,
        cache statistics, and embedding dimensions for agent tool callers.
        """
        normalized_texts = self._normalize_texts(texts)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        embeddings: List[Optional[List[float]]] = [None] * len(normalized_texts)
        uncached_items: List[tuple[int, str]] = []
        starting_hits = self.cache.hits
        starting_misses = self.cache.misses

        for index, text in enumerate(normalized_texts):
            cached_embedding = self.cache.get(text, self.model_name)
            if cached_embedding is None:
                uncached_items.append((index, text))
            else:
                embeddings[index] = cached_embedding

        for batch in self._iter_batches(uncached_items, batch_size):
            batch_indices = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]
            batch_embeddings = self._encode_batch(batch_texts, batch_size=batch_size)
            for index, text, embedding in zip(batch_indices, batch_texts, batch_embeddings):
                embedding_list = [float(value) for value in embedding]
                embeddings[index] = embedding_list
                self.cache.set(text, self.model_name, embedding_list)

        if uncached_items:
            self.cache.save()

        final_embeddings = [embedding for embedding in embeddings if embedding is not None]
        dimension = len(final_embeddings[0]) if final_embeddings else 0
        return {
            "embeddings": final_embeddings,
            "embedding_count": len(final_embeddings),
            "dimension": dimension,
            "model_name": self.model_name,
            "cache_hits": self.cache.hits - starting_hits,
            "cache_misses": self.cache.misses - starting_misses,
            "cache_path": str(self.cache.cache_path),
        }

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text using the same cache path as batch embedding."""
        return self.cache.get_or_create_embedding(
            text=self._normalize_text(text),
            model_name=self.model_name,
            create_embedding=lambda value: self._encode_batch([value], batch_size=1)[0],
        )

    @property
    def model(self) -> Any:
        """Load and return the underlying sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for real embedding generation. "
                    "Install project dependencies or inject a compatible model."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode_batch(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Encode a batch with common sentence-transformers compatible APIs."""
        encoded = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [_to_float_list(embedding) for embedding in encoded]

    @staticmethod
    def _normalize_texts(texts: Iterable[str]) -> List[str]:
        """Validate and normalize a collection of input texts."""
        if isinstance(texts, str):
            raise TypeError("texts must be an iterable of strings, not a single string")

        normalized = [TextEmbedder._normalize_text(text) for text in texts]
        if not normalized:
            raise ValueError("texts cannot be empty")
        return normalized

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Validate and normalize one input text."""
        if not isinstance(text, str):
            raise TypeError("each text must be a string")
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("texts cannot contain empty strings")
        return normalized

    @staticmethod
    def _iter_batches(items: List[tuple[int, str]], batch_size: int) -> Iterable[List[tuple[int, str]]]:
        """Yield fixed-size batches from indexed text items."""
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]


def _to_float_list(embedding: Any) -> List[float]:
    """Convert numpy arrays, tensors, or plain iterables to a float list."""
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return np.array([float(value) for value in embedding])
