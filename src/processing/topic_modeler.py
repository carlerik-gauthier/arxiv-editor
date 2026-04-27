"""BERTopic-backed topic discovery for ArXiv paper metadata."""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.processing.embedder import DEFAULT_EMBEDDING_MODEL, TextEmbedder


DEFAULT_REPRESENTATIVE_PAPERS = 5
DEFAULT_TOPIC_REPRESENTATION_MODEL = "gpt-4o-mini"
DEFAULT_MMR_DIVERSITY = 0.3
DEFAULT_MMR_TOP_N_WORDS = 10
DEFAULT_TOPIC_REPRESENTATION_PROMPT = """
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by these keywords: [KEYWORDS]

Create a concise research-topic label of at most six words. Return only the label.
""".strip()


@dataclass
class PaperTopicRecord:
    """Normalized paper metadata used during topic modeling."""

    index: int
    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    categories: List[str]
    published: Optional[str]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalized paper metadata."""
        return {
            "index": self.index,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "summary": self.summary,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published,
            "text": self.text,
        }


class TopicModeler:
    """
    Discover topics from paper titles and abstracts using BERTopic.

    Args:
        embedder: TextEmbedder used to produce custom embeddings for BERTopic.
        topic_model: Optional injected BERTopic-compatible object for tests.
        title_generator: Optional callable used to turn keywords into titles.
        model_name: Embedding model name when an embedder is not injected.
        representation_model_name: OpenAI model used for BERTopic topic labels.
        openai_api_key: API key used to create an OpenAI client when needed.
        openai_client: Optional injected OpenAI-compatible client for tests.
        use_openai_representation: Whether BERTopic should label topics with OpenAI.
        use_mmr_representation: Whether BERTopic should diversify topic keywords with MMR.
        mmr_diversity: Diversity weight for MaximalMarginalRelevance.
        mmr_top_n_words: Number of words retained by MaximalMarginalRelevance.
    """

    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        topic_model: Optional[Any] = None,
        title_generator: Optional[Callable[[List[str], List[Dict[str, Any]]], str]] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        representation_model_name: str = DEFAULT_TOPIC_REPRESENTATION_MODEL,
        openai_api_key: Optional[str] = None,
        openai_client: Optional[Any] = None,
        use_openai_representation: bool = True,
        use_mmr_representation: bool = True,
        mmr_diversity: float = DEFAULT_MMR_DIVERSITY,
        mmr_top_n_words: int = DEFAULT_MMR_TOP_N_WORDS,
    ) -> None:
        self.embedder = embedder or TextEmbedder(model_name=model_name)
        self._topic_model = topic_model
        self.title_generator = title_generator or generate_topic_title
        self.representation_model_name = representation_model_name
        self.openai_api_key = openai_api_key
        self.openai_client = openai_client
        self.use_openai_representation = use_openai_representation
        self.use_mmr_representation = use_mmr_representation
        self.mmr_diversity = mmr_diversity
        self.mmr_top_n_words = mmr_top_n_words
        self._last_topics: List[int] = []
        self._last_probabilities: Any = None
        self._last_records: List[PaperTopicRecord] = []
        self._last_topic_results: Dict[int, Dict[str, Any]] = {}

    def extract_topics(
        self,
        papers: Iterable[Any],
        min_topic_size: int = 5,
        num_topics: Optional[int] = None,
        representative_papers_per_topic: int = DEFAULT_REPRESENTATIVE_PAPERS,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Run topic modeling and return structured topic information.

        Papers may be dictionaries, dataclasses with `to_dict`, or objects with
        metadata attributes. Each paper is represented by title plus summary, then
        embedded with TextEmbedder before BERTopic receives the custom vectors.
        """
        if min_topic_size < 2:
            raise ValueError("min_topic_size must be at least 2")
        if representative_papers_per_topic < 1:
            raise ValueError("representative_papers_per_topic must be at least 1")

        records = normalize_papers(papers)
        documents = [record.text for record in records]
        progress = [
            {"stage": "normalize_papers", "status": "completed", "paper_count": len(records)}
        ]

        embedding_result = self.embedder.embed_texts(documents, batch_size=batch_size)
        embeddings = embedding_result["embeddings"]
        progress.append(
            {
                "stage": "embed_texts",
                "status": "completed",
                "embedding_count": embedding_result["embedding_count"],
                "cache_hits": embedding_result["cache_hits"],
                "cache_misses": embedding_result["cache_misses"],
            }
        )

        model = self._get_topic_model(min_topic_size=min_topic_size, num_topics=num_topics)
        topics, probabilities = model.fit_transform(documents, embeddings=embeddings)
        topic_ids = [int(topic_id) for topic_id in topics]
        progress.append(
            {
                "stage": "fit_bertopic",
                "status": "completed",
                "assigned_papers": len(topic_ids),
            }
        )

        self._last_topics = topic_ids
        self._last_probabilities = probabilities
        self._last_records = records

        grouped_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for index, topic_id in enumerate(topic_ids):
            probability = _probability_for_assignment(probabilities, index)
            grouped_records[topic_id].append(
                {
                    **records[index].to_dict(),
                    "topic_id": topic_id,
                    "topic_probability": probability,
                }
            )

        topic_results: List[Dict[str, Any]] = []
        for topic_id in sorted(topic_id for topic_id in grouped_records if topic_id != -1):
            keywords = self._extract_keywords(topic_id, grouped_records[topic_id])
            representative_papers = _select_representative_papers(
                grouped_records[topic_id],
                limit=representative_papers_per_topic,
            )
            title = self.title_generator(keywords, representative_papers)
            topic_result = {
                "topic_id": topic_id,
                "title": title,
                "keywords": keywords,
                "paper_count": len(grouped_records[topic_id]),
                "representative_papers": representative_papers,
            }
            topic_results.append(topic_result)
            self._last_topic_results[topic_id] = topic_result

        outlier_count = len(grouped_records.get(-1, []))
        return {
            "topics": topic_results,
            "topic_count": len(topic_results),
            "paper_count": len(records),
            "outlier_count": outlier_count,
            "model_name": getattr(self.embedder, "model_name", DEFAULT_EMBEDDING_MODEL),
            "representation_model": (
                self.representation_model_name if self.use_openai_representation else None
            ),
            "representation_models": self._representation_model_names(),
            "embedding": {
                "dimension": embedding_result["dimension"],
                "cache_hits": embedding_result["cache_hits"],
                "cache_misses": embedding_result["cache_misses"],
                "cache_path": embedding_result["cache_path"],
            },
            "progress": progress,
        }

    def get_topic_info(self, topic_id: int) -> Dict[str, Any]:
        """
        Return keywords, paper count, and representatives for a fitted topic.

        `extract_topics` must be called first. The method reads the last modeled
        result and refreshes keywords from BERTopic when that API is available.
        """
        if topic_id not in self._last_topic_results:
            raise ValueError(f"Topic {topic_id} is not available. Run extract_topics first.")

        topic_info = dict(self._last_topic_results[topic_id])
        topic_info["keywords"] = self._extract_keywords(
            topic_id,
            topic_info.get("representative_papers", []),
        )
        return topic_info

    def _get_topic_model(self, min_topic_size: int, num_topics: Optional[int]) -> Any:
        """Return an injected or lazily initialized BERTopic model."""
        if self._topic_model is not None:
            return self._topic_model

        try:
            from bertopic import BERTopic
            from bertopic.representation import MaximalMarginalRelevance
            from bertopic.representation import OpenAI as OpenAIRepresentation
            from bertopic.vectorizers import ClassTfidfTransformer
            from hdbscan import HDBSCAN
            from umap import UMAP
        except ImportError as exc:
            raise ImportError(
                "BERTopic, UMAP, HDBSCAN, and BERTopic OpenAI representation support "
                "are required for topic discovery. "
                "Install project dependencies or inject a BERTopic-compatible topic_model."
            ) from exc

        representation_model = None
        representation_models = self._build_representation_models(
            maximal_marginal_relevance_class=MaximalMarginalRelevance,
            openai_representation_class=OpenAIRepresentation,
        )
        if representation_models:
            representation_model = (
                representation_models[0]
                if len(representation_models) == 1
                else representation_models
            )

        umap_model = UMAP(
            n_neighbors=max(2, min(15, min_topic_size * 2)),
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        ctfidf_model = ClassTfidfTransformer()
        self._topic_model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            nr_topics=num_topics,
            calculate_probabilities=True,
            verbose=False,
        )
        return self._topic_model

    def _build_representation_models(
        self,
        maximal_marginal_relevance_class: Any,
        openai_representation_class: Any,
    ) -> List[Any]:
        """
        Build BERTopic representation models in execution order.

        MMR runs first to diversify keyword candidates. OpenAI then receives the
        diversified representation and generates the final short label with
        `gpt-4o-mini`.
        """
        representation_models: List[Any] = []
        if self.use_mmr_representation:
            representation_models.append(
                maximal_marginal_relevance_class(
                    diversity=self.mmr_diversity,
                    top_n_words=self.mmr_top_n_words,
                )
            )
        if self.use_openai_representation:
            representation_models.append(
                self._build_openai_representation_model(openai_representation_class)
            )
        return representation_models

    def _build_openai_representation_model(self, openai_representation_class: Any) -> Any:
        """
        Build BERTopic's OpenAI representation model with gpt-4o-mini.

        BERTopic 0.16 calls `client.chat.completions.create` when `chat=True`,
        so this method creates the modern OpenAI Python client unless a
        compatible client was injected.
        """
        client = self.openai_client or _create_openai_client(self.openai_api_key)
        return openai_representation_class(
            client,
            model=self.representation_model_name,
            chat=True,
            prompt=DEFAULT_TOPIC_REPRESENTATION_PROMPT,
            generator_kwargs={"temperature": 0},
        )

    def _representation_model_names(self) -> List[str]:
        """Return configured BERTopic representation model names for metadata."""
        names: List[str] = []
        if self.use_mmr_representation:
            names.append("MaximalMarginalRelevance")
        if self.use_openai_representation:
            names.append(self.representation_model_name)
        return names

    def _extract_keywords(
        self,
        topic_id: int,
        papers: Sequence[Dict[str, Any]],
        limit: int = 8,
    ) -> List[str]:
        """Extract topic keywords from BERTopic or fall back to paper text."""
        if self._topic_model is not None and hasattr(self._topic_model, "get_topic"):
            topic_words = self._topic_model.get_topic(topic_id)
            if topic_words:
                return [str(word) for word, _score in topic_words[:limit]]
        return _keywords_from_papers(papers, limit=limit)


def normalize_papers(papers: Iterable[Any]) -> List[PaperTopicRecord]:
    """
    Normalize paper-like objects into title, summary, metadata, and modeling text.

    Raises:
        ValueError: If no usable papers are supplied.
    """
    normalized: List[PaperTopicRecord] = []
    for index, paper in enumerate(papers):
        paper_dict = _paper_to_dict(paper)
        title = str(paper_dict.get("title") or "Untitled paper").strip()
        summary = str(
            paper_dict.get("summary")
            or paper_dict.get("abstract")
            or paper_dict.get("description")
            or ""
        ).strip()
        text = " ".join(part for part in [title, summary] if part).strip()
        if not text:
            continue

        normalized.append(
            PaperTopicRecord(
                index=index,
                arxiv_id=str(
                    paper_dict.get("arxiv_id")
                    or paper_dict.get("id")
                    or paper_dict.get("entry_id")
                    or index
                ),
                title=title,
                summary=summary,
                authors=[str(author) for author in paper_dict.get("authors", [])],
                categories=[str(category) for category in paper_dict.get("categories", [])],
                published=str(paper_dict.get("published")) if paper_dict.get("published") else None,
                text=text,
            )
        )

    if not normalized:
        raise ValueError("papers must contain at least one paper with a title or summary")
    return normalized


def generate_topic_title(
    topic_keywords: Sequence[str],
    sample_papers: Sequence[Dict[str, Any]],
) -> str:
    """
    Generate a readable deterministic topic title from keywords and sample papers.

    This is the local fallback for the agent tool. The tool can use an injected
    LLM client for richer titles while preserving this output contract.
    """
    keywords = [keyword for keyword in topic_keywords if keyword]
    if keywords:
        return " / ".join(keyword.title() for keyword in keywords[:3])

    title_words: List[str] = []
    for paper in sample_papers:
        title_words.extend(
            _tokenize(f"{paper.get('title', '')} {paper.get('summary', '')}")
        )
    if title_words:
        common_words = [word for word, _count in Counter(title_words).most_common(3)]
        return " / ".join(word.title() for word in common_words)
    return "Emerging Research Theme"


def _create_openai_client(openai_api_key: Optional[str] = None) -> Any:
    """Create an OpenAI client for BERTopic topic representation."""
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or _settings_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for BERTopic OpenAI representation labels. "
            "Set OPENAI_API_KEY, pass openai_api_key, inject openai_client, or disable "
            "use_openai_representation."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for BERTopic OpenAI representation labels."
        ) from exc
    return OpenAI(api_key=api_key)


def _settings_openai_api_key() -> str:
    """Read OpenAI API key from Pydantic settings when available."""
    try:
        from config.settings import Settings
    except Exception:
        return ""

    try:
        settings = Settings()
    except Exception:
        return ""
    return settings.openai_api_key or settings.llm_api_key


def _paper_to_dict(paper: Any) -> Dict[str, Any]:
    """Convert common paper objects into dictionaries."""
    if isinstance(paper, dict):
        return dict(paper)
    if hasattr(paper, "to_dict") and callable(paper.to_dict):
        return dict(paper.to_dict())
    return {
        "arxiv_id": getattr(paper, "arxiv_id", None),
        "title": getattr(paper, "title", None),
        "summary": getattr(paper, "summary", None),
        "authors": getattr(paper, "authors", []),
        "categories": getattr(paper, "categories", []),
        "published": getattr(paper, "published", None),
    }


def _probability_for_assignment(probabilities: Any, index: int) -> Optional[float]:
    """Extract a representative confidence score from BERTopic probabilities."""
    if probabilities is None:
        return None
    try:
        row = probabilities[index]
    except (IndexError, TypeError):
        return None

    if isinstance(row, (int, float)):
        if math.isnan(float(row)):
            return None
        return float(row)
    if hasattr(row, "tolist"):
        row = row.tolist()
    try:
        numeric_values = [float(value) for value in row if value is not None]
    except TypeError:
        return None
    if not numeric_values:
        return None
    return max(numeric_values)


def _select_representative_papers(
    papers: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Select the highest-confidence papers for a topic."""
    ranked = sorted(
        papers,
        key=lambda paper: (
            paper.get("topic_probability") is not None,
            paper.get("topic_probability") or 0.0,
            paper.get("title", ""),
        ),
        reverse=True,
    )
    return [
        {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "summary": paper["summary"],
            "authors": paper["authors"],
            "categories": paper["categories"],
            "topic_probability": paper.get("topic_probability"),
        }
        for paper in ranked[:limit]
    ]


def _keywords_from_papers(papers: Sequence[Dict[str, Any]], limit: int) -> List[str]:
    """Fallback keyword extraction from representative paper titles and summaries."""
    tokens: List[str] = []
    for paper in papers:
        tokens.extend(_tokenize(f"{paper.get('title', '')} {paper.get('summary', '')}"))
    return [word for word, _count in Counter(tokens).most_common(limit)]


def _tokenize(text: str) -> List[str]:
    """Tokenize text into simple lowercase keywords with stopword removal."""
    stopwords = {
        "and",
        "are",
        "for",
        "from",
        "into",
        "that",
        "the",
        "this",
        "with",
        "using",
        "paper",
        "study",
        "studies",
        "result",
        "results",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())
        if token not in stopwords
    ]
