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
        self._last_grouped_records: Dict[int, List[Dict[str, Any]]] = {}

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
        self._last_grouped_records = {
            topic_id: list(records) for topic_id, records in grouped_records.items()
        }

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

    def select_representative_papers(
        self,
        topic_id: int,
        papers: Iterable[Any],
        n: int = DEFAULT_REPRESENTATIVE_PAPERS,
        diversity_threshold: float = 0.7,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Select representative papers for a topic using confidence, centrality, and diversity.

        The method prefers papers assigned to `topic_id` by the latest BERTopic
        run, but it also accepts already-filtered paper dictionaries with
        `topic_id`/`topic_probability` fields. Papers are first scored by topic
        probability plus distance to the topic centroid, then greedily selected
        while avoiding near-duplicates above `diversity_threshold`.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        if not 0.0 <= diversity_threshold <= 1.0:
            raise ValueError("diversity_threshold must be between 0 and 1")

        candidates = self._selection_candidates(topic_id, papers)
        if not candidates:
            raise ValueError(f"No papers are available for topic {topic_id}")

        embedding_result = self.embedder.embed_texts(
            [candidate["text"] for candidate in candidates],
            batch_size=batch_size,
        )
        embeddings = embedding_result["embeddings"]
        centroid = _centroid(embeddings)

        scored_candidates: List[Dict[str, Any]] = []
        for candidate, embedding in zip(candidates, embeddings):
            topic_probability = candidate.get("topic_probability")
            probability_score = _score_probability(topic_probability)
            centrality_score = _bounded_cosine_similarity(embedding, centroid)
            if topic_probability is None:
                representativeness_score = centrality_score
            else:
                representativeness_score = (0.65 * probability_score) + (
                    0.35 * centrality_score
                )
            scored_candidates.append(
                {
                    **candidate,
                    "embedding": embedding,
                    "topic_probability": topic_probability,
                    "probability_score": probability_score,
                    "centrality_score": centrality_score,
                    "representativeness_score": representativeness_score,
                }
            )

        ranked_candidates = sorted(
            scored_candidates,
            key=lambda paper: (
                paper["representativeness_score"],
                paper.get("title", ""),
            ),
            reverse=True,
        )
        selected = _diverse_paper_selection(
            ranked_candidates,
            limit=min(n, len(ranked_candidates)),
            diversity_threshold=diversity_threshold,
        )

        selected_papers = []
        for rank, paper in enumerate(selected, start=1):
            selected_papers.append(_selection_result_paper(paper, rank))

        return {
            "topic_id": topic_id,
            "selected_papers": selected_papers,
            "paper_count": len(selected_papers),
            "candidate_count": len(candidates),
            "requested_count": n,
            "diversity_threshold": diversity_threshold,
            "embedding": {
                "dimension": embedding_result["dimension"],
                "cache_hits": embedding_result["cache_hits"],
                "cache_misses": embedding_result["cache_misses"],
                "cache_path": embedding_result["cache_path"],
            },
        }

    def rank_papers_by_relevance(
        self,
        papers: Iterable[Any],
        query: str,
        n: Optional[int] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Rank papers by semantic similarity between a query and title/abstract text."""
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if n is not None and n < 1:
            raise ValueError("n must be at least 1 when provided")

        candidates = _paper_candidates(papers)
        if not candidates:
            raise ValueError("papers must contain at least one paper with a title or summary")

        texts = [query, *[candidate["text"] for candidate in candidates]]
        embedding_result = self.embedder.embed_texts(texts, batch_size=batch_size)
        embeddings = embedding_result["embeddings"]
        query_embedding = embeddings[0]
        paper_embeddings = embeddings[1:]

        ranked = []
        for candidate, embedding in zip(candidates, paper_embeddings):
            ranked.append(
                {
                    **candidate,
                    "relevance_score": _bounded_cosine_similarity(
                        query_embedding,
                        embedding,
                    ),
                }
            )
        ranked.sort(
            key=lambda paper: (paper["relevance_score"], paper.get("title", "")),
            reverse=True,
        )

        limit = n or len(ranked)
        ranked_papers = [
            _relevance_result_paper(paper, rank)
            for rank, paper in enumerate(ranked[:limit], start=1)
        ]
        return {
            "query": " ".join(query.split()),
            "ranked_papers": ranked_papers,
            "paper_count": len(ranked_papers),
            "candidate_count": len(candidates),
            "embedding": {
                "dimension": embedding_result["dimension"],
                "cache_hits": embedding_result["cache_hits"],
                "cache_misses": embedding_result["cache_misses"],
                "cache_path": embedding_result["cache_path"],
            },
        }

    def _selection_candidates(
        self,
        topic_id: int,
        papers: Iterable[Any],
    ) -> List[Dict[str, Any]]:
        """Normalize and filter paper candidates for representative selection."""
        provided_candidates = _paper_candidates(papers)
        if not provided_candidates:
            return []

        topic_filtered = [
            candidate
            for candidate in provided_candidates
            if candidate.get("topic_id") is not None
            and int(candidate["topic_id"]) == int(topic_id)
        ]
        if topic_filtered:
            return topic_filtered

        modeled_topic_records = self._last_grouped_records.get(int(topic_id), [])
        if not modeled_topic_records:
            return provided_candidates

        modeled_by_id = {
            str(record.get("arxiv_id")): record for record in modeled_topic_records
        }
        modeled_by_index = {
            int(record["index"]): record
            for record in modeled_topic_records
            if record.get("index") is not None
        }

        matched_candidates: List[Dict[str, Any]] = []
        for candidate in provided_candidates:
            modeled_record = modeled_by_id.get(str(candidate.get("arxiv_id")))
            if modeled_record is None and candidate.get("index") is not None:
                modeled_record = modeled_by_index.get(int(candidate["index"]))
            if modeled_record is None:
                continue
            matched_candidates.append(
                {
                    **candidate,
                    "topic_id": topic_id,
                    "topic_probability": modeled_record.get("topic_probability"),
                }
            )
        return matched_candidates or provided_candidates

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


def _paper_candidates(papers: Iterable[Any]) -> List[Dict[str, Any]]:
    """Normalize paper-like inputs while preserving topic selection metadata."""
    candidates: List[Dict[str, Any]] = []
    for fallback_index, paper in enumerate(papers):
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

        candidates.append(
            {
                "index": paper_dict.get("index", fallback_index),
                "arxiv_id": str(
                    paper_dict.get("arxiv_id")
                    or paper_dict.get("id")
                    or paper_dict.get("entry_id")
                    or fallback_index
                ),
                "title": title,
                "summary": summary,
                "authors": [str(author) for author in paper_dict.get("authors", [])],
                "categories": [
                    str(category) for category in paper_dict.get("categories", [])
                ],
                "published": (
                    str(paper_dict.get("published")) if paper_dict.get("published") else None
                ),
                "topic_id": paper_dict.get("topic_id"),
                "topic_probability": paper_dict.get("topic_probability"),
                "text": text,
            }
        )
    return candidates


def _centroid(embeddings: Sequence[Sequence[float]]) -> List[float]:
    """Return the arithmetic centroid for a non-empty embedding matrix."""
    if not embeddings:
        return []
    dimension = len(embeddings[0])
    if dimension == 0:
        return []
    totals = [0.0] * dimension
    for embedding in embeddings:
        for index, value in enumerate(embedding[:dimension]):
            totals[index] += float(value)
    return [total / len(embeddings) for total in totals]


def _score_probability(probability: Any) -> float:
    """Normalize topic probability-like values to a bounded score."""
    if probability is None:
        return 0.0
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value):
        return 0.0
    return max(0.0, min(value, 1.0))


def _bounded_cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return cosine similarity clipped to the [0, 1] scoring range."""
    cosine = _cosine_similarity(left, right)
    return max(0.0, min(cosine, 1.0))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity with zero-vector protection."""
    dimension = min(len(left), len(right))
    if dimension == 0:
        return 0.0

    dot_product = sum(float(left[index]) * float(right[index]) for index in range(dimension))
    left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(dimension)))
    right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(dimension)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _diverse_paper_selection(
    ranked_candidates: Sequence[Dict[str, Any]],
    limit: int,
    diversity_threshold: float,
) -> List[Dict[str, Any]]:
    """Greedily choose high-scoring papers while suppressing near-duplicates."""
    selected: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []

    for candidate in ranked_candidates:
        max_similarity = _max_similarity_to_selected(candidate, selected)
        diversity_score = 1.0 - max_similarity
        candidate["max_similarity_to_selected"] = max_similarity
        candidate["diversity_score"] = diversity_score
        candidate["combined_score"] = _combined_selection_score(candidate)

        if max_similarity <= diversity_threshold or not selected:
            selected.append(candidate)
            if len(selected) == limit:
                return selected
        else:
            deferred.append(candidate)

    for candidate in deferred:
        if len(selected) == limit:
            break
        candidate["combined_score"] = _combined_selection_score(candidate)
        selected.append(candidate)

    selected.sort(key=lambda paper: paper["combined_score"], reverse=True)
    return selected[:limit]


def _max_similarity_to_selected(
    candidate: Dict[str, Any],
    selected: Sequence[Dict[str, Any]],
) -> float:
    """Return the highest bounded cosine similarity to already selected papers."""
    if not selected:
        return 0.0
    similarities = [
        _bounded_cosine_similarity(candidate["embedding"], paper["embedding"])
        for paper in selected
    ]
    return max(similarities)


def _combined_selection_score(paper: Dict[str, Any]) -> float:
    """Blend representativeness and diversity into one sortable score."""
    return (0.8 * paper["representativeness_score"]) + (
        0.2 * paper.get("diversity_score", 1.0)
    )


def _selection_result_paper(paper: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Convert an internal selection candidate into tool-facing metadata."""
    return {
        "rank": rank,
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "summary": paper["summary"],
        "authors": paper["authors"],
        "categories": paper["categories"],
        "published": paper.get("published"),
        "topic_id": paper.get("topic_id"),
        "topic_probability": paper.get("topic_probability"),
        "scores": {
            "topic_probability": paper["probability_score"],
            "centrality": paper["centrality_score"],
            "diversity": paper.get("diversity_score", 1.0),
            "representativeness": paper["representativeness_score"],
            "combined": paper["combined_score"],
            "max_similarity_to_selected": paper.get("max_similarity_to_selected", 0.0),
        },
        "justification": _selection_justification(paper),
    }


def _selection_justification(paper: Dict[str, Any]) -> str:
    """Create a compact deterministic explanation for a selected paper."""
    return (
        "Selected for high topic fit "
        f"({paper['representativeness_score']:.3f}) and diversity "
        f"({paper.get('diversity_score', 1.0):.3f}) relative to other candidates."
    )


def _relevance_result_paper(paper: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Convert a relevance candidate into tool-facing metadata."""
    return {
        "rank": rank,
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "summary": paper["summary"],
        "authors": paper["authors"],
        "categories": paper["categories"],
        "published": paper.get("published"),
        "relevance_score": paper["relevance_score"],
        "justification": (
            "Ranked by semantic similarity between the query and the paper title/abstract."
        ),
    }


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
