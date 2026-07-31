"""Topic modelling for persisted arXiv paper metadata.

The public :func:`compute_topics` function intentionally imports the expensive
BERTopic stack lazily. This keeps agent construction and unit tests fast, and
only requires the modelling dependencies when a topic extraction is requested.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import openai
import pandas as pd
import tiktoken
from dotenv import load_dotenv

load_dotenv(override=True)

SUMMARY_PROMPT = """
I have a topic that is described by the following keywords: [KEYWORDS]
In this topic, the following documents are a small but representative subset of all documents in the topic:
[DOCUMENTS]

Based on the information above, give a concise description of this topic in the following format:
topic: <description>
"""

TITLE_PROMPT = """
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"""

REQUIRED_COLUMNS = frozenset({"arxiv_id", "title", "summary"})


def compute_topics(
    path: str | Path,
    n_topics: int = 1,
    n_papers_per_topic: int = 3,
) -> list[dict[str, Any]]:
    """Cluster paper titles and abstracts into representative research topics.

    Args:
        path: CSV file created by the specialist paper-fetching workflow.
        n_topics: Maximum number of non-outlier clusters to return.
        n_papers_per_topic: Maximum number of representative papers per cluster.

    Returns:
        Topic dictionaries containing a label, description, paper count, and
        representative arXiv IDs and titles.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If metadata is invalid or does not provide enough papers to
            build meaningful clusters.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Paper metadata CSV not found: {csv_path}")

    topic_limit = _positive_count(n_topics, "n_topics")
    paper_limit = _positive_count(n_papers_per_topic, "n_papers_per_topic")
    data = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Paper metadata CSV is missing required columns: {missing}")

    data = data.dropna(subset=["arxiv_id", "title", "summary"]).copy()
    if len(data) < 2:
        raise ValueError("At least two papers with titles and summaries are required for topic extraction.")
    data["document"] = data["title"].astype(str) + " -- " + data["summary"].astype(str)
    documents = data["document"].tolist()

    topic_model = _build_topic_model()
    labels, probabilities = topic_model.fit_transform(documents)
    data["topic"] = labels
    data["probability"] = _confidence_scores(probabilities, len(data))

    topic_info = topic_model.get_topic_info()
    selected_topics = topic_info[topic_info["Topic"] != -1].nlargest(topic_limit, "Count").copy()
    if selected_topics.empty:
        return []
    selected_topics.rename(columns={"Topic": "topic", "Count": "nb_papers"}, inplace=True)

    representatives = _representative_papers(data, selected_topics["topic"].tolist(), paper_limit)
    results: list[dict[str, Any]] = []
    for _, row in selected_topics.iterrows():
        topic_id = row["topic"]
        papers = representatives.get(topic_id, [])
        if not papers:
            continue
        results.append(
            {
                "topic_title": _topic_text(row.get("topic_title_"), fallback=f"Topic {topic_id}"),
                "nb_papers": int(row["nb_papers"]),
                "topic_description": _topic_text(
                    row.get("topic_summary_"),
                    fallback="A cluster of closely related arXiv papers.",
                ),
                "representative_papers_arxiv_id": [paper["arxiv_id"] for paper in papers],
                "representative_papers_title": [paper["title"] for paper in papers],
            }
        )
    return results


def _build_topic_model() -> Any:
    """Build BERTopic with semantic embeddings and OpenAI label generators.

    Returns:
        Any: Configured BERTopic model ready to fit paper documents.

    Raises:
        ImportError: If optional BERTopic or embedding dependencies are missing.
        Exception: If the configured OpenAI or model dependencies cannot be
            initialized.
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.representation import OpenAI as OpenAIRepresentation
    from sentence_transformers import SentenceTransformer

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    summary_representation = [
        MaximalMarginalRelevance(diversity=0.3),
        OpenAIRepresentation(
            client,
            model=os.getenv("OPENAI_TOPIC_MODEL", "gpt-4o-mini"),
            prompt=SUMMARY_PROMPT,
            chat=True,
            nr_docs=10,
            doc_length=2000,
            diversity=0.3,
            tokenizer=tokenizer,
        ),
    ]
    title_representation = [
        MaximalMarginalRelevance(diversity=0.3),
        OpenAIRepresentation(
            client,
            model=os.getenv("OPENAI_TOPIC_MODEL", "gpt-4o-mini"),
            prompt=TITLE_PROMPT,
            chat=True,
            nr_docs=10,
            doc_length=200,
            diversity=0.3,
            tokenizer=tokenizer,
        ),
    ]
    return BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        representation_model={
            "Main": KeyBERTInspired(),
            "topic_summary_": summary_representation,
            "topic_title_": title_representation,
        },
    )


def _positive_count(value: int, parameter_name: str) -> int:
    """Validate and return a strictly positive integer parameter.

    Args:
        value: Candidate count to validate.
        parameter_name: Parameter name used in the validation message.

    Returns:
        int: The validated positive integer.

    Raises:
        ValueError: If ``value`` is a boolean, non-integer, or less than one.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{parameter_name} must be a positive integer.")
    return value


def _confidence_scores(probabilities: Any, expected_count: int) -> list[float]:
    """Convert BERTopic confidence output into one sortable score per paper.

    Args:
        probabilities: BERTopic confidence values, potentially scalar-like or
            matrix-like per paper.
        expected_count: Number of scores required to align with paper rows.

    Returns:
        list[float]: One numeric score per expected paper, or zero scores when
        the model output cannot be aligned.
    """
    if probabilities is None:
        return [0.0] * expected_count
    scores: list[float] = []
    for probability in probabilities:
        if hasattr(probability, "max"):
            probability = probability.max()
        try:
            scores.append(float(probability))
        except (TypeError, ValueError):
            scores.append(0.0)
    return scores if len(scores) == expected_count else [0.0] * expected_count


def _representative_papers(
    data: pd.DataFrame,
    topic_ids: list[int],
    paper_limit: int,
) -> dict[int, list[dict[str, str]]]:
    """Select the highest-confidence representative papers per topic.

    Args:
        data: Paper dataframe containing topic, probability, ID, and title.
        topic_ids: Selected non-outlier topic identifiers.
        paper_limit: Maximum representative papers to retain for each topic.

    Returns:
        dict[int, list[dict[str, str]]]: Topic IDs mapped to ordered paper ID
        and title dictionaries.
    """
    candidates = data[data["topic"].isin(topic_ids)].sort_values(
        ["topic", "probability"], ascending=[True, False]
    )
    candidates = candidates.groupby("topic", sort=False).head(paper_limit)
    grouped: dict[int, list[dict[str, str]]] = {}
    for topic_id, papers in candidates.groupby("topic", sort=False):
        grouped[int(topic_id)] = [
            {"arxiv_id": str(paper.arxiv_id), "title": str(paper.title)}
            for paper in papers.itertuples(index=False)
        ]
    return grouped


def _topic_text(value: Any, fallback: str) -> str:
    """Normalize BERTopic label formats into a readable non-empty string.

    Args:
        value: Label value returned by BERTopic, possibly a sequence or empty.
        fallback: Text returned when ``value`` has no usable content.

    Returns:
        str: Trimmed label text or ``fallback``.
    """
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    text = str(value or "").strip()
    return text or fallback
