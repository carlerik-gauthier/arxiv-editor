"""Phase 2 ChrisAgent implementation with topic and main-result tools."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from agents import Agent, Runner, function_tool, trace
from openai import OpenAI

from src_new.arxiv_fetcher import ArxivFetcher
from src_new.chris_agent_phase1 import (
    DEFAULT_FETCH_THRESHOLD,
    DEFAULT_MAX_RESULTS,
    CHRIS_SYSTEM_PROMPT,
    _extract_date_range,
    _extract_tool_parameters,
    _get_arxiv_categories,
    arxiv_fetcher_tool,
    check_paper_tool,
)
from src_new.topic_finder import compute_topics

CHRIS_CATEGORIES_PHASE2 = ("math.PR", "math.ST")
DEFAULT_MAX_TOPICS = 5
DEFAULT_PDF_OUTPUT_DIR = Path("data/pdfs")


@function_tool(name_override="find_topic_tool")
def find_topic_tool(
    csv_path: str,
    n_topics: int = DEFAULT_MAX_TOPICS,
    n_papers_per_topic: int = 3,
) -> Dict[str, Any]:
    """Extract top topics from a CSV file containing fetched paper metadata."""
    path = Path(csv_path)
    if not path.exists():
        return {
            "status": "failure",
            "reason": f"CSV file not found: {csv_path}",
            "topics": [],
        }

    n_topics = max(1, min(n_topics, DEFAULT_MAX_TOPICS))
    try:
        topics = compute_topics(
            path=str(path),
            n_topics=n_topics,
            n_papers_per_topic=max(1, n_papers_per_topic),
        )
    except Exception as exc:
        return {
            "status": "failure",
            "reason": f"Topic extraction failed: {exc}",
            "topics": [],
        }

    return {
        "status": "success",
        "reason": f"Extracted {len(topics)} topics from {csv_path}.",
        "topics": topics,
    }


@function_tool(name_override="extract_main_result_tool")
def extract_main_result_tool(
    arxiv_id: str,
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """Download a paper content from ArXiv and explain its main results."""
    fetcher = ArxivFetcher()
    output_dir = Path(os.getenv("ARXIV_PDF_OUTPUT_DIR", str(DEFAULT_PDF_OUTPUT_DIR)))

    try:
        markdown = fetcher.fetch_paper_markdown(
            paper_id=arxiv_id,
            output_dir=str(output_dir),
            force_redownload=False,
        )
    except Exception as exc:
        return {
            "status": "failure",
            "reason": f"Failed to download/extract paper {arxiv_id}: {exc}",
            "arxiv_id": arxiv_id,
            "main_result": None,
        }

    content = (markdown or "").strip()
    if not content:
        return {
            "status": "failure",
            "reason": f"No content extracted for paper {arxiv_id}.",
            "arxiv_id": arxiv_id,
            "main_result": None,
        }

    clipped = content[:max(1000, max_chars)]
    try:
        main_result = _extract_main_result_with_llm(arxiv_id=arxiv_id, content=clipped)
    except Exception as exc:
        return {
            "status": "failure",
            "reason": f"Main-result extraction failed for {arxiv_id}: {exc}",
            "arxiv_id": arxiv_id,
            "main_result": None,
        }

    return {
        "status": "success",
        "reason": f"Extracted main result for paper {arxiv_id}.",
        "arxiv_id": arxiv_id,
        "main_result": main_result,
    }


def _extract_main_result_with_llm(arxiv_id: str, content: str) -> str:
    """Use the configured OpenAI model to summarize the paper's core contribution."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    prompt = (
        f"Paper ID: {arxiv_id}\n"
        "You are a probability/statistics expert.\n"
        "From the paper content below, extract the main theorem or main result, then explain it in 1-3 concise bullet points its importance.\n"
        "Explain why. Be precise and avoid speculation.\n\n"
        f"Paper content:\n{content}"
    )
    response = client.responses.create(model=model, input=prompt, temperature=0.1)
    text = (response.output_text or "").strip()
    if not text:
        raise ValueError("Empty model output")
    return text


def build_chris_agent_phase2() -> Agent:
    """Create ChrisAgent for Phase 2 with topic and result-extraction tools."""
    return Agent(
        name="ChrisAgent",
        instructions=(
            f"{CHRIS_SYSTEM_PROMPT}\n"
            f"Allowed categories only: {', '.join(CHRIS_CATEGORIES_PHASE2)}.\n"
            "Use `check_paper_tool` to verify whether CSV data already exists for the requested date range.\n"
            "If data does not exist, call `arxiv_fetcher_tool`.\n"
            "Use `find_topic_tool` only after papers were fetched and a CSV path is available.\n"
            "Use `extract_main_result_tool` when full-paper main results are requested, passing arxiv_id."
        ),
        tools=[
            check_paper_tool,
            arxiv_fetcher_tool,
            find_topic_tool,
            extract_main_result_tool,
        ],
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )


def run_chris_agent_phase2(message: str) -> Dict[str, Any]:
    """Run one ChrisAgent phase-2 turn with SDK tracing enabled."""
    start_date, end_date = _extract_date_range(message)
    categories = _get_arxiv_categories(message)
    enriched_message = (
        f"{message}\n\n"
        f"Requested date range to use if needed: start_date={start_date}, end_date={end_date}.\n"
        f"Use categories={categories} when calling arxiv_fetcher_tool."
    )

    agent = build_chris_agent_phase2()
    with trace(
        "phase2-chris-agent-run",
        metadata={
            "agent": "ChrisAgent",
            "start_date": start_date,
            "end_date": end_date,
            "categories": ",".join(categories),
        },
        disabled=os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "0") == "1",
    ):
        result = Runner.run_sync(agent, enriched_message, max_turns=8)

    return {
        "reply": str(getattr(result, "final_output", "")),
        "tool_parameters": _extract_tool_parameters(getattr(result, "new_items", [])),
    }
